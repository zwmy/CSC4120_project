import networkx as nx
import pulp
from collections import defaultdict
from student_utils import *

def ptp_solver(G: nx.DiGraph, H: list, alpha: float):
    """
    PTP solver that automatically selects method based on problem size.
    
    Parameters:
        G (nx.DiGraph): A NetworkX graph representing the city.
        H (list): A list of home nodes.
        alpha (float): The coefficient for calculating cost.
        
    Returns:
        tuple: A tuple containing:
            - tour (list): A list of nodes traversed by your car.
            - pick_up_locs_dict (dict): A dictionary where:
                - Keys are pick-up locations.
                - Values are lists containing friends' home nodes who get picked up
                  at that specific pick-up location.
    """
    tour1, pick_up_locs_dict1 = ptp_solver_heuristic(G, H, alpha)
    _,drive_cost1, walk_cost1 = analyze_solution(G, H, alpha, tour1, pick_up_locs_dict1)
    tour2, pick_up_locs_dict2 =  ptp_solver_ilp(G, H, alpha)
    _,drive_cost2, walk_cost2 = analyze_solution(G, H, alpha, tour2, pick_up_locs_dict2)

    if drive_cost1 + walk_cost1 < drive_cost2 + walk_cost2:
        return tour1, pick_up_locs_dict1
    else:
        return tour2, pick_up_locs_dict2
    #return tour1, pick_up_locs_dict1

def ptp_solver_ilp(G:nx.DiGraph, H:list, alpha:float):
    nodes = list(G.nodes)
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for i, node in enumerate(nodes)}
    home_idx = node_to_idx[0]
    
    dist_matrix = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    path_dict = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
    
    prob = pulp.LpProblem("PTP_Optimization", pulp.LpMinimize)
    
    x = pulp.LpVariable.dicts("x", (range(n), range(n)), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", range(n), cat=pulp.LpBinary)
    z = pulp.LpVariable.dicts("z", (range(len(H)), range(n)), cat=pulp.LpBinary)
    u = pulp.LpVariable.dicts("u", range(n), lowBound=0, upBound=n, cat=pulp.LpContinuous)

    drive_cost = pulp.lpSum([
        alpha * dist_matrix[idx_to_node[i]][idx_to_node[j]] * x[i][j] 
        for i in range(n) for j in range(n) 
        if idx_to_node[j] in dist_matrix[idx_to_node[i]] and i != j
    ])
    
    walk_cost = pulp.lpSum([
        dist_matrix[H[m]][idx_to_node[i]] * z[m][i] 
        for m in range(len(H)) for i in range(n) 
        if idx_to_node[i] in dist_matrix[H[m]]
    ])
    
    prob += drive_cost + walk_cost

    for i in range(n):
        prob += pulp.lpSum([x[i][j] for j in range(n) if i != j]) == y[i]
        prob += pulp.lpSum([x[j][i] for j in range(n) if i != j]) == y[i]
    
    prob += y[home_idx] == 1
    
    for m in range(len(H)):
        prob += pulp.lpSum([z[m][i] for i in range(n)]) == 1
        friend_home = H[m]
        valid_nodes = set(G.successors(friend_home)) | set(G.predecessors(friend_home)) | {friend_home}
        for i in range(n):
            node_id = idx_to_node[i]
            if node_id not in valid_nodes:
                prob += z[m][i] == 0
            prob += z[m][i] <= y[i]

    for i in range(n):
        for j in range(n):
            if i != j and i != home_idx and j != home_idx:
                prob += u[i] - u[j] + n * x[i][j] <= n - 1

    try:
        status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=1200))
    except:
        return [0], {0: H}

    if status != pulp.LpStatusOptimal:
        return [0], {0: H}

    logical_edges = []
    for i in range(n):
        for j in range(n):
            val = pulp.value(x[i][j])
            if val is not None and val > 0.5:
                logical_edges.append((i, j))
    
    if not logical_edges:
        return [0], {0: H}

    logical_tour = [home_idx]
    curr = home_idx
    while True:
        next_node = None
        for idx, (u_e, v_e) in enumerate(logical_edges):
            if u_e == curr:
                next_node = v_e
                logical_edges.pop(idx)
                break
        if next_node is None or next_node == home_idx:
            break
        logical_tour.append(next_node)
        curr = next_node
    logical_tour.append(home_idx)

    tour = []
    for k in range(len(logical_tour) - 1):
        u_node = idx_to_node[logical_tour[k]]
        v_node = idx_to_node[logical_tour[k+1]]
        path_segment = path_dict[u_node][v_node]
        tour.extend(path_segment[:-1])
    tour.append(0)

    pick_up_locs_dict = {}
    for m in range(len(H)):
        for i in range(n):
            z_val = pulp.value(z[m][i])
            if z_val is not None and z_val > 0.5:
                p_node = idx_to_node[i]
                if p_node not in pick_up_locs_dict:
                    pick_up_locs_dict[p_node] = []
                pick_up_locs_dict[p_node].append(H[m])
                break

    return tour, pick_up_locs_dict


def ptp_solver_heuristic(G: nx.DiGraph, H: list, alpha: float):

    G_undirected = G.to_undirected()
    
    dist = dict(nx.all_pairs_dijkstra_path_length(G_undirected, weight='weight'))
    
    candidate_locations = {}
    for home in H:
        candidate_locations[home] = list(set([home] + list(G_undirected.neighbors(home))))
    
    best_tour = None
    best_cost = float('inf')
    
    strategies = [
        generate_initial_tour_strategy1,
        generate_initial_tour_strategy2,
        generate_initial_tour_strategy3,
    ]
    
    for strategy_idx, strategy in enumerate(strategies):
        
        initial_tour = strategy(H, candidate_locations, dist)
        
        improved_tour = insert_delete_heuristic(initial_tour, H, candidate_locations, dist, alpha)
        improved_tour = two_opt(improved_tour, dist)
        
        cost, infeasibility, _ = calculate_cost_and_feasibility(improved_tour, H, candidate_locations, dist, alpha)
        
        if infeasibility > 0:
            improved_tour = repair_infeasibility(improved_tour, H, candidate_locations, dist)
            cost, infeasibility, _ = calculate_cost_and_feasibility(improved_tour, H, candidate_locations, dist, alpha)
        
        if infeasibility == 0 and cost < best_cost:
            best_tour = improved_tour
            best_cost = cost
    
    if best_tour is None:
        best_tour = greedy_construction(H, candidate_locations, dist, alpha)
    
    best_tour = two_opt(best_tour, dist)
    
    best_tour = validate_and_repair_tour(best_tour, G_undirected)
    
    _, _, final_pickup_dict = calculate_cost_and_feasibility(best_tour, H, candidate_locations, dist, alpha)
    
    return best_tour, final_pickup_dict

def generate_initial_tour_strategy1(H, candidate_locations, dist):
    nodes_to_visit = set([0])
    for home in H:
        nodes_to_visit.update(candidate_locations[home])
    
    return nearest_neighbor_tsp(list(nodes_to_visit), dist)

def generate_initial_tour_strategy2(H, candidate_locations, dist):
    tour = [0, 0]
    
    for home in H:
        best_insertion = None
        best_insertion_cost = float('inf')
        
        for candidate in candidate_locations[home]:
            for pos in range(1, len(tour)):
                new_tour = tour[:pos] + [candidate] + tour[pos:]
                insertion_cost = calculate_tour_increase(tour, new_tour, dist)
                
                if insertion_cost < best_insertion_cost:
                    best_insertion_cost = insertion_cost
                    best_insertion = (candidate, pos)
        
        if best_insertion:
            candidate, pos = best_insertion
            tour = tour[:pos] + [candidate] + tour[pos:]
    
    return tour

def generate_initial_tour_strategy3(H, candidate_locations, dist):
    nodes = set([0])
    for home in H:
        nodes.update(candidate_locations[home])
    nodes = list(nodes)
    
    if len(nodes) <= 3:
        return nearest_neighbor_tsp(nodes, dist)
    
    complete_graph = nx.Graph()
    for i in nodes:
        for j in nodes:
            if i != j:
                complete_graph.add_edge(i, j, weight=dist[i][j])
    
    mst = nx.minimum_spanning_tree(complete_graph)
    
    multigraph = nx.MultiGraph(mst)
    
    odd_vertices = [v for v, degree in mst.degree() if degree % 2 == 1]
    
    matching = greedy_matching(odd_vertices, dist)
    
    for u, v in matching:
        multigraph.add_edge(u, v, weight=dist[u][v])
    
    try:
        eulerian_circuit = list(nx.eulerian_circuit(multigraph, source=0))
    except:
        try:
            eulerian_circuit = list(nx.eulerian_circuit(multigraph))
        except:
            return nearest_neighbor_tsp(nodes, dist)
    
    tour = [0]
    visited = {0}
    
    for u, v in eulerian_circuit:
        if v not in visited:
            tour.append(v)
            visited.add(v)
    
    tour.append(0)
    return tour

def insert_delete_heuristic(tour, H, candidate_locations, dist, alpha):
    n_nodes = len(dist)
    current_tour = tour[:]
    current_cost, current_infeasibility, _ = calculate_cost_and_feasibility(
        current_tour, H, candidate_locations, dist, alpha
    )
    
    improvement = True
    iteration = 0
    max_iterations = min(100, n_nodes * 5)
    
    while improvement and iteration < max_iterations:
        iteration += 1
        improvement = False
        
        best_tour = current_tour
        best_cost = current_cost
        best_infeasibility = current_infeasibility
        
        nodes_in_tour = set(current_tour)
        for node in nodes_in_tour - {0}:
            indices = [i for i, x in enumerate(current_tour) if x == node]
            for idx in indices:
                if 0 < idx < len(current_tour) - 1:
                    new_tour = current_tour[:idx] + current_tour[idx+1:]
                    new_tour = two_opt(new_tour, dist)
                    
                    new_cost, new_infeasibility, _ = calculate_cost_and_feasibility(
                        new_tour, H, candidate_locations, dist, alpha
                    )
                    
                    if is_improvement(current_infeasibility, current_cost,
                                      new_infeasibility, new_cost):
                        if (new_infeasibility < best_infeasibility or 
                            (new_infeasibility == best_infeasibility and new_cost < best_cost)):
                            best_tour = new_tour
                            best_cost = new_cost
                            best_infeasibility = new_infeasibility
                            improvement = True
        
        all_nodes = set(range(n_nodes))
        nodes_not_in_tour = all_nodes - nodes_in_tour
        
        for node in nodes_not_in_tour:
            for pos in range(1, len(current_tour)):
                new_tour = current_tour[:pos] + [node] + current_tour[pos:]
                new_tour = two_opt(new_tour, dist)
                
                new_cost, new_infeasibility, _ = calculate_cost_and_feasibility(new_tour, H, candidate_locations, dist, alpha)
                
                if is_improvement(current_infeasibility, current_cost,
                                  new_infeasibility, new_cost):
                    if (new_infeasibility < best_infeasibility or 
                        (new_infeasibility == best_infeasibility and new_cost < best_cost)):
                        best_tour = new_tour
                        best_cost = new_cost
                        best_infeasibility = new_infeasibility
                        improvement = True
        
        if improvement:
            current_tour = best_tour
            current_cost = best_cost
            current_infeasibility = best_infeasibility
    
    return current_tour

def calculate_cost_and_feasibility(tour, H, candidate_locations, dist, alpha):
    driving_distance = 0
    for i in range(len(tour)-1):
        driving_distance += dist[tour[i]][tour[i+1]]
    
    driving_cost = alpha * driving_distance
    
    walking_cost = 0
    infeasibility_count = 0
    pickup_dict = {}
    
    tour_set = set(tour)
    
    for home in H:
        min_walking_distance = float('inf')
        best_pickup = None
        
        for candidate in candidate_locations[home]:
            if candidate in tour_set:
                walking_distance = dist[home][candidate]
                if walking_distance < min_walking_distance:
                    min_walking_distance = walking_distance
                    best_pickup = candidate
        
        if best_pickup is None:
            walking_cost += float('inf')
            infeasibility_count += 1
        else:
            walking_cost += min_walking_distance
            if best_pickup not in pickup_dict:
                pickup_dict[best_pickup] = []
            pickup_dict[best_pickup].append(home)
    
    total_cost = driving_cost + walking_cost
    return total_cost, infeasibility_count, pickup_dict

def is_improvement(current_infeasibility, current_cost, new_infeasibility, new_cost):
    if current_infeasibility == 0:
        return new_infeasibility == 0 and new_cost < current_cost
    else:
        return (new_infeasibility < current_infeasibility or 
                (new_infeasibility == current_infeasibility and new_cost < current_cost))

def two_opt(tour, dist):
    n = len(tour)
    best_tour = tour
    best_length = tour_length(tour, dist)
    improved = True
    
    while improved:
        improved = False
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                new_tour = best_tour[:i] + best_tour[i:j+1][::-1] + best_tour[j+1:]
                new_length = tour_length(new_tour, dist)
                
                if new_length < best_length:
                    best_tour = new_tour
                    best_length = new_length
                    improved = True
                    break
            if improved:
                break
    
    return best_tour

def tour_length(tour, dist):
    length = 0
    for i in range(len(tour)-1):
        length += dist[tour[i]][tour[i+1]]
    return length

def nearest_neighbor_tsp(nodes, dist, start=0):
    unvisited = set(nodes)
    if start not in unvisited:
        unvisited.add(start)
    else:
        unvisited.remove(start)
    
    tour = [start]
    current = start
    
    while unvisited:
        next_node = min(unvisited, key=lambda x: dist[current][x])
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    
    tour.append(start)
    return tour

def greedy_matching(nodes, dist):
    edges = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            u, v = nodes[i], nodes[j]
            edges.append((dist[u][v], u, v))
    
    edges.sort()
    
    matching = []
    matched = set()
    
    for weight, u, v in edges:
        if u not in matched and v not in matched:
            matching.append((u, v))
            matched.add(u)
            matched.add(v)
    
    return matching

def greedy_construction(H, candidate_locations, dist, alpha):
    n_nodes = len(dist)
    tour = [0, 0]
    
    friends_to_cover = set(H)
    
    while friends_to_cover:
        best_improvement = float('-inf')
        best_node = None
        best_position = None
        
        for node in range(n_nodes):
            if node not in tour:
                for pos in range(1, len(tour)):
                    new_tour = tour[:pos] + [node] + tour[pos:]
                    
                    improvement = calculate_tour_improvement(tour, new_tour, H, candidate_locations, dist, alpha)
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_node = node
                        best_position = pos
        
        if best_improvement > 0 and best_node is not None:
            tour = tour[:best_position] + [best_node] + tour[best_position:]
            
            tour_set = set(tour)
            friends_to_cover = {
                home for home in H 
                if not any(candidate in tour_set for candidate in candidate_locations[home])
            }
        else:
            break
    
    if tour[0] != 0:
        tour = [0] + tour
    if tour[-1] != 0:
        tour.append(0)
    
    return tour

def calculate_tour_improvement(old_tour, new_tour, H, candidate_locations, dist, alpha):
    old_cost, old_infeasibility, _ = calculate_cost_and_feasibility(old_tour, H, candidate_locations, dist, alpha)
    
    new_cost, new_infeasibility, _ = calculate_cost_and_feasibility(new_tour, H, candidate_locations, dist, alpha)
    
    improvement = old_cost - new_cost
    
    if new_infeasibility < old_infeasibility:
        improvement += 1000 * (old_infeasibility - new_infeasibility)
    
    return improvement

def repair_infeasibility(tour, H, candidate_locations, dist):
    tour_set = set(tour)
    
    for home in H:
        has_candidate_in_tour = False
        for candidate in candidate_locations[home]:
            if candidate in tour_set:
                has_candidate_in_tour = True
                break
        
        if not has_candidate_in_tour:
            best_candidate = min(candidate_locations[home], key=lambda c: dist[home][c])
            
            best_pos = None
            best_insertion_cost = float('inf')
            
            for pos in range(1, len(tour)):
                new_tour = tour[:pos] + [best_candidate] + tour[pos:]
                insertion_cost = calculate_tour_increase(tour, new_tour, dist)
                
                if insertion_cost < best_insertion_cost:
                    best_insertion_cost = insertion_cost
                    best_pos = pos
            
            if best_pos is not None:
                tour = tour[:best_pos] + [best_candidate] + tour[best_pos:]
                tour_set.add(best_candidate)
    
    return tour

def calculate_tour_increase(old_tour, new_tour, dist):
    old_length = tour_length(old_tour, dist)
    new_length = tour_length(new_tour, dist)
    return new_length - old_length

def validate_and_repair_tour(tour, G_undirected):
    if not tour:
        return [0, 0]
    
    if tour[0] != 0:
        tour = [0] + tour
    if tour[-1] != 0:
        tour.append(0)
    
    repaired_tour = [tour[0]]
    for i in range(1, len(tour)):
        u = repaired_tour[-1]
        v = tour[i]
        
        if G_undirected.has_edge(u, v):
            repaired_tour.append(v)
        else:
            try:
                path = nx.shortest_path(G_undirected, u, v, weight='weight')
                repaired_tour.extend(path[1:])
            except:
                repaired_tour.append(v)
    
    simplified_tour = []
    for node in repaired_tour:
        if not simplified_tour or node != simplified_tour[-1]:
            simplified_tour.append(node)
    
    return simplified_tour

if __name__ == "__main__":
    pass