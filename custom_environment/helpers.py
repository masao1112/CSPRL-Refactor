import json
import pickle
import math
import osmnx as ox
import numpy as np
import networkx as nx
from math import sin, cos, sqrt, atan2, radians, ceil

"""
Utility model and help functions.
"""


def prepare_graph(my_graph_file, my_node_file):
    """
    loads graph and nodes prepared in load_graph.py
    """
    my_graph = ox.load_graphml(my_graph_file)
    with open(my_node_file, "r") as file:
        my_node_list = eval(file.readline())
    return my_graph, my_node_list


def cost_single(my_node, my_station, my_node_dict, my_cost_dict, graph):
    """
    calculate the social cost for one station
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    node_id, station_id = my_node[0], s_pos[0]
    # check if distance has to be calculated
    if station_id in my_node_dict[node_id]:
        distance = my_node_dict[node_id][station_id]
    else:
        distance = calculate_distance(s_pos, my_node, graph)
        my_node_dict[node_id][station_id] = distance
    # check if cost has to be calculated
    if node_id not in my_cost_dict:
        my_cost_dict[node_id] = {}
    station_signature = (
        tuple(np.asarray(s_x).tolist()),
        float(s_dict.get("W_s", 0.0)),
        float(s_dict.get("service rate", 0.0)),
    )
    cached_entry = my_cost_dict[node_id].get(station_id)
    if isinstance(cached_entry, dict) and cached_entry.get("state") == station_signature:
        node_cost = cached_entry["cost"]
    else:
        cost_travel = alpha * (distance / VELOCITY) * (1 + weak_demand(my_node)) # demand as traffic density factor
        cost_boring = (1 - alpha) * (s_dict["W_s"] + 1 / (s_dict["service rate"] + eps))
        node_cost = cost_travel + cost_boring
        my_cost_dict[node_id][station_id] = {
            "state": station_signature,
            "cost": node_cost
        }
    return node_cost, my_node_dict, my_cost_dict


def station_seeking(my_plan, my_node_list, my_node_dict, my_cost_dict, graph):
    """
    output station assignment: Each node gets assigned the charging station with minimal social cost
    """
    for node in my_node_list:
        old_station_id = node[1].get("charging station")
        cost_list = []
        for station in my_plan:
            node_cost, my_node_dict, my_cost_dict = cost_single(node, station, my_node_dict, my_cost_dict, graph)
            cost_list.append(node_cost)
        costminindex = np.argmin(cost_list)
        chosen_station = my_plan[costminindex]
        s_pos = chosen_station[0]
        node[1]["charging station"] = s_pos[0]
        node[1]["distance"] = my_node_dict[node[0]][s_pos[0]]
        # Update D_s and W_s for the newly chosen station
        total_number_EVs(chosen_station, my_node_list)
        avg_waiting(chosen_station)
        # If the node moved from a different station, update the old station too
        if old_station_id is not None and old_station_id != s_pos[0]:
            for station in my_plan:
                if station[0][0] == old_station_id:
                    total_number_EVs(station, my_node_list)
                    avg_waiting(station)
                    break
    return my_node_list, my_node_dict, my_cost_dict


def calculate_distance(s_pos, my_node, graph):
    """
    Calculates distance between two nodes using the precomputed distance matrix.
    Falls back to haversine if matrix lookup fails.
    """
    try:
        # s_pos[0] and my_node[0] là the OSM node IDs
        u = s_pos[0]
        v = my_node[0]
        distance = nx.shortest_path_length(graph, u, v, weight='length')
        return distance / 1000.0
    except (KeyError, IndexError):
        # Fallback if node not found in matrix
        # print(f"Matrix lookup failed for {u} -> {v}, falling back to Haversine")
        pass
    except Exception as e:
        # print(f"Distance loader error: {e}")
        pass
    # if not available, use haversine instead
    return haversine(s_pos, my_node)


################################################################################################
def installment_fee(my_station):
    """
    returns cost to install the respective chargers at that position
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    n_chargers = np.sum(s_x)
    charger_cost = np.sum(INSTALL_FEE * s_x)
    fee = evs_parking_area * n_chargers * s_pos[1]['land_price'] + charger_cost
    s_dict["fee"] = fee  # [fee] = €t
    return my_station


def charging_capability(my_station):
    """
    returns the summed up charging capability of the CS
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    total_capacity = np.sum(CHARGING_POWER * s_x)
    s_dict["capability"] = total_capacity / 1000.0  # [capability] = MW
    return my_station


def weak_demand(my_node):
    return my_node[1]["demand"] * (1 - 0.1 * my_node[1]["private_cs"])

def dynamic_demand(my_node, my_plan, scaling_factor=None, distance_decay_factor=None):
    """Demand at a node after nearby installed capacity has absorbed part of it.

    scaling_factor (eta) and distance_decay_factor (beta) fall back to the
    module-level DEMAND_ETA / DEMAND_BETA, so an ablation can rebind them once
    before the env is built instead of threading them through every call site.
    """
    scaling_factor = DEMAND_ETA if scaling_factor is None else scaling_factor
    distance_decay_factor = DEMAND_BETA if distance_decay_factor is None else distance_decay_factor
    power_factor = 0
    base_demand = weak_demand(my_node)
    for station in my_plan:
        s_pos, s_x, s_dict = station[0], station[1], station[2]
        s_r = s_dict["radius"]
        s_cap = s_dict["capability"]
        distance = haversine(s_pos, my_node)
        if distance < s_r:
            power_factor += s_cap * np.exp(-distance_decay_factor * distance)
    power_factor *= -scaling_factor
    new_demand = base_demand * np.exp(power_factor)

    return new_demand

def influence_radius(my_station):
    """
    gives the radius of the nodes whose charging demand the CS could satisfy
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    total_capacity = s_dict["capability"]  # [capability] = MW
    # Logistic saturation of installed capacity, r(s) = R_max / (1 + e^{-C/C_0}).
    # C_0 has to sit on the scale of a real station. The divisor used to be
    # 1000 * capacity_unit against a capability already expressed in MW, i.e.
    # C_0 = 1 GW, which pinned the whole fleet to r = 0.500 km: an 11 kW site and
    # a 25 MW site differed by 6 metres. Coverage benefit, fairness and the
    # demand-attenuation radius were therefore blind to how big a station was.
    # At C_0 = 1 MW the curve spans the range the fleet actually occupies --
    # 0.50 km at 11 kW, 0.56 at 240 kW, 0.95 at 3 MW, ~1.00 above 10 MW.
    radius_s = RADIUS_MAX / (1 + np.exp(-total_capacity / RADIUS_CAPACITY_SCALE))
    s_dict["radius"] = radius_s  # [radius] = km
    return my_station


_haversine_cache = {}

def haversine(s_pos, my_node):
    """
    yields the approximate distance of two GPS points, middle computational cost
    """
    try:
        key = (s_pos[0], my_node[0])
        if key in _haversine_cache:
            return _haversine_cache[key]
    except Exception:
        pass

    lon1, lat1 = s_pos[1]['x'], s_pos[1]['y']
    R_earth = 6372800  # approximate radius of earth. [R_earth] = m
    lon2, lat2 = my_node[1]['x'], my_node[1]['y']
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R_earth * c  # [distance] = m
    if distance < 0.1:  # to avoid ZeroDivisionError
        distance = 0.1
    distance = distance / 1000.0
    
    try:
        _haversine_cache[key] = distance
        _haversine_cache[(my_node[0], s_pos[0])] = distance
    except Exception:
        pass
        
    return distance


def station_coverage(my_station, my_node_list):
    """yields the number of nodes within a station influential radius (raw count - use for other purposes)"""
    node_counts = 0
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    radius_s = s_dict["radius"]
    for node in my_node_list:
        distance = haversine(s_pos, node)
        if distance < radius_s:
            node_counts += 1
    return node_counts


def station_benefit(my_station, my_node_list):
    """
    Yields the benefit of nodes within a station's influential radius.
    More nodes covered = more benefit (no diminishing returns here).
    This is different from node_coverage which has diminishing returns
    due to redundancy (multiple stations covering one node).
    For fair comparison with node_coverage, we normalize by total possible nodes.
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    radius_s = s_dict["radius"]

    # Count nodes within radius
    covered_nodes = 0
    for node in my_node_list:
        distance = haversine(s_pos, node)
        if distance < radius_s:
            covered_nodes += 1

    # Normalize to 0-1 range based on total nodes, then scale to be comparable to node_coverage
    # (which typically ranges from ~1-4 with diminishing returns)
    normalized_coverage = (covered_nodes / len(my_node_list)) * 10  # scale factor to match node_coverage range

    return normalized_coverage


def node_benefit(my_plan, my_node):
    """
    yields the number of station nodes which cover a given node
    """
    station_counts, diminishing_benefit = 0, 0
    priv_CS = my_node[1]["private_cs"]
    for my_station in my_plan:
        s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
        radius_s = s_dict["radius"]
        distance = haversine(s_pos, my_node)
        if distance <= radius_s:
            station_counts += 1
    my_node[1]['n_stations'] = station_counts
    for ith in range(station_counts):
        diminishing_benefit += 1 / (
                    ith + 1)  # diminishing return, as more stations cover node v, the higher the benefit
    single_benefit = diminishing_benefit * (1 - 0.1 * priv_CS)
    return single_benefit


def total_number_EVs(my_station, my_node_list):
    """
    yields total number of EVs coming to S in a unit time interval for charging
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    # D_s = sum([1 / my_node[1]["distance"] * weak_demand(my_node) if my_node[1]["charging station"] == s_pos[0]
    #            else 0 for my_node in my_node_list])
    D_s = sum([ceil(ev_per_capita * my_node[1]['pop']) if my_node[1]["charging station"] == s_pos[0]
               else 0 for my_node in my_node_list])
    s_dict["D_s"] = D_s  # dimensionless
    return my_station


def service_rate(my_station):
    """
    returns how many cars can be served within one hour
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    n_chargers = int(np.sum(s_x))
    mu = s_dict["capability"] * 1000 / BATTERY  # aggregate rate, [service rate] = 1/h
    s_dict["service rate"] = mu
    # Per-charger rate. The aggregate capacity is split evenly over the chargers,
    # so c * mu_1 == mu and the station's total throughput is exactly what the
    # single-server model assumed; only the discipline differs. mu_1 is the rate
    # a single driver actually sees, and the M/M/c/N queue runs on it.
    s_dict["c_servers"] = n_chargers
    s_dict["service rate 1"] = mu / n_chargers if n_chargers > 0 else 0.0
    return my_station


_queue_cache = {}


def _mmcn_solve(a, c, N):
    """
    Blocking probability and mean system size of an M/M/c/N queue at offered load
    a (Erlang), c servers, system capacity N.

    The birth-death ratios r_n = r_{n-1} * a / min(n, c) are accumulated in log
    space and normalised once. That is exact at every traffic intensity -- there
    is no rho < 1 / rho = 1 / rho > 1 case split -- and it cannot overflow the
    way a**N does (a reaches ~3e3 for a station of slow chargers under load).
    """
    key = (a, c, N)
    cached = _queue_cache.get(key)
    if cached is not None:
        return cached

    n = np.arange(1, N + 1)
    log_r = np.concatenate(([0.0], np.cumsum(np.log(a) - np.log(np.minimum(n, c)))))
    r = np.exp(log_r - log_r.max())
    P = r / r.sum()
    PN = float(P[-1])
    Ls = float(np.dot(np.arange(N + 1), P))

    if len(_queue_cache) > 200000:
        _queue_cache.clear()
    _queue_cache[key] = (PN, Ls)
    return PN, Ls


def avg_waiting(my_station, kappa=None, eps=1e-9):
    """
    Expected time in the system (queueing + charging) at one station, modelled as
    a finite-capacity multi-server M/M/c/N queue.

    Every charger is an explicit server: c = n(s) chargers each at rate
    mu_1 = mu / n(s). The aggregate rate mu -- and with it the traffic intensity
    rho = D / mu -- is exactly the one the single-server model used, so only the
    service discipline changes. That change matters: pooling n chargers into one
    server of rate mu lets a single vehicle draw the full station power, which
    credited an 8x30 kW site with charging a car in 21 min instead of 170.

    The system capacity is structural, N = (1 + kappa) * c, i.e. kappa waiting
    bays per charger. It deliberately does NOT depend on demand. A demand-sized
    buffer (the old N_eff = max(1, min(N, ceil(D)))) is not an M/M/c/N at all:
    it makes the blocking probability non-monotone in D -- more demand enlarges
    the buffer, which lowers blocking -- and it ties the queue's structure to the
    node assignment that the queue itself decides. kappa = 0 gives the Erlang
    loss system M/M/c/c; large kappa approaches the unbounded M/M/c.

    Writes into the station dict: W_s (mean sojourn time, h), P_N (blocking
    probability) and unserved (turned-away demand D * P_N, veh/h). Under
    saturation D * P_N -> D - mu, the plain capacity shortfall, which is
    independent of kappa -- see unserved_demand().
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    kappa = WAIT_BAY_RATIO if kappa is None else kappa

    ar = s_dict.get("D_s", 0.0)  # arrival rate D(s), [1/h]
    c = int(np.sum(s_x))         # servers = chargers

    if c <= 0:
        # No charger installed: nothing can be served, every arrival is turned away.
        s_dict["N_sys"] = 0
        s_dict["P_N"] = 1.0 if ar > 0 else 0.0
        s_dict["unserved"] = ar
        s_dict["W_s"] = my_inf if ar > 0 else 0.0
        return my_station

    if ar <= 0.0:
        # No demand routed here: no queue, and no waiting cost either since
        # waiting(p) weights W_s by D(s).
        s_dict["N_sys"] = c + int(ceil(kappa * c))
        s_dict["P_N"] = 0.0
        s_dict["unserved"] = 0.0
        s_dict["W_s"] = 0.0
        return my_station

    mu = max(s_dict.get("service rate", 0.0), eps)  # aggregate rate
    mu_1 = mu / c                                   # per-charger rate
    N_sys = c + int(ceil(kappa * c))                # system capacity

    a = ar / mu_1                                   # offered load [Erlang] = c * rho
    PN, Ls = _mmcn_solve(a, c, N_sys)

    lambda_eff = ar * (1.0 - PN)  # only admitted traffic is served
    s_dict["N_sys"] = N_sys
    s_dict["P_N"] = PN
    s_dict["unserved"] = ar * PN
    s_dict["W_s"] = Ls / (lambda_eff + eps)  # Little's law

    return my_station


def s_dictionnary(my_station, my_node_list):
    """
    returns the dictionnary for the station
    """
    my_station = installment_fee(my_station)
    my_station = charging_capability(my_station)
    my_station = influence_radius(my_station)
    my_station = total_number_EVs(my_station, my_node_list)
    my_station = service_rate(my_station)
    my_station = avg_waiting(my_station)
    return my_station


# SCORE over the plan #####################################################################



def social_benefit(my_plan, my_node_list):
    """
    Returns the social benefit of the charging plan.
    Combines two balanced components with fair weighting:
    1. Node coverage: how many stations cover each node (with diminishing returns)
    2. Station coverage: how many nodes each station covers (with diminishing returns)
    Both components use diminishing returns to encourage balanced, distributed placement.
    """
    if not my_plan:
        return 0

    # Component 1: Node perspective - how well are nodes covered by stations
    # (how many charging stations can each node access)
    node_benefit_total = 0
    for my_node in my_node_list:
        node_benefit_total += node_benefit(my_plan, my_node)
    node_benefit_total = node_benefit_total / len(my_node_list)

    # Component 2: Station perspective - how efficiently do stations cover nodes
    # (with diminishing returns to encourage balanced coverage)
    station_benefit_total = 0
    for station in my_plan:
        station_benefit_total += station_benefit(station, my_node_list)
    station_benefit_total = station_benefit_total / len(my_plan)

    # Balance both components equally
    # This ensures neither metric dominates the benefit calculation
    my_benefit = (node_benefit_total + station_benefit_total) / 2
    return my_benefit


def travel_cost(my_node_list):
    """ yields the estimated travel time of all vehicles """
    my_cost_travel = sum([my_node[1]["distance"] * (1 + weak_demand(my_node)) / VELOCITY for my_node in my_node_list])
    return my_cost_travel


def travel_metric(my_node_list):
    """Max (worst-case) travel time in minutes across all nodes, demand-weighted.

    Reported alongside the normalized travel cost because the aggregate hides
    which node is worst off; both compare_rl.py and evaluate_all.py report it.
    """
    big_travel_list = []
    for my_node in my_node_list:
        travel = my_node[1]["distance"] / VELOCITY * 60
        times = ceil(10 * weak_demand(my_node))
        for _ in range(times):
            big_travel_list.append(travel)
    return max(big_travel_list) if big_travel_list else 0


def waiting_metric(my_plan):
    """Max (worst-case) waiting time in minutes across all stations."""
    big_waiting_list = []
    for my_station in my_plan:
        times = ceil(my_station[2]["D_s"])
        for _ in range(times):
            big_waiting_list.append(my_station[2]["W_s"] * 60)
    return max(big_waiting_list) if big_waiting_list else 0


def charging_time(my_plan):
    """
    yields the total charging time given the capability of the CS of the charging plan
    """
    # my_charg_time = sum([my_station[2]["D_s"] / my_station[2]["service rate"] for my_station in my_plan])
    my_charg_time = 0
    for my_station in my_plan:
        my_charg_time += (my_station[2]["D_s"] / (my_station[2]["service rate"] + 1e-6))
    return my_charg_time / time_unit


def waiting_time(my_plan):
    """
    returns the average total waiting time of the charging plan
    """
    my_wait_time = sum([my_station[2]["D_s"] * my_station[2]["W_s"] for my_station in my_plan])
    return my_wait_time / time_unit


def unserved_demand(my_plan):
    """
    Demand the plan cannot absorb: sum_s D(s) * P_N(s), in vehicles per time unit.

    This is the congestion signal that survives the choice of buffer size. Once a
    station saturates, D * P_N converges to D - mu, the raw gap between what
    arrives and what the chargers deliver, independent of the waiting-bay ratio
    kappa and of c. W_s does not have that property: above saturation it grows
    roughly like N / mu, so it scales with kappa and is only meaningful as a
    delay while the station still has slack.
    """
    total = sum([my_station[2].get("unserved", 0.0) for my_station in my_plan])
    return total / time_unit


def unserved_ratio(my_plan):
    """
    Fraction of arriving demand the plan turns away, sum_s D(s) P_N(s) / sum_s D(s).

    The denominator is the same for every plan: station_seeking assigns each node
    to exactly one station, so sum_s D(s) is just the district's total demand and
    does not move when stations are added. That makes this an absolute 0..1
    figure -- "15% of drivers get served" reads the same in every district and
    does not depend on how bad the existing infrastructure happens to be, which a
    ratio against p_0 would bake in.

    Note this is a level, not a spread: because the districts are provisioned far
    below their demand, every reachable plan sits high in [0, 1] (0.85 to 1.00 on
    DongDa), so the term shifts the objective much more than it tilts it. Raise
    UNSERVED_WEIGHT if the gradient needs to bite harder.
    """
    total_demand = sum([my_station[2].get("D_s", 0.0) for my_station in my_plan])
    if total_demand <= 0:
        return 0.0
    return sum([my_station[2].get("unserved", 0.0) for my_station in my_plan]) / total_demand


def social_cost(my_plan, my_node_list):
    """
    returns the social cost, i.e. the negative side of the charging plan
    """
    cost_travel = travel_cost(my_node_list)  # dimensionless
    charg_time = charging_time(my_plan)  # dimensionless
    wait_time = waiting_time(my_plan)  # dimensionless
    cost_boring = charg_time + wait_time  # dimensionless
    my_social_cost = alpha * cost_travel + (1 - alpha) * cost_boring
    return my_social_cost


def existing_score(my_existing_plan, my_node_list):
    """
    computes the score of the existing infrastructure
    """
    my_benefit = social_benefit(my_existing_plan, my_node_list)
    travel_time = travel_cost(my_node_list)  # dimensionless
    charg_time = charging_time(my_existing_plan)  # dimensionless
    wait_time = waiting_time(my_existing_plan)
    cost_boring = charg_time + wait_time  # dimensionless
    my_cost = alpha * travel_time + (1 - alpha) * cost_boring
    fairness = social_fairness(my_node_list)
    return my_benefit, my_cost, charg_time, wait_time, travel_time, fairness


def social_fairness(my_node_list):
    """
    Return a scalar fairness score for the node coverage distribution.
    Higher values indicate more fair (more even) coverage across nodes.
    We use the inverse of the standard deviation of station counts so that
    perfectly even coverage -> higher fairness, and skewed coverage -> lower fairness.
    """
    counts = np.array([node[1].get('n_stations', 0) for node in my_node_list], dtype=float)
    if counts.size == 0:
        return 0.0
    std = float(np.std(counts))
    return 1.0 / (1.0 + std)


def norm_score(my_plan, my_node_list, norm_benefit, norm_charg, norm_wait, norm_travel, grid_penalty=None):
    """
    same as score, but normalised.
    """
    my_score = -my_inf
    if not my_plan:
        # Every caller unpacks the full tuple, so the guard has to keep the shape.
        # Returning the bare score here raised TypeError instead of signalling
        # "no plan", which is what this branch exists to say.
        return my_score, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    benefit = social_benefit(my_plan, my_node_list) / norm_benefit
    cost_travel = travel_cost(my_node_list) / norm_travel # dimensionless
    charg_time = charging_time(my_plan) / norm_charg # dimensionless
    wait_time = waiting_time(my_plan) / norm_wait # dimensionless
    cost = (alpha * cost_travel + (1 - alpha) * (charg_time + wait_time)) / 3
    fairness = social_fairness(my_node_list)
    if grid_penalty is not None:
        if isinstance(grid_penalty, dict):
            dist_p = abs(grid_penalty.get('dist_penalty', 0.0))
            cap_p = abs(grid_penalty.get('cap_penalty', 0.0))
        elif isinstance(grid_penalty, tuple) and len(grid_penalty) == 2:
            dist_p = abs(grid_penalty[0])
            cap_p = abs(grid_penalty[1])
        else:
            dist_p = abs(grid_penalty)
            cap_p = 0.0
        # dist_p is an extensive sum over stations -> average it per station.
        # cap_p arrives already normalized: calculate_grid_penalty divides the sum
        # of per-bus overload ratios by the district's bus count, a constant. Do
        # not divide it by len(my_plan) here -- a plan-dependent divisor would let
        # the agent dilute an overloaded bus by building at buses with headroom.
        avg_penalty = (dist_p / max(1, len(my_plan))) + cap_p
        my_score = (benefit - cost + fairness) / 3 - GRID_PENALTY_WEIGHT * avg_penalty
    else:
        my_score = (benefit - cost + fairness) / 3
    return my_score, benefit, cost, charg_time, wait_time, cost_travel, fairness


def score(my_plan, my_node_list):
    """
    returns the final result, i.e., the social score
    """
    my_score = -my_inf
    benefit = 0
    cost = 0
    if not my_plan:
        return my_score, benefit, cost
    benefit = social_benefit(my_plan, my_node_list)  # dimensionless
    cost = social_cost(my_plan, my_node_list)
    my_score = my_lambda * benefit - (1 - my_lambda) * cost
    return my_score, benefit, cost

def get_relocate_cost(station_config_index):
    move_cost = RELOCATION_FACTOR * INSTALL_FEE[station_config_index]
    return move_cost

# Constraints checks ############################################################################
def station_capacity_check(my_plan):
    """
    check if number of stations exceed capacity
    """
    for my_station in my_plan:
        s_x = my_station[1]
        if sum(s_x) > K:
            print("Error: More chargers at the station than admitted: {} chargers".format(sum(s_x)))


def installment_cost_check(my_plan, my_basic_cost):
    """
    check if instalment costs exceed budget
    """
    total_inst_cost = sum([my_station[2]["fee"] for my_station in my_plan]) - my_basic_cost
    if total_inst_cost > BUDGET:
        print("Error: Maximal BUDGET for installation costs exceeded.")


def control_charg_decision(my_plan, my_node_list):
    for my_node in my_node_list:
        station_sum = sum([1 for my_station in my_plan if my_node[1]["charging station"] == my_station[0]])
        if station_sum > 1:
            print("Error: More than one station is assigned to a node.")


def waiting_time_check(my_plan):
    """
    check that wiating time is bounded
    """
    for my_station in my_plan:
        s_dict = my_station[2]
        if s_dict["W_s"] == my_inf:
            print("Error: Waiting time goes to infinity.")


def constraint_check(my_plan, my_node_list, basic_cost):
    """
    test if solution satisfies all constraints
    """
    installment_cost_check(my_plan, basic_cost)
    control_charg_decision(my_plan, my_node_list)
    station_capacity_check(my_plan)
    waiting_time_check(my_plan)


def get_lookup(path):
    with open(path, 'r') as f:
        lookup = json.load(f)
    return lookup


def initial_solution(my_config_dict, my_node_list, s_pos):
    """
    get the initial solution for the charging configuration
    """
    W = 0  # minimum capacity constraint
    radius = RADIUS_MAX
    # search for all nodes within station radius
    for my_node in my_node_list:
        if haversine(s_pos, my_node) <= radius:
            W += weak_demand(my_node)
    W = ceil(W) * BATTERY
    key_list = sorted(list(my_config_dict.keys()))
    for key in key_list:
        if int(key) > W:  # convert str to int
            break
    best_config = my_config_dict[key]
    return best_config


def coverage(my_node_list, my_plan):
    """
    see which nodes are covered by the charging plan
    """
    for my_node in my_node_list:
        cover = node_benefit(my_plan, my_node)
        my_node[1]["benefit"] = cover


def choose_node_new_benefit(free_list, all_node_list, R_search=0.7):
    """
    Pick location with highest potential based on coverage and local benefit.
    Calculates potential coverage (how many nodes a station here would cover)
    and local social benefit (how many existing stations serve nearby nodes).

    Args:
        free_list: list of candidate node tuples (node_id, node_attrs)
        all_node_list: list of all nodes in the network
        R_search: search radius in km for identifying beneficiary nodes

    Returns:
        best candidate node tuple
    """
    if not free_list:
        return None

    potential_scores = []
    n_nodes = len(all_node_list)

    # Default search radius (in km) for station coverage if not specified
    # default_radius = RADIUS_MAX * 0.5 if 'RADIUS_MAX' in globals() else 2.0

    for candidate_node in free_list:
        # 1. Calculate potential coverage: how many nodes would be covered by a station here
        # This is the "station scope" - count nodes within R_search of this candidate
        covered_nodes_count = 0
        local_benefit_list = []
        for node in all_node_list:
            if haversine(candidate_node, node) <= R_search:
                covered_nodes_count += 1
                # Diminishing returns: each additional station provides less marginal benefit
                n_stations = node[1].get("n_stations", 0)
                priv_CS = node[1].get("private_cs", 0)

                # Áp dụng trọng số nhu cầu (demand weight) của từng node
                # Đảm bảo đặt trạm ở nơi có demand thực tế để giảm waiting và travel time
                # node_demand = weak_demand(node)
                marginal_benefit = (1.0 / (n_stations + 1)) * (1 - 0.1 * priv_CS) #* (1 + 2.0 * node_demand)
                local_benefit_list.append(marginal_benefit)

        station_scope = covered_nodes_count / n_nodes if n_nodes > 0 else 0

        # 2. Calculate local social benefit: sum of marginal benefits over covered nodes
        # This captures the actual increase in social benefit for the nodes
        social_node_benefit = sum(local_benefit_list) / n_nodes if n_nodes > 0 else 0.0

        # 3. Combine metrics: weight station coverage and local benefit
        _score = social_node_benefit + station_scope
        potential_scores.append(_score)

    # Return node with highest score
    best_index = np.argmax(potential_scores)
    return free_list[best_index]


def choose_node_bydemand(free_list, my_plan, add=False):
    """
    pick location with highest dynamic demand
    """
    chosen_node = None
    if add:
        # choose the node with the highest waiting time
        priority_list = [station[2]["D_s"] * station[2]["W_s"] + station[2]["D_s"] / (station[2]["service rate"] + eps)
                         for station in my_plan]
        max_station_index = np.argmax(priority_list)
        max_station = my_plan[max_station_index]
        chosen_node = max_station[0]

    else:
        demand_list = [dynamic_demand(my_node, my_plan) for my_node in free_list]
        chosen_index = demand_list.index(max(demand_list))
        chosen_node = free_list[chosen_index]
    return chosen_node


def anti_choose_node_bybenefit(my_node_list, my_plan):
    """
    choose station with the least coverage
    """
    if not my_plan:
        return None

    coverage_scores = [station_benefit(station, my_node_list)
                       for station in my_plan]
    min_coverage_index = np.argmin(coverage_scores)
    remove_station = my_plan[min_coverage_index]
    return remove_station


def _support_stations(station):
    charg_time = station[2]["D_s"] / (station[2]["service rate"] + 1e-6)
    wait_time = station[2]["D_s"] * station[2]["W_s"]
    neediness = (wait_time + charg_time)
    return neediness


def support_stations(my_plan, free_list):
    """
    choose a station which needs support due to highest waiting + charging time
    """
    cost_list = [_support_stations(station) for station in my_plan]
    if not cost_list:
        chosen_node = choose_node_bydemand(free_list)
    else:
        index = np.argmax(cost_list)
        station_sos = my_plan[index]
        if sum(station_sos[1]) < K:
            chosen_node = station_sos[0]
        else:
            # look for nearest node that could support the station
            dis_list = [haversine(station_sos[0], my_node) for my_node in free_list]
            min_index = dis_list.index(min(dis_list))
            chosen_node = free_list[min_index]
    return chosen_node


# Parameters ########################################################
alpha = 0.8
my_lambda = 0.5
GRID_PENALTY_WEIGHT = 1.0  # tunable weight on the grid (distance + capacity) penalty in norm_score
# Queue model (see avg_waiting). kappa = waiting bays per charger; a station of
# c chargers has system capacity N = (1 + kappa) * c. This is a property of the
# site, not of its demand. kappa = 0 -> Erlang loss system M/M/c/c.
WAIT_BAY_RATIO = 1.0
# Weight on the turned-away-demand term inside the (1 - alpha) group of norm_score.
UNSERVED_WEIGHT = 1.0
# Dynamic-demand model (see dynamic_demand). These shape the observation and the
# demand-targeting heuristic behind actions 1 and 3 -- they do NOT enter norm_score.
DEMAND_ETA = 0.4    # scaling_factor: how strongly installed capacity absorbs demand
DEMAND_BETA = 0.6   # distance_decay_factor: how fast that absorption falls off with distance
eps = 1e-6
ev_per_capita = 0.022
evs_parking_area = 15  # meter square

K = 100  # maximal number of chargers at a station
RADIUS_MAX = 1  # [radius_max] = km
RADIUS_CAPACITY_SCALE = 1.0  # C_0 in the radius logistic (see influence_radius), [C_0] = MW
CHARGING_POWER = np.array([3, 7, 11, 20, 22, 30, 60, 80, 120, 150, 180, 250])
INSTALL_FEE = np.array([5, 11, 12, 100, 12, 143, 278, 397, 416, 676, 956, 3272])
BATTERY = 85  # battery capacity, [BATTERY] = kWh
RELOCATION_FACTOR = 0.2  # Assumption: Moving costs 20% of a new one

BUDGET = 900000

time_unit = 1  # [time_unit] = h, introduced for getting the units correctly
VELOCITY = 40  # km/h

my_inf = 10 ** 6
my_dis_inf = 10 ** 7
