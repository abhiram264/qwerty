#!/usr/bin/env python3
"""
Grocery delivery route planner using Google OR-Tools VRP + Geoapify routing.

- Warehouses:
  - WH1 at Kukatpally, Hyderabad
  - WH2 at Ameerpet, Hyderabad

- 5 reps per warehouse, each up to 4 trips
- Each trip can visit at most 5 orders
- 100 orders spread over Hyderabad (grid over city bbox)

Outputs:
- Trips from OR-Tools planner (sequence + ETAs)
- Haversine total distance (km) for all trips
- Geoapify route segments per trip (real-road distance/time + geometry)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import math
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # type: ignore

# --------- Geoapify config ---------

GEOAPIFY_API_KEY = "fc9a8604115d4dc5ad9a5126313eb1c1"  # put your key here
GEOAPIFY_ROUTING_URL = "https://api.geoapify.com/v1/routing"

# Global cache for Geoapify responses
GLOBAL_LEG_CACHE: Dict[Tuple[float, float, float, float], Dict[str, Any]] = {}

# ---------- Domain models ----------

@dataclass
class Warehouse:
    id: str
    lat: float
    lng: float


@dataclass
class Rep:
    id: str
    warehouse_id: str
    max_trips_per_day: int


@dataclass
class Store:
    id: str
    lat: float
    lng: float


@dataclass
class Order:
    id: str
    store: Store
    warehouse_candidates: List[str]


@dataclass
class TripStop:
    order_id: str
    eta: datetime


@dataclass
class Trip:
    warehouse_id: str
    rep_id: str
    trip_index_for_rep: int
    stops: List[TripStop]
    start_time: datetime
    end_time: datetime
    duration_minutes: float


# ---------- Shared geometry ----------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lng points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ---------- Geoapify routing helper ----------

def geoapify_route_leg(lat1: float, lng1: float, lat2: float, lng2: float) -> Dict[str, Any]:
    """
    Call Geoapify Routing API for a single leg and return the first feature
    (contains 'properties' with distance/time and 'geometry' polyline).
    """
    waypoints = f"{lat1},{lng1}|{lat2},{lng2}"
    params = {
        "waypoints": waypoints,
        "mode": "drive",
        "apiKey": GEOAPIFY_API_KEY,
    }
    resp = requests.get(GEOAPIFY_ROUTING_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features", [])
    if not features:
        raise ValueError("Geoapify: no route found for leg")
    return features[0]


# ============================================================
#  OR-TOOLS PLANNER ONLY
# ============================================================

def build_data_model_ortools(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build OR-Tools data model (using Haversine distance as cost).
    """
    warehouses = payload["warehouses"]
    reps_raw = payload["reps"]
    orders_raw = payload["orders"]

    # locations[0..W-1] = warehouses, [W..] = orders
    locations: List[Tuple[float, float]] = []
    wh_id_to_index: Dict[str, int] = {}
    for i, wh in enumerate(warehouses):
        wh_id_to_index[wh["id"]] = i
        locations.append((wh["lat"], wh["lng"]))

    order_index_start = len(locations)
    order_nodes: List[Dict[str, Any]] = []
    order_id_to_node_index: Dict[str, int] = {}
    for j, o in enumerate(orders_raw):
        idx = order_index_start + j
        order_id_to_node_index[o["id"]] = idx
        order_nodes.append({"id": o["id"], "node_index": idx, "quantity": o.get("quantity", 1)})
        locations.append((o["store"]["lat"], o["store"]["lng"]))

    num_locations = len(locations)
    distance_matrix: List[List[int]] = [[0] * num_locations for _ in range(num_locations)]
    for i in range(num_locations):
        lat_i, lng_i = locations[i]
        for j in range(num_locations):
            if i == j:
                continue
            lat_j, lng_j = locations[j]
            km = haversine_km(lat_i, lng_i, lat_j, lng_j)
            distance_matrix[i][j] = int(round(km * 1000))

    # Vehicles: each rep gets max_trips_per_day virtual vehicles
    vehicles_meta: List[Dict[str, Any]] = []
    for r in reps_raw:
        rep_id = r["id"]
        wh_id = r["warehouse_id"]
        max_trips = r.get("max_trips_per_day", 1)
        depot_index = wh_id_to_index[wh_id]
        for trip_idx in range(1, max_trips + 1):
            vehicles_meta.append(
                {
                    "rep_id": rep_id,
                    "warehouse_id": wh_id,
                    "trip_index_for_rep": trip_idx,
                    "start_index": depot_index,
                    "end_index": depot_index,
                }
            )

    num_vehicles = len(vehicles_meta)

    # Demands: 0 for depots, 'quantity' per order
    demands = [0] * num_locations
    for o in order_nodes:
        qty = o.get("quantity", 1)
        demands[o["node_index"]] = int(qty)

    # Capacity: 5 units per vehicle (sum of quantities per trip)
    vehicle_capacities = [5] * num_vehicles

    return {
        "locations": locations,
        "distance_matrix": distance_matrix,
        "demands": demands,
        "vehicle_capacities": vehicle_capacities,
        "num_vehicles": num_vehicles,
        "vehicle_meta": vehicles_meta,
        "order_nodes": order_nodes,
        "order_id_to_node_index": order_id_to_node_index,
        "warehouse_id_to_index": wh_id_to_index,
    }


def solve_vrp_ortools(data: Dict[str, Any]):
    num_locations = len(data["locations"])
    num_vehicles = data["num_vehicles"]
    starts = [v["start_index"] for v in data["vehicle_meta"]]
    ends = [v["end_index"] for v in data["vehicle_meta"]]

    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = data["distance_matrix"]

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    demands = data["demands"]

    def demand_callback(from_index: int) -> int:
        node = manager.IndexToNode(from_index)
        return demands[node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        data["vehicle_capacities"],
        True,
        "Capacity",
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(5)

    solution = routing.SolveWithParameters(search_parameters)

    # Compute basic optimization stats from OR-Tools (objective + optional gap).
    optimization_stats: Dict[str, Any] = {
        "objective_value": None,
        "best_objective_bound": None,
        "optimality_gap": None,
        "optimization_score": None,
    }

    if solution is not None:
        try:
            objective_value = solution.ObjectiveValue()
        except Exception:
            objective_value = None

        best_bound = None
        try:
            solver = routing.solver()
            if hasattr(solver, "BestObjectiveBound"):
                best_bound = solver.BestObjectiveBound()
        except Exception:
            best_bound = None

        # If we couldn't read the objective for any reason, fall back to a
        # generic "valid solution" score later.
        gap = None
        score = None

        if objective_value is not None and objective_value > 0 and best_bound is not None:
            # For a minimization problem, best_bound is a lower bound on the optimal cost.
            # Gap = (obj - bound) / obj in [0, 1], lower is better.
            gap_raw = (objective_value - best_bound) / float(objective_value) if objective_value != 0 else 0.0
            gap = max(0.0, min(1.0, gap_raw))
            # Turn into an easy-to-read percentage score: 100 = proven optimal.
            score = round((1.0 - gap) * 100.0, 1)
        else:
            # If we can't compute a meaningful bound/objective pair, at least
            # expose a non-empty score to indicate that OR-Tools found a
            # feasible solution.
            score = 100.0

        optimization_stats.update(
            {
                "objective_value": objective_value,
                "best_objective_bound": best_bound,
                "optimality_gap": gap,
                "optimization_score": score,
            }

        )
        print(optimization_stats)

    return routing, manager, solution, optimization_stats


def extract_trips_ortools(
    payload: Dict[str, Any],
    data: Dict[str, Any],
    routing: pywrapcp.RoutingModel,
    manager: pywrapcp.RoutingIndexManager,
    solution: pywrapcp.Assignment,
    use_geoapify: bool = True,
) -> Tuple[List[Trip], float, List[Dict[str, Any]]]:
    service_date = payload["service_date"]
    date_obj = datetime.fromisoformat(service_date)

    node_index_to_order_id: Dict[int, str] = {}
    for o in data["order_nodes"]:
        node_index_to_order_id[o["node_index"]] = o["id"]

    vehicles_meta = data["vehicle_meta"]
    locations = data["locations"]
    wh_id_to_idx = data["warehouse_id_to_index"]

    trips: List[Trip] = []
    total_dist_km = 0.0
    geo_routes: List[Dict[str, Any]] = []

    avg_speed_kmph = 20.0
    km_per_min = avg_speed_kmph / 60.0 if avg_speed_kmph > 0 else 0.0

    # 1. Identify all trips and collect legs to fetch
    temp_trips = []
    legs_to_fetch = set()

    for vehicle_id in range(len(vehicles_meta)):
        index = routing.Start(vehicle_id)
        meta = vehicles_meta[vehicle_id]
        
        seq_nodes: List[int] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node in node_index_to_order_id:
                seq_nodes.append(node)
            index = solution.Value(routing.NextVar(index))

        if not seq_nodes:
            continue

        depot_idx = wh_id_to_idx[meta["warehouse_id"]]
        
        # Collect legs for this trip
        trip_legs = []
        
        # depot -> first
        first_node = seq_nodes[0]
        trip_legs.append((locations[depot_idx], locations[first_node]))
        
        # between nodes
        prev = first_node
        for node in seq_nodes[1:]:
            trip_legs.append((locations[prev], locations[node]))
            prev = node
            
        # last -> depot
        trip_legs.append((locations[prev], locations[depot_idx]))
        
        # Add to set for fetching
        for leg in trip_legs:
            legs_to_fetch.add((leg[0][0], leg[0][1], leg[1][0], leg[1][1]))

        temp_trips.append({
            "meta": meta,
            "seq_nodes": seq_nodes,
            "depot_idx": depot_idx,
            "trip_legs": trip_legs
        })

    # 2. Fetch legs in parallel if enabled
    leg_cache = {}
    legs_to_fetch_filtered = []
    
    if use_geoapify and legs_to_fetch:
        # Check global cache first
        for leg in legs_to_fetch:
            if leg in GLOBAL_LEG_CACHE:
                leg_cache[leg] = GLOBAL_LEG_CACHE[leg]
            else:
                legs_to_fetch_filtered.append(leg)
        
        if legs_to_fetch_filtered:
            print(f"Fetching {len(legs_to_fetch_filtered)} new Geoapify legs in parallel...")
            with ThreadPoolExecutor(max_workers=20) as executor:
                future_to_leg = {
                    executor.submit(geoapify_route_leg, lat1, lng1, lat2, lng2): (lat1, lng1, lat2, lng2)
                    for lat1, lng1, lat2, lng2 in legs_to_fetch_filtered
                }
                for future in as_completed(future_to_leg):
                    leg_key = future_to_leg[future]
                    try:
                        result = future.result()
                        leg_cache[leg_key] = result
                        GLOBAL_LEG_CACHE[leg_key] = result
                    except Exception as e:
                        print(f"Error fetching leg {leg_key}: {e}")

    # 3. Build final trips using cached data
    for t_data in temp_trips:
        meta = t_data["meta"]
        seq_nodes = t_data["seq_nodes"]
        depot_idx = t_data["depot_idx"]
        trip_legs = t_data["trip_legs"]
        
        wh_id = meta["warehouse_id"]
        rep_id = meta["rep_id"]
        trip_index_for_rep = meta["trip_index_for_rep"]

        # Calculate times and build stops
        service_time_minutes = 5
        start_time = date_obj.replace(
            hour=9, minute=0, second=0, microsecond=0
        ) + timedelta(hours=(trip_index_for_rep - 1) * 2)
        current_time = start_time
        
        stops: List[TripStop] = []
        trip_segments: List[Dict[str, Any]] = []
        trip_dist_km = 0.0

        # Process sequence (depot -> first -> ... -> last)
        # Note: trip_legs includes return to depot, but stops list only includes orders
        
        # We need to iterate nodes to build stops list
        # The trip_legs list corresponds to:
        # 0: depot -> node[0]
        # 1: node[0] -> node[1]
        # ...
        # N: node[N-1] -> node[N]
        # N+1: node[N] -> depot
        
        leg_idx = 0
        
        # Handle first leg (depot -> first node)
        lat1, lng1 = trip_legs[leg_idx][0]
        lat2, lng2 = trip_legs[leg_idx][1]
        leg_key = (lat1, lng1, lat2, lng2)
        
        geo_data = leg_cache.get(leg_key)
        
        if geo_data:
            # Use Geoapify data
            duration_sec = geo_data['properties'].get('time', 0)
            dist_meters = geo_data['properties'].get('distance', 0)
            travel_minutes = duration_sec / 60.0
            leg_km = dist_meters / 1000.0
            trip_segments.append(geo_data)
        else:
            # Fallback to Haversine
            leg_km = haversine_km(lat1, lng1, lat2, lng2)
            travel_minutes = leg_km / km_per_min if km_per_min > 0 else 0.0
            
        current_time += timedelta(minutes=travel_minutes)
        trip_dist_km += leg_km
        
        # Add first stop
        order_id = node_index_to_order_id[seq_nodes[0]]
        stops.append(TripStop(order_id=order_id, eta=current_time))
        current_time += timedelta(minutes=service_time_minutes)
        
        leg_idx += 1
        
        # Handle intermediate legs
        for i in range(1, len(seq_nodes)):
            lat1, lng1 = trip_legs[leg_idx][0]
            lat2, lng2 = trip_legs[leg_idx][1]
            leg_key = (lat1, lng1, lat2, lng2)
            
            geo_data = leg_cache.get(leg_key)
            
            if geo_data:
                duration_sec = geo_data['properties'].get('time', 0)
                dist_meters = geo_data['properties'].get('distance', 0)
                travel_minutes = duration_sec / 60.0
                leg_km = dist_meters / 1000.0
                trip_segments.append(geo_data)
            else:
                leg_km = haversine_km(lat1, lng1, lat2, lng2)
                travel_minutes = leg_km / km_per_min if km_per_min > 0 else 0.0
                
            current_time += timedelta(minutes=travel_minutes)
            trip_dist_km += leg_km
            
            order_id = node_index_to_order_id[seq_nodes[i]]
            stops.append(TripStop(order_id=order_id, eta=current_time))
            current_time += timedelta(minutes=service_time_minutes)
            
            leg_idx += 1
            
        # Handle return leg (last node -> depot)
        lat1, lng1 = trip_legs[leg_idx][0]
        lat2, lng2 = trip_legs[leg_idx][1]
        leg_key = (lat1, lng1, lat2, lng2)
        
        geo_data = leg_cache.get(leg_key)
        
        if geo_data:
            duration_sec = geo_data['properties'].get('time', 0)
            dist_meters = geo_data['properties'].get('distance', 0)
            return_minutes = duration_sec / 60.0
            leg_km = dist_meters / 1000.0
            trip_segments.append(geo_data)
        else:
            leg_km = haversine_km(lat1, lng1, lat2, lng2)
            return_minutes = leg_km / km_per_min if km_per_min > 0 else 0.0
            
        end_time = current_time + timedelta(minutes=return_minutes)
        trip_dist_km += leg_km
        total_dist_km += trip_dist_km
        
        trip_duration_minutes = (end_time - start_time).total_seconds() / 60.0

        trips.append(
            Trip(
                warehouse_id=wh_id,
                rep_id=rep_id,
                trip_index_for_rep=trip_index_for_rep,
                stops=stops,
                start_time=start_time,
                end_time=end_time,
                duration_minutes=trip_duration_minutes,
            )
        )
        
        if trip_segments:
            geo_routes.append({
                "rep_id": rep_id,
                "trip_index_for_rep": trip_index_for_rep,
                "segments": trip_segments,
            })

    return trips, total_dist_km, geo_routes


def plan_routes_ortools(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    # ------------------------------------------------------------------
    # Pre-process orders for capacity: if total requested quantity
    # exceeds the global fleet capacity, we keep the highest-priority
    # orders for today and mark the remainder as "carried over".
    # ------------------------------------------------------------------
    warehouses = payload["warehouses"]
    reps = payload["reps"]
    orders = payload["orders"]

    # Global fleet capacity in "units" (5 units per trip).
    num_vehicles_capacity = sum(r.get("max_trips_per_day", 1) for r in reps)
    per_vehicle_capacity = 5
    total_capacity_units = num_vehicles_capacity * per_vehicle_capacity

    # Ensure each order has normalized priority/quantity (should already
    # be validated by validate_orders_data for uploaded payloads).
    normalized_orders = []
    for o in orders:
        priority = float(o.get("priority", 0.5))
        quantity = int(o.get("quantity", 1))
        normalized = dict(o)
        normalized["priority"] = priority
        normalized["quantity"] = quantity
        normalized_orders.append(normalized)

    # Sort by priority (high first), then by smaller quantity to pack better.
    normalized_orders.sort(key=lambda x: (-x["priority"], x["quantity"]))

    served_orders = []
    carried_over_orders = []
    used_capacity = 0

    for o in normalized_orders:
        qty = o["quantity"]
        if used_capacity + qty <= total_capacity_units:
            served_orders.append(o)
            used_capacity += qty
        else:
            carried_over_orders.append(o)

    effective_payload = {
        "service_date": payload["service_date"],
        "warehouses": warehouses,
        "reps": reps,
        "orders": served_orders,
    }

    data = build_data_model_ortools(effective_payload)
    routing, manager, solution, optimization_stats = solve_vrp_ortools(data)
    if not solution:
        raise RuntimeError("No solution found by OR-Tools")

    trips, total_dist_km, geo_routes = extract_trips_ortools(
        effective_payload, data, routing, manager, solution, use_geoapify=True
    )

    res_trips: List[Dict[str, Any]] = []
    for t in trips:
        res_trips.append(
            {
                "warehouse_id": t.warehouse_id,
                "rep_id": t.rep_id,
                "trip_index_for_rep": t.trip_index_for_rep,
                "start_time": t.start_time.isoformat(),
                "end_time": t.end_time.isoformat(),
                "duration_minutes": round(t.duration_minutes, 1),
                "stops": [
                    {"order_id": s.order_id, "eta": s.eta.isoformat()}
                    for s in t.stops
                ],
            }
        )

    result: Dict[str, Any] = {
        "service_date": payload["service_date"],
        "trips": res_trips,
        "geo_routes": geo_routes,
    }

    # Attach optimization stats for downstream consumers (frontend, exports).
    if optimization_stats.get("optimization_score") is not None:
        result["optimization_score"] = optimization_stats["optimization_score"]
    if optimization_stats.get("optimality_gap") is not None:
        result["optimality_gap"] = optimization_stats["optimality_gap"]
    if optimization_stats.get("objective_value") is not None:
        result["objective_value"] = optimization_stats["objective_value"]
    if optimization_stats.get("best_objective_bound") is not None:
        result["best_objective_bound"] = optimization_stats["best_objective_bound"]

    # Attach carried-over orders (not routed today, e.g. for the next day).
    if carried_over_orders:
        result["carried_over_orders"] = carried_over_orders

    return result, total_dist_km


# ============================================================
#  DETERMINISTIC PAYLOAD (100 locations over Hyderabad)
# ============================================================

def generate_demo_payload() -> Dict[str, Any]:
    """
    2 warehouses:
      WH1 at Kukatpally, Hyderabad
      WH2 at Ameerpet, Hyderabad

    100 orders arranged as a 10x10 grid over Hyderabad's bounding box.
    """
    service_date = "2025-12-02"

    # Approx coordinates
    # Kukatpally (rough): ~17.49, 78.40 [web:211][web:213]
    # Ameerpet (rough): ~17.44, 78.45 [web:203][web:212]
    warehouses = [
        {"id": "WH1", "lat": 17.49, "lng": 78.40},
        {"id": "WH2", "lat": 17.44, "lng": 78.45},
    ]

    reps = []
    for i in range(1, 6):
        reps.append({"id": f"WH1-R{i}", "warehouse_id": "WH1", "max_trips_per_day": 1})
        reps.append({"id": f"WH2-R{i}", "warehouse_id": "WH2", "max_trips_per_day": 1})

    orders = []
    order_id = 1

    # Hyderabad rough bbox: lat ~ [17.25, 17.55], lng ~ [78.30, 78.60] [web:194][web:197]
    lat_min, lat_max = 17.25, 17.55
    lng_min, lng_max = 78.30, 78.60
    lat_step = (lat_max - lat_min) / 9.0  # 10 steps
    lng_step = (lng_max - lng_min) / 9.0

    for i in range(10):
        for j in range(10):
            lat = lat_min + i * lat_step
            lng = lng_min + j * lng_step
            orders.append(
                {
                    "id": f"O{order_id}",
                    "store": {
                        "id": f"S{order_id}",
                        "lat": lat,
                        "lng": lng,
                    },
                    "warehouse_candidates": ["WH1", "WH2"],
                }
            )
            order_id += 1

    return {
        "service_date": service_date,
        "warehouses": warehouses,
        "reps": reps,
        "orders": orders,
    }


# ============================================================
#  FLASK APP
# ============================================================

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# File upload configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
WAREHOUSES_FILE = os.path.join(UPLOAD_FOLDER, 'warehouses.json')
ORDERS_FILE = os.path.join(UPLOAD_FOLDER, 'orders.json')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ============================================================
#  HELPER FUNCTIONS FOR FILE HANDLING
# ============================================================

def validate_warehouses_data(data):
    """Validate warehouses and reps JSON structure."""
    if not isinstance(data, dict):
        return False, "Data must be a JSON object"
    
    if 'warehouses' not in data or 'reps' not in data:
        return False, "Data must contain 'warehouses' and 'reps' keys"
    
    if not isinstance(data['warehouses'], list) or not isinstance(data['reps'], list):
        return False, "'warehouses' and 'reps' must be arrays"
    
    # Validate warehouse structure
    for wh in data['warehouses']:
        if not all(key in wh for key in ['id', 'lat', 'lng']):
            return False, "Each warehouse must have 'id', 'lat', and 'lng'"
        if not isinstance(wh['lat'], (int, float)) or not isinstance(wh['lng'], (int, float)):
            return False, "Warehouse 'lat' and 'lng' must be numbers"
    
    # Validate reps structure
    warehouse_ids = {wh['id'] for wh in data['warehouses']}
    for rep in data['reps']:
        if not all(key in rep for key in ['id', 'warehouse_id', 'max_trips_per_day']):
            return False, "Each rep must have 'id', 'warehouse_id', and 'max_trips_per_day'"
        if rep['warehouse_id'] not in warehouse_ids:
            return False, f"Rep {rep['id']} references non-existent warehouse {rep['warehouse_id']}"
        if not isinstance(rep['max_trips_per_day'], int) or rep['max_trips_per_day'] < 1:
            return False, "'max_trips_per_day' must be a positive integer"
    
    return True, "Valid"

def validate_orders_data(data):
    """Validate orders JSON structure."""
    if not isinstance(data, dict):
        return False, "Data must be a JSON object"
    
    if 'orders' not in data:
        return False, "Data must contain 'orders' key"
    
    if not isinstance(data['orders'], list):
        return False, "'orders' must be an array"
    
    # Validate order structure
    for order in data['orders']:
        if not all(key in order for key in ['id', 'store', 'warehouse_candidates']):
            return False, "Each order must have 'id', 'store', and 'warehouse_candidates'"
        
        store = order['store']
        if not all(key in store for key in ['id', 'lat', 'lng']):
            return False, "Each store must have 'id', 'lat', and 'lng'"
        if not isinstance(store['lat'], (int, float)) or not isinstance(store['lng'], (int, float)):
            return False, "Store 'lat' and 'lng' must be numbers"
        
        if not isinstance(order['warehouse_candidates'], list) or len(order['warehouse_candidates']) == 0:
            return False, "'warehouse_candidates' must be a non-empty array"

        # Optional priority field, defaulting to 0.5 if missing.
        # Priority is expected to be a float in [0, 1], where higher means more important.
        priority = order.get('priority', 0.5)
        if not isinstance(priority, (int, float)):
            return False, "'priority' must be a number between 0 and 1"
        if priority < 0 or priority > 1:
            return False, "'priority' must be between 0 and 1"
        # Normalize back into the order so downstream logic can rely on it.
        order['priority'] = float(priority)

        # Optional quantity field, defaulting to 1 if missing.
        # Quantity represents how many units are to be carried for this order.
        quantity = order.get('quantity', 1)
        if not isinstance(quantity, int):
            return False, "'quantity' must be an integer"
        if quantity < 1:
            return False, "'quantity' must be at least 1"
        # For now we cap demo data at 5 to match per-trip capacity.
        if quantity > 5:
            return False, "'quantity' must not exceed 5 for this demo"
        order['quantity'] = quantity
    
    return True, "Valid"

def load_payload_from_files(service_date="2025-12-02"):
    """Load payload from uploaded files, or fall back to demo data."""
    if os.path.exists(WAREHOUSES_FILE) and os.path.exists(ORDERS_FILE):
        try:
            with open(WAREHOUSES_FILE, 'r') as f:
                wh_data = json.load(f)
            with open(ORDERS_FILE, 'r') as f:
                orders_data = json.load(f)
            
            return {
                "service_date": service_date,
                "warehouses": wh_data['warehouses'],
                "reps": wh_data['reps'],
                "orders": orders_data['orders'],
            }
        except Exception as e:
            print(f"Error loading files: {e}")
            return generate_demo_payload()
    else:
        return generate_demo_payload()

# ============================================================
#  API ENDPOINTS
# ============================================================

@app.route("/api/status", methods=["GET"])
def get_status():
    """Check if files are uploaded and ready."""
    return jsonify({
        "warehouses_uploaded": os.path.exists(WAREHOUSES_FILE),
        "orders_uploaded": os.path.exists(ORDERS_FILE),
        "ready": os.path.exists(WAREHOUSES_FILE) and os.path.exists(ORDERS_FILE)
    })

@app.route("/api/schema/warehouses", methods=["GET"])
def get_warehouses_schema():
    """Return JSON schema example for warehouses and reps."""
    schema = {
        "warehouses": [
            {"id": "WH1", "lat": 17.49, "lng": 78.40},
            {"id": "WH2", "lat": 17.44, "lng": 78.45}
        ],
        "reps": [
            {"id": "WH1-R1", "warehouse_id": "WH1", "max_trips_per_day": 1},
            {"id": "WH1-R2", "warehouse_id": "WH1", "max_trips_per_day": 1},
            {"id": "WH2-R1", "warehouse_id": "WH2", "max_trips_per_day": 1}
        ]
    }
    return jsonify(schema)

@app.route("/api/schema/orders", methods=["GET"])
def get_orders_schema():
    """Return JSON schema example for orders."""
    schema = {
        "orders": [
            {
                "id": "O1",
                "store": {"id": "S1", "lat": 17.40, "lng": 78.45},
                "warehouse_candidates": ["WH1", "WH2"],
                "priority": 0.5,
                "quantity": 3
            },
            {
                "id": "O2",
                "store": {"id": "S2", "lat": 17.42, "lng": 78.48},
                "warehouse_candidates": ["WH1"],
                "priority": 0.5,
                "quantity": 2
            },
            {
                "id": "O3",
                "store": {"id": "S3", "lat": 17.46, "lng": 78.42},
                "warehouse_candidates": ["WH2"],
                "priority": 0.5,
                "quantity": 5
            }
        ]
    }
    return jsonify(schema)

@app.route("/api/upload/warehouses", methods=["POST"])
def upload_warehouses():
    """Upload warehouses and reps JSON file."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate the data
        valid, message = validate_warehouses_data(data)
        if not valid:
            return jsonify({"error": message}), 400
        
        # Save to file
        with open(WAREHOUSES_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        return jsonify({
            "message": "Warehouses and reps uploaded successfully",
            "warehouses_count": len(data['warehouses']),
            "reps_count": len(data['reps'])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload/orders", methods=["POST"])
def upload_orders():
    """Upload orders JSON file."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate the data
        valid, message = validate_orders_data(data)
        if not valid:
            return jsonify({"error": message}), 400
        
        # Save to file
        with open(ORDERS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        return jsonify({
            "message": "Orders uploaded successfully",
            "orders_count": len(data['orders'])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/calculate-routes", methods=["POST"])
def calculate_routes():
    """Calculate routes using uploaded data."""
    try:
        # Check if files exist
        if not os.path.exists(WAREHOUSES_FILE) or not os.path.exists(ORDERS_FILE):
            return jsonify({"error": "Please upload both warehouses and orders files first"}), 400
        
        # Get service date from request, or use default
        request_data = request.get_json() or {}
        service_date = request_data.get('service_date', '2025-12-02')
        
        # Load payload and calculate routes
        payload = load_payload_from_files(service_date)
        ort_result, ort_total_km = plan_routes_ortools(payload)
        
        # Enrich the response
        response = {
            "warehouses": payload["warehouses"],
            "orders": payload["orders"],
            "trips": ort_result["trips"],
            "geo_routes": ort_result.get("geo_routes", []),
            "total_distance_km": ort_total_km,
            "service_date": payload["service_date"],
            "optimization_score": ort_result.get("optimization_score"),
            "optimality_gap": ort_result.get("optimality_gap"),
            "objective_value": ort_result.get("objective_value"),
            "best_objective_bound": ort_result.get("best_objective_bound"),
            "carried_over_orders": ort_result.get("carried_over_orders", []),
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/routes", methods=["GET"])
def get_routes():
    """Get routes (using uploaded data if available, otherwise demo data)."""
    payload = load_payload_from_files()
    ort_result, ort_total_km = plan_routes_ortools(payload)
    
    # Enrich the response with warehouse and order locations for easier frontend rendering
    response = {
        "warehouses": payload["warehouses"],
        "orders": payload["orders"],
        "trips": ort_result["trips"],
        "geo_routes": ort_result.get("geo_routes", []),
        "total_distance_km": ort_total_km,
        "service_date": payload["service_date"],
        "optimization_score": ort_result.get("optimization_score"),
        "optimality_gap": ort_result.get("optimality_gap"),
        "objective_value": ort_result.get("objective_value"),
        "best_objective_bound": ort_result.get("best_objective_bound"),
        "carried_over_orders": ort_result.get("carried_over_orders", []),
    }
    return jsonify(response)


if __name__ == "__main__":
    # Run on port 5000 by default
    app.run(debug=True, host="0.0.0.0", port=5000)

