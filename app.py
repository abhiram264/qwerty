#!/usr/bin/env python3
"""
Grocery delivery route planner using Google OR-Tools VRP + Geoapify routing.
Includes:
- Priority aging for carried-over orders.
- Priority-based penalty in OR-Tools for unserved orders.
- Persistent data storage using JSON files.
"""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from ortools.constraint_solver import pywrapcp, routing_enums_pb2 # type: ignore

# ----------------- Configuration & Global State -----------------

# Geoapify config
GEOAPIFY_API_KEY = "fc9a8604115d4dc5ad9a5126313eb1c1" # Use your key
GEOAPIFY_ROUTING_URL = "https://api.geoapify.com/v1/routing"
GEOAPIFY_ROUTEMATRIX_URL = "https://api.geoapify.com/v1/routematrix"

# Global cache for Geoapify responses (Improves performance by reducing API calls)
GLOBAL_LEG_CACHE: Dict[Tuple[float, float, float, float], Dict[str, Any]] = {}

# File paths
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
WAREHOUSES_FILE = os.path.join(UPLOAD_FOLDER, 'warehouses.json')
ORDERS_FILE = os.path.join(UPLOAD_FOLDER, 'orders.json')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ---------- Domain Models (Simplified for Clarity) ----------

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
    priority: float = 0.5 # New: Priority for aging
    quantity: int = 1 # New: Quantity for capacity

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

# ----------------- Data Persistence and Priority Aging -----------------

class DataStore:
    """Handles loading, saving, and updating core data."""

    def __init__(self, wh_path: str, orders_path: str):
        self.wh_path = wh_path
        self.orders_path = orders_path

    def load_data(self) -> Tuple[List[Warehouse], List[Rep], List[Order]]:
        """Loads data strictly from JSON files. No mock/demo generation."""
        try:
            with open(self.wh_path, 'r') as f:
                wh_data = json.load(f)
                warehouses = [Warehouse(**d) for d in wh_data.get('warehouses', [])]
                reps = [Rep(**d) for d in wh_data.get('reps', [])]

            with open(self.orders_path, 'r') as f:
                orders_data = json.load(f)
                # Ensure priority/quantity are set even if not in file
                orders = [
                    Order(
                        id=d['id'],
                        store=Store(**d['store']),
                        warehouse_candidates=d['warehouse_candidates'],
                        priority=float(d.get('priority', 0.5)),
                        quantity=int(d.get('quantity', 1))
                    ) for d in orders_data.get('orders', [])
                ]
            
            # Simple validation check
            if not all(o.warehouse_candidates for o in orders) or not (warehouses and reps):
                raise ValueError("Loaded data is incomplete/invalid.")
            
            return warehouses, reps, orders
        
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            # Do not generate mock data here; bubble up a clear error so the API
            # can tell the client to upload proper JSON files.
            raise RuntimeError(f"Failed to load warehouses/orders JSON: {e}")

    def save_orders(self, orders: List[Order]):
        """Saves the current list of orders back to the file."""
        data = {
            "orders": [
                {
                    "id": o.id,
                    "store": {"id": o.store.id, "lat": o.store.lat, "lng": o.store.lng},
                    "warehouse_candidates": o.warehouse_candidates,
                    "priority": round(o.priority, 3), # Round for cleaner storage
                    "quantity": o.quantity
                } for o in orders
            ]
        }
        with open(self.orders_path, 'w') as f:
            json.dump(data, f, indent=2)

    def update_orders_priority(self, served_ids: Set[str], current_orders: List[Order]) -> List[Order]:
        """
        Increases the priority of orders not served and returns the new list
        of orders for the next day.
        """
        next_day_orders: List[Order] = []
        
        for order in current_orders:
            if order.id not in served_ids:
                # Priority Aging: Increase priority by 0.1, capped at 1.0
                new_priority = min(1.0, order.priority + 0.1)
                order.priority = new_priority
                next_day_orders.append(order)
                
        # Save the updated list (the carried-over orders) for the next day
        self.save_orders(next_day_orders)
        
        return next_day_orders

    # Note: demo payload generation has been removed. Data must come from JSON files.

# ----------------- Shared Utility Functions -----------------

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

def geoapify_route_leg(lat1: float, lng1: float, lat2: float, lng2: float) -> Dict[str, Any]:
    """
    Call Geoapify Routing API for a single leg.
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

# ----------------- OR-Tools VRP Solver -----------------

def build_data_model_ortools(
    warehouses: List[Warehouse], reps: List[Rep], orders: List[Order]
) -> Dict[str, Any]:
    """
    Build OR-Tools data model with Geoapify real-road distances and capacity constraints.
    Uses parallel API calls with caching for performance. Falls back to Haversine if Geoapify fails.
    Adds **Priority-based Penalty** for unserved orders.
    """
    # 1. Locations: [0..W-1] = warehouses, [W..] = orders
    locations: List[Tuple[float, float]] = []
    wh_id_to_index: Dict[str, int] = {}
    for i, wh in enumerate(warehouses):
        wh_id_to_index[wh.id] = i
        locations.append((wh.lat, wh.lng))

    order_nodes: List[Dict[str, Any]] = []
    order_id_to_node_index: Dict[str, int] = {}
    order_index_start = len(locations)
    for j, o in enumerate(orders):
        idx = order_index_start + j
        order_id_to_node_index[o.id] = idx
        order_nodes.append({"id": o.id, "node_index": idx, "quantity": o.quantity, "priority": o.priority})
        locations.append((o.store.lat, o.store.lng))

    num_locations = len(locations)
    
    # 2. Distance Matrix for OR-Tools using Geoapify Route Matrix (real-road distances)
    # We call the matrix API ONCE for all locations to avoid N^2 per-leg calls.
    distance_matrix: List[List[int]] = [[0] * num_locations for _ in range(num_locations)]
    try:
        # Geoapify Route Matrix has a hard limit of 1000 elements per call.
        # We chunk the global matrix into smaller source/target blocks so that
        # (len(sources_block) * len(targets_block)) <= 1000.
        max_block_size = 31  # 31 * 31 = 961 < 1000
        print(f"Requesting Geoapify route matrix for {num_locations} locations in blocks...")

        for src_start in range(0, num_locations, max_block_size):
            src_end = min(num_locations, src_start + max_block_size)
            src_indices = list(range(src_start, src_end))

            for tgt_start in range(0, num_locations, max_block_size):
                tgt_end = min(num_locations, tgt_start + max_block_size)
                tgt_indices = list(range(tgt_start, tgt_end))

                # Build sources/targets for this block
                sources_block = [
                    {"location": [locations[i][1], locations[i][0]]}  # [lon, lat]
                    for i in src_indices
                ]
                targets_block = [
                    {"location": [locations[j][1], locations[j][0]]}  # [lon, lat]
                    for j in tgt_indices
                ]

                matrix_body = {
                    "mode": "drive",
                    "sources": sources_block,
                    "targets": targets_block,
                }

                resp = requests.post(
                    GEOAPIFY_ROUTEMATRIX_URL + f"?apiKey={GEOAPIFY_API_KEY}",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(matrix_body),
                    timeout=60,
                )
                if not resp.ok:
                    print(f"Geoapify route matrix error {resp.status_code}: {resp.text}")
                    resp.raise_for_status()

                matrix_data = resp.json()
                sources_to_targets = matrix_data.get("sources_to_targets") or []

                if len(sources_to_targets) != len(src_indices):
                    raise ValueError("Route matrix block size mismatch from Geoapify (sources).")

                # Fill the corresponding block in the global distance matrix
                for local_i, i in enumerate(src_indices):
                    row = sources_to_targets[local_i]
                    if len(row) != len(tgt_indices):
                        raise ValueError("Route matrix block size mismatch from Geoapify (targets).")
                    for local_j, j in enumerate(tgt_indices):
                        if i == j:
                            continue
                        cell = row[local_j] or {}
                        dist_meters = cell.get("distance", 0)
                        if dist_meters and dist_meters > 0:
                            distance_matrix[i][j] = int(round(dist_meters))
                        else:
                            # Fallback to Haversine for this cell if distance missing/zero
                            lat_i, lng_i = locations[i]
                            lat_j, lng_j = locations[j]
                            km = haversine_km(lat_i, lng_i, lat_j, lng_j)
                            distance_matrix[i][j] = int(round(km * 1000))

        print("Geoapify route matrix (chunked) loaded successfully for OR-Tools.")

    except Exception as e:
        # If anything goes wrong with the matrix API, fall back to Haversine
        print(f"Warning: Geoapify route matrix failed ({e}), falling back to Haversine distances.")
        for i in range(num_locations):
            lat_i, lng_i = locations[i]
            for j in range(num_locations):
                if i == j:
                    continue
                lat_j, lng_j = locations[j]
                km = haversine_km(lat_i, lng_i, lat_j, lng_j)
                distance_matrix[i][j] = int(round(km * 1000))

    # 3. Vehicles
    vehicles_meta: List[Dict[str, Any]] = []
    for r in reps:
        depot_index = wh_id_to_index[r.warehouse_id]
        for trip_idx in range(1, r.max_trips_per_day + 1):
            vehicles_meta.append(
                {
                    "rep_id": r.id,
                    "warehouse_id": r.warehouse_id,
                    "trip_index_for_rep": trip_idx,
                    "start_index": depot_index,
                    "end_index": depot_index,
                }
            )
    num_vehicles = len(vehicles_meta)

    # 4. Demands and Capacities
    demands = [0] * num_locations
    for o in order_nodes:
        demands[o["node_index"]] = o["quantity"]

    # Constraint: Each trip can visit at most 5 orders (units)
    vehicle_capacities = [5] * num_vehicles
    
    # 5. Priority Penalties for Unserved Orders
    # Penalty cost should be high enough to encourage delivery of high-priority orders
    # but low enough that it doesn't force extremely long routes for low-priority ones.
    # Max distance: ~100km * 1000 = 100,000 meters. Set max penalty much higher.
    MAX_OBJECTIVE_COST = 10_000_000 
    
    penalties = [0] * num_locations
    for o in order_nodes:
        # A priority of 1.0 gets the max penalty. A priority of 0.1 gets 10% of max.
        # This makes it a minimization problem where avoiding a delivery costs
        # (MAX_OBJECTIVE_COST * priority).
        penalty = int(MAX_OBJECTIVE_COST * o["priority"])
        penalties[o["node_index"]] = penalty

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
        "penalties": penalties, # New: Penalties for OR-Tools
    }

def solve_vrp_ortools(data: Dict[str, Any]) -> Tuple[pywrapcp.RoutingModel, pywrapcp.RoutingIndexManager, Optional[pywrapcp.Assignment], Dict[str, Any]]:
    """Solves the VRP using OR-Tools, incorporating priority penalties."""
    num_locations = len(data["locations"])
    num_vehicles = data["num_vehicles"]
    starts = [v["start_index"] for v in data["vehicle_meta"]]
    ends = [v["end_index"] for v in data["vehicle_meta"]]

    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    # 1. Distance/Cost dimension
    distance_matrix = data["distance_matrix"]
    def distance_callback(from_index: int, to_index: int) -> int:
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # 2. Capacity dimension
    #demands means the demand at each node (e.g. order quantity)
    demands = data["demands"]
    def demand_callback(from_index: int) -> int:
        return demands[manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0, # Null capacity (Slack variable is 0)
        data["vehicle_capacities"],
        True, # Start cumul to 0
        "Capacity",
    )
    
    # 3. Priority Penalty (New)
    penalties = data["penalties"]
    for node in data["order_nodes"]:
        node_index = manager.NodeToIndex(node["node_index"])
        # routing.SetAllowedCheapestArc(node_index, False) # Can't just be skipped for free
        # Add the Disjunction cost (penalty) for skipping a node
        routing.AddDisjunction([node_index], penalties[node["node_index"]])

    # Search parameters (can be tuned)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(5)

    solution = routing.SolveWithParameters(search_parameters)

    # Compute stats
    optimization_stats: Dict[str, Any] = {}
    if solution is not None:
        objective_value = solution.ObjectiveValue()
        optimization_stats["objective_value"] = objective_value
        # The solver doesn't always provide a best bound depending on strategy
        # Simplified scoring: if we find a solution, assign a score.
        optimization_stats["optimization_score"] = 90.0 if objective_value else 0.0

    return routing, manager, solution, optimization_stats


def extract_trips_ortools(
    service_date: str,
    data: Dict[str, Any],
    routing: pywrapcp.RoutingModel,
    manager: pywrapcp.RoutingIndexManager,
    solution: pywrapcp.Assignment,
) -> Tuple[List[Trip], Set[str], float, List[Dict[str, Any]]]:
    """Extracts trip details, computes ETAs/real-road distance, and identifies served orders."""
    
    date_obj = datetime.fromisoformat(service_date)
    node_index_to_order_id: Dict[int, str] = {o["node_index"]: o["id"] for o in data["order_nodes"]}
    vehicles_meta = data["vehicle_meta"]
    locations = data["locations"]
    wh_id_to_idx = data["warehouse_id_to_index"]

    served_order_ids: Set[str] = set()
    temp_trips = []
    legs_to_fetch = set()

    # 1. Collect all served order IDs and unique legs
    for vehicle_id in range(len(vehicles_meta)):
        index = routing.Start(vehicle_id)
        meta = vehicles_meta[vehicle_id]
        
        seq_nodes: List[int] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node in node_index_to_order_id:
                seq_nodes.append(node)
                served_order_ids.add(node_index_to_order_id[node])
            index = solution.Value(routing.NextVar(index))

        if not seq_nodes: continue

        depot_idx = wh_id_to_idx[meta["warehouse_id"]]
        
        # Collect legs: Depot -> First, Inter-stops, Last -> Depot
        trip_locations = [locations[depot_idx]] + [locations[n] for n in seq_nodes] + [locations[depot_idx]]
        
        trip_legs = []
        for i in range(len(trip_locations) - 1):
            coord1 = trip_locations[i]
            coord2 = trip_locations[i+1]
            # Key for cache: (lat1, lng1, lat2, lng2)
            leg_key = (coord1[0], coord1[1], coord2[0], coord2[1]) 
            legs_to_fetch.add(leg_key)
            trip_legs.append(leg_key) # Store keys for easy lookup later
        
        temp_trips.append({
            "meta": meta,
            "seq_nodes": seq_nodes,
            "trip_legs": trip_legs,
        })

    # 2. Fetch legs in parallel (using cache)
    leg_cache: Dict[Tuple[float, float, float, float], Dict[str, Any]] = {}
    legs_to_fetch_filtered = []
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
    final_trips: List[Trip] = []
    total_dist_km = 0.0
    geo_routes: List[Dict[str, Any]] = []
    service_time_minutes = 5
    avg_speed_kmph = 20.0
    km_per_min = avg_speed_kmph / 60.0 if avg_speed_kmph > 0 else 0.0

    for t_data in temp_trips:
        meta = t_data["meta"]
        seq_nodes = t_data["seq_nodes"]
        
        wh_id = meta["warehouse_id"]
        rep_id = meta["rep_id"]
        trip_index_for_rep = meta["trip_index_for_rep"]

        # Assume 9 AM start, with 2 hours gap between trips
        start_time = date_obj.replace(
            hour=9, minute=0, second=0, microsecond=0
        ) + timedelta(hours=(trip_index_for_rep - 1) * 2)
        current_time = start_time
        
        stops: List[TripStop] = []
        trip_segments: List[Dict[str, Any]] = []
        trip_dist_km = 0.0

        # Iterate through the legs (Depot -> O1, O1 -> O2, ..., ON -> Depot)
        for leg_key in t_data["trip_legs"]:
            lat1, lng1, lat2, lng2 = leg_key
            geo_data = leg_cache.get(leg_key)
            
            if geo_data:
                # Use Geoapify real-road data
                duration_sec = geo_data['properties'].get('time', 0)
                dist_meters = geo_data['properties'].get('distance', 0)
                travel_minutes = duration_sec / 60.0
                leg_km = dist_meters / 1000.0
                trip_segments.append(geo_data)
            else:
                # Fallback to Haversine estimate
                leg_km = haversine_km(lat1, lng1, lat2, lng2)
                travel_minutes = leg_km / km_per_min if km_per_min > 0 else 0.0
            
            current_time += timedelta(minutes=travel_minutes)
            trip_dist_km += leg_km
            
            # Identify if the destination is an order stop (not the final depot return)
            is_order_stop = False
            for node in seq_nodes:
                if locations[node] == (lat2, lng2):
                    is_order_stop = True
                    order_id = node_index_to_order_id[node]
                    stops.append(TripStop(order_id=order_id, eta=current_time))
                    current_time += timedelta(minutes=service_time_minutes)
                    break
        
        # The final current_time after all legs is the end time at the depot
        end_time = current_time 
        total_dist_km += trip_dist_km
        trip_duration_minutes = (end_time - start_time).total_seconds() / 60.0

        final_trips.append(
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

    return final_trips, served_order_ids, total_dist_km, geo_routes


def plan_routes_daily(
    service_date: str,
    datastore: DataStore
) -> Tuple[Dict[str, Any], float]:
    """
    Main planning function. Loads orders, calculates capacity, solves VRP, 
    and updates carried-over orders.
    """
    # 1. Load Data
    warehouses, reps, orders = datastore.load_data()
    
    # 2. Capacity Planning & Order Prioritization
    num_vehicles_capacity = sum(r.max_trips_per_day for r in reps)
    per_vehicle_capacity = 5
    total_capacity_units = num_vehicles_capacity * per_vehicle_capacity

    # Sort by priority (high first), then by smaller quantity to pack better.
    orders.sort(key=lambda x: (-x.priority, x.quantity))

    served_orders: List[Order] = []
    carried_over_orders: List[Order] = []
    used_capacity = 0

    for o in orders:
        if used_capacity + o.quantity <= total_capacity_units:
            served_orders.append(o)
            used_capacity += o.quantity
        else:
            carried_over_orders.append(o)

    print(f"Planning for {service_date}: {len(served_orders)} orders served (capacity used: {used_capacity}/{total_capacity_units}).")
    if carried_over_orders:
        print(f"WARNING: {len(carried_over_orders)} orders carried over due to capacity/time constraints.")

    # 3. Build Model & Solve
    data = build_data_model_ortools(warehouses, reps, served_orders)
    routing, manager, solution, optimization_stats = solve_vrp_ortools(data)
    
    if solution is None:
        raise RuntimeError("No feasible solution found by OR-Tools.")

    # 4. Extract Trips & Metrics
    trips, served_ids, total_dist_km, geo_routes = extract_trips_ortools(
        service_date, data, routing, manager, solution
    )
    
    # OR-Tools might leave out some 'served_orders' if it's too costly (due to penalties)
    # Filter the served_orders list to only include those actually routed.
    final_served_orders = [o for o in served_orders if o.id in served_ids]
    
    # Add un-routed orders back to the carried-over list for priority aging
    unrouted_orders = [o for o in served_orders if o.id not in served_ids]
    carried_over_orders.extend(unrouted_orders)
    
    # 5. Priority Aging and Persistence (Crucial new feature)
    next_day_orders = datastore.update_orders_priority(served_ids, orders)

    # 6. Format Result
    res_trips: List[Dict[str, Any]] = [
        {
            "warehouse_id": t.warehouse_id,
            "rep_id": t.rep_id,
            "trip_index_for_rep": t.trip_index_for_rep,
            "start_time": t.start_time.isoformat(),
            "end_time": t.end_time.isoformat(),
            "duration_minutes": round(t.duration_minutes, 1),
            "stops": [{"order_id": s.order_id, "eta": s.eta.isoformat()} for s in t.stops],
        } for t in trips
    ]
    
    # Format carried over orders for API response
    res_carried_over: List[Dict[str, Any]] = [
        {
            "id": o.id,
            "priority": round(o.priority, 3), 
            "quantity": o.quantity,
            "store_lat": o.store.lat,
            "store_lng": o.store.lng
        } for o in carried_over_orders
    ]

    result: Dict[str, Any] = {
        "service_date": service_date,
        "warehouses": [wh.__dict__ for wh in warehouses],
        "orders": [o.__dict__ for o in final_served_orders],
        "trips": res_trips,
        "geo_routes": geo_routes,
        "total_distance_km": round(total_dist_km, 2),
        "optimization_stats": optimization_stats,
        "carried_over_orders": res_carried_over,
        "next_day_orders_count": len(next_day_orders)
    }

    return result, total_dist_km

# ----------------- Flask App Setup -----------------

app = Flask(__name__)
CORS(app)
data_store = DataStore(WAREHOUSES_FILE, ORDERS_FILE)


@app.route("/api/status", methods=["GET"])
def get_status():
    """Check if files are uploaded and ready."""
    return jsonify({
        "warehouses_uploaded": os.path.exists(WAREHOUSES_FILE),
        "orders_uploaded": os.path.exists(ORDERS_FILE),
        "ready": os.path.exists(WAREHOUSES_FILE) and os.path.exists(ORDERS_FILE)
    })

# NOTE: Simplified upload endpoints (removed validation logic for brevity,
# assuming a robust client/front-end handles it, but kept file saving.)

@app.route("/api/upload/warehouses", methods=["POST"])
def upload_warehouses():
    """Upload warehouses and reps JSON data."""
    try:
        data = request.get_json()
        if not data: return jsonify({"error": "No JSON data provided"}), 400
        
        # Placeholder for complex validation (removed for brevity)
        if 'warehouses' not in data or 'reps' not in data:
            return jsonify({"error": "Missing 'warehouses' or 'reps'"}), 400
            
        with open(WAREHOUSES_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        return jsonify({"message": "Warehouses and reps uploaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload/orders", methods=["POST"])
def upload_orders():
    """Upload initial orders JSON data."""
    try:
        data = request.get_json()
        if not data: return jsonify({"error": "No JSON data provided"}), 400
        
        # Simple check for 'orders' key
        if 'orders' not in data:
            return jsonify({"error": "Missing 'orders' key"}), 400

        # Create Order objects from raw data to normalize priority/quantity
        initial_orders = [
            Order(
                id=d['id'],
                store=Store(**d['store']),
                warehouse_candidates=d['warehouse_candidates'],
                priority=float(d.get('priority', 0.5)),
                quantity=int(d.get('quantity', 1))
            ) for d in data['orders']
        ]
        
        data_store.save_orders(initial_orders)
        
        return jsonify({"message": "Orders uploaded successfully", "orders_count": len(initial_orders)})
    except Exception as e:
        return jsonify({"error": f"Error processing orders: {e}"}), 500

@app.route("/api/calculate-routes", methods=["POST"])
def calculate_routes_api():
    """
    Main API endpoint. Calculates routes for a specified day, 
    persists carried-over orders, and updates their priority.
    """
    try:
        request_data = request.get_json() or {}
        # Allows for explicit date or defaults to today's date in 'YYYY-MM-DD'
        service_date = request_data.get('service_date', datetime.now().strftime('%Y-%m-%d'))
        
        # The core planning logic is encapsulated here:
        result, _ = plan_routes_daily(service_date, data_store)
        
        # The response already contains all necessary information including the 
        # carried_over_orders and the total_distance_km.
        return jsonify(result)
        
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)