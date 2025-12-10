import json
import random
from datetime import datetime, timedelta
import requests

API_KEY = "fc9a8604115d4dc5ad9a5126313eb1c1"


LAT_MIN = 17.20
LAT_MAX = 17.60
LNG_MIN = 78.30
LNG_MAX = 78.75


def snap_to_road(lat, lng):
    url = f"https://api.geoapify.com/v1/snap-to-road?lat={lat}&lon={lng}&type=nearest&apiKey={API_KEY}"
    r = requests.get(url).json()

    if "features" in r and len(r["features"]) > 0:
        coords = r["features"][0]["geometry"]["coordinates"]
        return coords[1], coords[0]  # (lat, lng)
    
    return lat, lng  # fallback


def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def fix_coordinates(lat, lng):
    lat = clamp(lat, LAT_MIN, LAT_MAX)
    lng = clamp(lng, LNG_MIN, LNG_MAX)
    return lat, lng



def preprocess_coordinate(lat, lng, snap=False):
    # Step 1: clamp
    lat, lng = fix_coordinates(lat, lng)
    
    # Step 2: snap if needed
    if snap:
        lat, lng = snap_to_road(lat, lng)

    return lat, lng


# load existing orders.json
with open("orders.json", "r") as f:
    data = json.load(f)

orders = data["orders"]
existing_count = len(orders)
target_total = 1000   # final count
new_orders_to_add = target_total - existing_count

priority_choices = [0.4, 0.5, 0.6]
warehouse_candidates = ["WH1", "WH2"]

base_date = datetime.strptime(data["metadata"]["base_date"], "%Y-%m-%d")

for i in range(existing_count + 1, target_total + 1):
    day = random.randint(1, 7)
    date = (base_date + timedelta(days=day - 1)).strftime("%Y-%m-%d")

    new_order = {
        "id": f"O{i}",
        "store": {
            "id": f"S{i}",
            "lat": round(random.uniform(17.25, 17.55), 3),
            "lng": round(random.uniform(78.35, 78.70), 3),
        },
        "warehouse_candidates": warehouse_candidates,
        "priority": random.choice(priority_choices),
        "day": day,
        "quantity": random.randint(1, 4),
        "date": date
    }

    orders.append(new_order)


for order in orders:
    lat = order["store"]["lat"]
    lng = order["store"]["lng"]

    # Set snap=True only if you want overhead API calls
    new_lat, new_lng = preprocess_coordinate(lat, lng, snap=False)

    order["store"]["lat"] = new_lat
    order["store"]["lng"] = new_lng
    
# update metadata
data["metadata"]["total_orders"] = len(orders)

# save back
with open("orders-expanded.json", "w") as f:
    json.dump(data, f, indent=2)

print("Mock data generated successfully -> orders-expanded.json")
