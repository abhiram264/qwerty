#!/usr/bin/env python3
"""
Test Geoapify Geocoding + Routing API with one script.

Route: Kukatpally metro, Hyderabad -> Nampally metro, Hyderabad
"""

import requests

API_KEY = "fc9a8604115d4dc5ad9a5126313eb1c1"  # for testing only

GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"
ROUTING_URL = "https://api.geoapify.com/v1/routing"


def geocode(place: str):
    params = {
        "text": place,
        "apiKey": API_KEY,
        "format": "json",
    }
    resp = requests.get(GEOCODE_URL, params=params)
    print(f"Geocode '{place}' status:", resp.status_code)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        raise ValueError(f"No geocoding result for: {place}")
    lat = results[0]["lat"]
    lon = results[0]["lon"]
    print(f"{place} -> lat={lat}, lon={lon}")
    return lat, lon


def route(start_lat, start_lon, end_lat, end_lon):
    # Same structure as the JS sample: waypoints=lat,lon|lat,lon&mode=drive&apiKey=...
    waypoints = f"{start_lat},{start_lon}|{end_lat},{end_lon}"
    params = {
        "waypoints": waypoints,
        "mode": "drive",
        "apiKey": API_KEY,
    }
    print("\nRouting request:")
    print("URL:", ROUTING_URL)
    print("Params:", params)

    resp = requests.get(ROUTING_URL, params=params)
    print("Routing status:", resp.status_code)
    print("Routing raw JSON (first 1000 chars):")
    print(resp.text[:1000])
    resp.raise_for_status()

    data = resp.json()
    features = data.get("features", [])
    if not features:
        raise ValueError("No route found in features[]")

    props = features[0].get("properties", {})
    dist_m = props.get("distance", 0)
    time_s = props.get("time", 0)

    distance_km = dist_m / 1000.0
    time_min = time_s / 60.0

    print("\nParsed route:")
    print(f"Distance: {distance_km:.2f} km")
    print(f"Travel time: {time_min:.1f} minutes")

    return distance_km, time_min


def main():
    from_place = "Kukatpally metro, Hyderabad"
    to_place = "Nampally metro, Hyderabad"

    # 1) Geocode both places
    from_lat, from_lon = geocode(from_place)
    to_lat, to_lon = geocode(to_place)

    # 2) Call routing API
    print("\nRouting result:")
    route(from_lat, from_lon, to_lat, to_lon)


if __name__ == "__main__":
    main()
