import React from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
    '#ffffff', '#000000'
];

const RouteMap = ({ warehouses, orders, trips, geoRoutes }) => {
    // Calculate center of the map
    const centerLat = warehouses.length > 0 ? warehouses[0].lat : 12.900;
    const centerLng = warehouses.length > 0 ? warehouses[0].lng : 77.600;

    console.log('RouteMap Props:', {
        warehousesCount: warehouses.length,
        ordersCount: orders.length,
        tripsCount: trips.length,
        geoRoutesCount: geoRoutes ? geoRoutes.length : 0,
        sampleGeoRoute: geoRoutes && geoRoutes[0] ? geoRoutes[0] : null
    });

    // Helper to find location by ID
    const findLocation = (id, type) => {
        if (type === 'warehouse') {
            return warehouses.find(w => w.id === id);
        } else {
            // Order ID might be O1, but we need the store location
            const order = orders.find(o => o.id === id);
            return order ? order.store : null;
        }
    };

    // Helper to extract positions from Geoapify geometry
    const extractGeoPositions = (geoRoute) => {
        const positions = [];
        if (!geoRoute || !geoRoute.segments) return positions;

        for (const segment of geoRoute.segments) {
            if (segment.geometry && segment.geometry.coordinates) {
                // Geoapify returns coordinates in nested array format
                // geometry.coordinates = [[[lng, lat], [lng, lat], ...]]
                const coords = segment.geometry.coordinates;

                // Handle nested array structure
                if (Array.isArray(coords)) {
                    // coords is an array, check if it's nested
                    if (coords.length > 0 && Array.isArray(coords[0])) {
                        // It's a nested array [[[lng, lat], ...]]
                        // We need to flatten one level
                        for (const coordArray of coords) {
                            if (Array.isArray(coordArray)) {
                                for (const coord of coordArray) {
                                    if (Array.isArray(coord) && coord.length >= 2) {
                                        // Convert [lng, lat] to [lat, lng] for Leaflet
                                        positions.push([coord[1], coord[0]]);
                                    }
                                }
                            }
                        }
                    } else if (coords.length >= 2 && typeof coords[0] === 'number') {
                        // It's a simple [lng, lat] pair
                        positions.push([coords[1], coords[0]]);
                    }
                }
            }
        }

        console.log(`Extracted ${positions.length} positions from geo_route`);
        return positions;
    };

    // Create a map of trip key to geo route
    const geoRouteMap = {};
    if (geoRoutes && Array.isArray(geoRoutes)) {
        geoRoutes.forEach(gr => {
            const key = `${gr.rep_id}-${gr.trip_index_for_rep}`;
            geoRouteMap[key] = gr;
        });
    }

    return (
        <MapContainer center={[centerLat, centerLng]} zoom={13} scrollWheelZoom={true}>
            <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            {/* Warehouses */}
            {warehouses.map(wh => (
                <CircleMarker
                    key={wh.id}
                    center={[wh.lat, wh.lng]}
                    radius={10}
                    pathOptions={{ color: 'red', fillColor: 'red', fillOpacity: 0.8 }}
                >
                    <Popup>
                        Warehouse: {wh.id}
                    </Popup>
                </CircleMarker>
            ))}

            {/* Orders */}
            {orders.map(order => (
                <CircleMarker
                    key={order.id}
                    center={[order.store.lat, order.store.lng]}
                    radius={5}
                    pathOptions={{ color: 'blue', fillColor: 'blue', fillOpacity: 0.6 }}
                >
                    <Popup>
                        Order: {order.id}<br />
                        Store: {order.store.id}
                    </Popup>
                </CircleMarker>
            ))}

            {/* Routes */}
            {trips.map((trip, index) => {
                const color = COLORS[index % COLORS.length];
                const tripKey = `${trip.rep_id}-${trip.trip_index_for_rep}`;
                const geoRoute = geoRouteMap[tripKey];

                let positions = [];

                // Try to use Geoapify geometry if available
                if (geoRoute) {
                    positions = extractGeoPositions(geoRoute);
                }

                // Fallback to straight lines if no Geoapify data
                if (positions.length === 0) {
                    // Start at warehouse
                    const wh = findLocation(trip.warehouse_id, 'warehouse');
                    if (wh) positions.push([wh.lat, wh.lng]);

                    // Stops
                    trip.stops.forEach(stop => {
                        const loc = findLocation(stop.order_id, 'order');
                        if (loc) positions.push([loc.lat, loc.lng]);
                    });

                    // Return to warehouse
                    if (wh) positions.push([wh.lat, wh.lng]);
                }

                return (
                    <Polyline
                        key={tripKey}
                        positions={positions}
                        pathOptions={{ color: color, weight: 4, opacity: 0.7 }}
                    >
                        <Popup>
                            <div style={{ minWidth: '200px' }}>
                                <strong>{trip.rep_id}</strong> - Trip {trip.trip_index_for_rep}<br />
                                <hr style={{ margin: '8px 0', border: 'none', borderTop: '1px solid #ccc' }} />
                                <div style={{ fontSize: '0.85em', lineHeight: '1.6' }}>
                                    📍 <strong>Stops:</strong> {trip.stops.length}<br />
                                    ⏰ <strong>Start:</strong> {trip.start_time ? new Date(trip.start_time).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : 'N/A'}<br />
                                    ⏱️ <strong>End:</strong> {trip.end_time ? new Date(trip.end_time).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : 'N/A'}<br />
                                    ⌛ <strong>Duration:</strong> {trip.duration_minutes ? `${Math.floor(trip.duration_minutes)} min` : 'N/A'}
                                </div>
                            </div>
                        </Popup>
                    </Polyline>
                );
            })}
        </MapContainer>
    );
};

export default RouteMap;
