# QWERTY01 Codebase Index

## Project Overview
A full-stack delivery route optimization application featuring:
- **Backend**: Python Flask API with Google OR-Tools VRP solver + Geoapify routing integration
- **Frontend**: React + Vite with interactive map visualization, filtering, and PDF export
- **Purpose**: Optimize delivery routes for multiple warehouses and representatives in Hyderabad
- **Key Features**: Real-road routing, parallel API requests, collapsible route details, PDF report generation

---

## Project Structure

```
/home/abhi/QWERTY01/
├── app.py                    # Main Flask backend
├── test.py                   # Geoapify API testing script
└── frontend/                 # React frontend application
    ├── package.json          # Node dependencies
    ├── package-lock.json
    ├── vite.config.js        # Vite build configuration
    ├── index.html            # HTML entry point
    ├── eslint.config.js      # ESLint configuration
    ├── README.md
    ├── public/
    │   └── vite.svg
    └── src/
        ├── main.jsx          # React entry point
        ├── App.jsx           # Main app component
        ├── App.css
        ├── index.css         # Global styles with design tokens
        ├── FilterPanel.jsx   # Warehouse/rep/trip filtering
        ├── FilterPanel.css
        ├── StatsPanel.jsx    # Statistics dashboard
        ├── StatsPanel.css
        ├── RouteMap.jsx      # Interactive Leaflet map
        ├── ProgressBar.jsx   # Loading progress indicator
        ├── ProgressBar.css
        └── assets/
            └── react.svg
```

---

## Backend (Python)

### app.py - Route Optimization Engine
**Purpose**: Main Flask application serving the route optimization API

**Key Components**:

1. **Domain Models** (dataclasses):
   - `Warehouse`: id, lat, lng
   - `Rep`: id, warehouse_id, max_trips_per_day
   - `Store`: id, lat, lng
   - `Order`: id, store, warehouse_candidates
   - `TripStop`: order_id, eta
   - `Trip`: warehouse_id, rep_id, trip_index_for_rep, stops, timing

2. **Geometry Utilities**:
   - `haversine_km()`: Calculate great-circle distance between lat/lng points
   - Uses WGS84 earth radius (6371 km)

3. **Geoapify Integration**:
   - `geoapify_route_leg()`: Fetch real-road routing data from Geoapify API
   - Global cache: `GLOBAL_LEG_CACHE` prevents redundant API calls
   - Parallel fetching using ThreadPoolExecutor (20 workers)

4. **OR-Tools VRP Solver**:
   - `build_data_model_ortools()`: Constructs distance matrix and vehicle capacity model
   - `solve_vrp_ortools()`: Solves vehicle routing problem
     - Uses PATH_CHEAPEST_ARC first solution strategy
     - GUIDED_LOCAL_SEARCH metaheuristic
     - 5-second time limit
   - `extract_trips_ortools()`: Converts solution to Trip objects with ETAs
     - Combines Geoapify routing geometry (if available) with Haversine fallback
     - Calculates trip duration, start/end times
     - 20 km/h average speed assumption

5. **Demo Data Generation**:
   - `generate_demo_payload()`: Creates deterministic test dataset
- 2 warehouses: WH1 (Kukatpally 17.49°N, 78.40°E), WH2 (Ameerpet 17.44°N, 78.45°E)
   - 5 reps per warehouse (10 total), 4 trips max per rep
   - 100 orders in 10x10 grid over Hyderabad bbox (17.25-17.55°N, 78.30-78.60°E)
   - 5 orders per trip capacity

6. **Flask Endpoint**:
   - `GET /api/routes`: Returns complete routing solution
     - Input: Generated demo payload
     - Output: trips array, warehouse/order locations, total distance, service date

**Key Dependencies**:
- `ortools`: Constraint solver for VRP
- `requests`: HTTP calls to Geoapify API
- `flask`, `flask_cors`: Web framework

**Configuration**:
- Geoapify API key: `fc9a8604115d4dc5ad9a5126313eb1c1`
- Service date: `2025-12-02`
- Start time: 9:00 AM + 2 hours per trip index
- Service time per stop: 5 minutes

---

### test.py - API Testing Script
**Purpose**: Standalone test for Geoapify geocoding + routing

**Functions**:
- `geocode(place)`: Geocode place name to lat/lon
- `route(lat1, lon1, lat2, lon2)`: Get routing distance and time
- `main()`: Tests route from Kukatpally to Nampally metro, Hyderabad

---

## Frontend (React + Vite)

### Configuration Files

**package.json**:
- React 19.2.0, React DOM 19.2.0
- Build tool: Vite 5.4.21
- UI libraries: React-Leaflet 5.0.0, Leaflet 1.9.4
- HTTP client: Axios 1.13.2
- PDF generation: jsPDF 2.5.2, jspdf-autotable 3.8.4
- Dev tools: ESLint 9.39.1, Vite React plugin

**vite.config.js**:
- Minimal config with React plugin for Fast Refresh

**index.html**:
- Standard React entry point with root div and main.jsx module

**eslint.config.js**:
- Extends: js.configs.recommended + react-hooks + react-refresh
- Target: ECMAScript 2020, JSX support
- Custom rule: Ignore unused vars starting with uppercase (constants)

**index.css - Design System**:
- Color palette: Dark theme with blues/purples
  - Primary: #3b82f6 (blue)
  - Accent: #8b5cf6 (purple)
  - Success/warning/error colors defined
- CSS variables for spacing (xs → 2xl), border-radius, colors
- Leaflet container and scrollbar styling
- Global utility classes (.glass, .card)

---

### Core Components

**main.jsx**:
- React StrictMode entry point
- Mounts App component to #root

**App.jsx - Main Application**:
- **State Management**:
  - `data`: Full route optimization response
  - `loading`, `error`: Async state
  - Filter state: selectedWarehouses, selectedReps, selectedTrips
  - Progress indicator (0-100%)
  
- **Data Fetching**:
  - GET `http://localhost:5000/api/routes` with 60-second timeout
  - Simulated progress bar (10% every 200ms)
  
- **Filtering Logic**:
  - `filteredTrips` memoized selector combining all filter states
  - Initialize filters to show all data on load
  
- **Layout**:
  - Header with title, badge showing active route count
  - Two-column layout: sidebar (filters + stats) + main (map)
  - Error state with retry button
  - Loading state with progress indicator

**App.css**:
- Responsive grid layout (1800px max-width)
- Sidebar: 350px fixed, scrollable
- Header with gradient text and badge styling
- Error container with centered content
- Mobile: Stacks layout vertically, reduces sidebar height

---

**FilterPanel.jsx - Filtering UI**:
- **Sections**:
  1. Warehouses: Checkbox list of all warehouses
  2. Representatives: Scrollable list (180px max-height)
  3. Trips: ~~Commented out~~ (removed from view, but logic remains in App.jsx)
  
- **Actions**:
  - Select All / Clear All buttons
  - Individual warehouse and rep toggles
  - Trip filtering handled automatically based on warehouse/rep selection
  
- **Styling** (FilterPanel.css):
  - Custom checkbox with gradient background when checked (primary → accent)
  - Hover effects: background changes to surface color, label color lightens
  - Scrollable sections with custom thin scrollbar (4px width)
  - Section separators with bottom borders
  - Button styling: uppercase text with hover animation

---

**StatsPanel.jsx - Analytics Dashboard**:
- **Stats Grid** (4 cards):
  - Total Trips (with icon)
  - Total Distance (km, with icon)
  - Representatives count (with icon)
  - Total Orders (with icon)
  
- **Breakdown Sections**:
  - Warehouse breakdown: trips count + stops per warehouse
  - Time statistics: avg trip duration, total delivery hours, operation window
  
- **PDF Export Feature**:
  - `handleExportPdf()` function generates comprehensive PDF report
  - Report sections: header band, summary table, trips table, detailed route timelines
  - Uses jsPDF and jspdf-autotable for structured PDF generation
  - Visual timeline with dots and connecting lines for each stop
  - Includes all trip metadata: ETA, order IDs, store IDs, duration
  - Auto-pagination for multi-page reports
  - Filename format: `delivery-routes-{service_date}.pdf`
  
- **Dynamic Display**:
  - Shows "Showing X of Y trips" when filtered
  - Formatted times with 12-hour format
  - Distance and duration rounded appropriately
  - Filter status indicator at bottom
  
- **Styling** (StatsPanel.css):
  - Single-column layout for stat cards with gradient icons
  - Hover effect: translateY(-2px) + shadow
  - Pills for value badges with muted background
  - Export button with gradient hover effect

---

**RouteMap.jsx - Interactive Leaflet Map**:
- **Map Features**:
  - Centered on first warehouse (default: Hyderabad area)
  - Zoom level: 13, scroll wheel enabled
  - OpenStreetMap tiles
  
- **Markers**:
  - Warehouses: Red circles (radius 10px, 80% opacity)
  - Orders: Blue circles (radius 5px, 60% opacity)
  - Popups on click showing ID/store info
  
- **Routes**:
  - Polylines for each trip in distinct colors (21-color cycle)
  - Uses Geoapify geometry if available (real-road paths)
  - Fallback: Straight lines (warehouse → stops → warehouse)
  
- **Geometry Handling**:
  - Converts Geoapify GeoJSON coordinates from [lng, lat] to Leaflet [lat, lng]
  - Handles nested coordinate arrays from routing API
  - Extracts complete polyline from multi-segment routes
  
- **Popups**:
  - Trip details: rep ID, trip index, stops count, start/end times, duration
  - Timestamps formatted to 12-hour format

---

**ProgressBar.jsx - Loading Indicator**:
- **Visual Elements**:
  - SVG spinner with CSS animation
  - Progress bar with gradient fill + shimmer effect
  - Percentage display
  
- **Status Messages**:
  - Dynamic hint: mentions warehouses/Geoapify/database based on progress
  - Generic "Loading Route Data" title
  
- **Styling** (ProgressBar.css):
  - Full-screen overlay (z-index: 9999)
  - Gradient background (primary → secondary)
  - Spinner: 2s rotate animation + dash animation
  - Shimmer effect on progress fill

---

## Data Flow

1. **Frontend loads** → App component mounts
2. **Fetches `/api/routes`** → Backend generates demo payload
3. **Backend processes**:
   - Builds OR-Tools data model with Haversine distances
   - Solves VRP (5-second limit)
   - Extracts trips, fetches Geoapify routing in parallel (20 workers)
   - Caches Geoapify results globally
   - Returns response with trips, geo_routes, warehouse/order locations
4. **Frontend filters** → Memoizes filtered trips based on selections
5. **RouteMap renders** → Displays warehouses, orders, and colored polylines
6. **User interacts** → Filters update → Map re-renders

---

## API Response Structure

```json
{
  "warehouses": [
    {"id": "WH1", "lat": 17.49, "lng": 78.40},
    {"id": "WH2", "lat": 17.44, "lng": 78.45}
  ],
  "orders": [
    {
      "id": "O1",
      "store": {"id": "S1", "lat": 17.25, "lng": 78.30},
      "warehouse_candidates": ["WH1", "WH2"]
    }
  ],
  "trips": [
    {
      "warehouse_id": "WH1",
      "rep_id": "WH1-R1",
      "trip_index_for_rep": 1,
      "start_time": "2025-12-02T09:00:00",
      "end_time": "2025-12-02T10:15:30",
      "duration_minutes": 75.5,
      "stops": [
        {"order_id": "O1", "eta": "2025-12-02T09:05:00"},
        {"order_id": "O2", "eta": "2025-12-02T09:15:00"}
      ]
    }
  ],
  "geo_routes": [
    {
      "rep_id": "WH1-R1",
      "trip_index_for_rep": 1,
      "segments": [
        {
          "properties": {"distance": 5000, "time": 900},
          "geometry": {"coordinates": [...]}
        }
      ]
    }
  ],
  "total_distance_km": 324.5,
  "service_date": "2025-12-02"
}
```

---

## Technology Stack Summary

| Layer | Technology | Version |
|-------|-----------|---------|
| Backend | Python | 3.12+ |
| API Framework | Flask | Latest |
| Optimization | OR-Tools | 9.14.6206 |
| Routing API | Geoapify | (Cloud-based) |
| Frontend Framework | React | 19.2.0 |
| Build Tool | Vite | 5.4.21 |
| Map Library | Leaflet + React-Leaflet | 1.9.4, 5.0.0 |
| HTTP Client | Axios | 1.13.2 |
| PDF Generation | jsPDF | 2.5.2 |
| PDF Tables | jspdf-autotable | 3.8.4 |
| Styling | CSS Variables | (Custom) |
| Linting | ESLint | 9.39.1 |

---

## Running the Application

1. **Backend**:
   ```bash
   cd /home/abhi/QWERTY01
   python app.py
   # Runs on http://localhost:5000
   ```

2. **Frontend**:
   ```bash
   cd /home/abhi/QWERTY01/frontend
   npm install
   npm run dev
   # Typically runs on http://localhost:5173
   ```

3. **Testing** (optional):
   ```bash
   python test.py
   # Tests Geoapify geocoding + routing
   ```

---

## Key Features

- ✅ Multi-warehouse route optimization
- ✅ Multiple representatives with trip limits
- ✅ Real-road routing via Geoapify (with Haversine fallback)
- ✅ Interactive map visualization with filtering
- ✅ Detailed trip analytics and statistics
- ✅ PDF report generation with detailed route timelines
- ✅ Parallel API request optimization (20 workers)
- ✅ Global caching for Geoapify API responses
- ✅ Dark theme UI with gradient accents
- ✅ Responsive design (desktop-first)
- ✅ Simulated loading progress with status hints

---

## Notes

- **Geoapify API key** is embedded in both `app.py` (line 33) and `test.py` (line 10)
- **Caching strategy**: Global `GLOBAL_LEG_CACHE` persists across requests in a single Flask session
- **Fallback behavior**: If Geoapify fails, uses Haversine distance + fixed speed assumption (20 km/h)
- **Demo data**: Deterministic 100-order grid (10x10); easily swappable for real data
- **Frontend timeout**: 60 seconds; retry button available on error
- **Trip sequencing**: OR-Tools assigns orders to trips; Geoapify provides real-road geometry
- **Virtual environment**: Python dependencies installed in `.venv/` directory
- **Frontend build**: Production build outputs to `frontend/dist/` directory
- **Reps configuration**: Currently 5 reps per warehouse (WH1-R1 through WH1-R5, WH2-R1 through WH2-R5)
- **Trip capacity**: Each vehicle can handle maximum 5 orders per trip
- **Service time**: Fixed 5 minutes per stop for deliveries
- **PDF export**: Includes color-coded timeline visualization with auto-pagination for large reports
