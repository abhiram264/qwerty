import React, { useState, useMemo } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import RouteMap from './RouteMap';
import FilterPanel from './FilterPanel';
import StatsPanel from './StatsPanel';
import './App.css';

function ResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [data, setData] = useState(location.state?.data);
  const [simulating, setSimulating] = useState(false);

  // Filter states
  const [selectedWarehouses, setSelectedWarehouses] = useState(
    data?.warehouses?.map(w => w.id) || []
  );
  const [selectedReps, setSelectedReps] = useState(
    data ? [...new Set(data.trips.map(t => t.rep_id))] : []
  );
  const [selectedTrips, setSelectedTrips] = useState(
    data?.trips?.map(t => `${t.rep_id}-${t.trip_index_for_rep}`) || []
  );

  // Memoized filtered trips
  const filteredTrips = useMemo(() => {
    if (!data) return [];

    return data.trips.filter(trip => {
      const tripKey = `${trip.rep_id}-${trip.trip_index_for_rep}`;
      return (
        selectedWarehouses.includes(trip.warehouse_id) &&
        selectedReps.includes(trip.rep_id) &&
        selectedTrips.includes(tripKey)
      );
    });
  }, [data, selectedWarehouses, selectedReps, selectedTrips]);

  // Reps visible under current warehouse selection
  const visibleReps = useMemo(() => {
    if (!data) return [];
    const set = new Set();
    data.trips.forEach(trip => {
      if (selectedWarehouses.length && !selectedWarehouses.includes(trip.warehouse_id)) return;
      set.add(trip.rep_id);
    });
    return [...set];
  }, [data, selectedWarehouses]);

  // Trips visible under current warehouse + rep selection
  const visibleTripsForFilter = useMemo(() => {
    if (!data) return [];
    return data.trips.filter(trip => {
      if (selectedWarehouses.length && !selectedWarehouses.includes(trip.warehouse_id)) return false;
      if (selectedReps.length && !selectedReps.includes(trip.rep_id)) return false;
      return true;
    });
  }, [data, selectedWarehouses, selectedReps]);

  // Calculate next day's date
  const getNextDay = (currentDate) => {
    const date = new Date(currentDate);
    date.setDate(date.getDate() + 1);
    return date.toISOString().split('T')[0];
  };

  // Calculate current day number from service_date for display
  const getCurrentDay = () => {
    if (data?.day) return data.day;
    // Calculate day of week (1 = Monday, 7 = Sunday)
    const date = new Date(data.service_date);
    const dayOfWeek = date.getDay(); // 0 = Sunday, 1 = Monday, etc.
    return dayOfWeek === 0 ? 7 : dayOfWeek; // Convert to 1-7 format
  };

  // Simulate next day's delivery
  const handleSimulateNextDay = async () => {
    if (!data || simulating) return;
    
    setSimulating(true);
    try {
      // Simply increment the date - the backend will filter orders by date
      const nextDate = getNextDay(data.service_date);
      
      const response = await axios.post('http://localhost:5000/api/calculate-routes', {
        service_date: nextDate
      });
      
      // Update data with new day's results
      setData(response.data);
      
      // Reset filters to show all
      setSelectedWarehouses(response.data.warehouses?.map(w => w.id) || []);
      setSelectedReps([...new Set(response.data.trips.map(t => t.rep_id))] || []);
      setSelectedTrips(response.data.trips?.map(t => `${t.rep_id}-${t.trip_index_for_rep}`) || []);
    } catch (err) {
      console.error('Error simulating next day:', err);
      alert(err.response?.data?.error || 'Failed to simulate next day');
    } finally {
      setSimulating(false);
    }
  };

  if (!data) {
    return (
      <div className="error-container">
        <div className="error-content">
          <div className="error-icon">⚠️</div>
          <h2>No Data Available</h2>
          <p>Please upload files and calculate routes first.</p>
          <Link to="/" className="btn-primary">
            Go to Upload Page
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="header-title">
            <h1>Delivery Route Planner</h1>
            <p className="header-subtitle">
              AI-Optimized Logistics for {data.service_date}
              {data.day && (
                <span className="day-badge"> • Day {data.day} {
                  ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][data.day - 1]
                }</span>
              )}
            </p>
          </div>
          <div className="header-actions">
            <button
              className="btn-simulate"
              onClick={handleSimulateNextDay}
              disabled={simulating}
              title="Simulate next day's delivery with spillover orders"
            >
              {simulating ? (
                <>
                  <div className="spinner-small"></div>
                  Simulating...
                </>
              ) : (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="9 18 15 12 9 6"></polyline>
                  </svg>
                  Simulate Next Day
                </>
              )}
            </button>
            <Link to="/" className="btn-secondary">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
              Upload New Files
            </Link>
            <div className="header-badge">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
              </svg>
              <span>{filteredTrips.length} Active Routes</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <div className="app-layout">
        {/* Sidebar */}
        <aside className="app-sidebar">
          <FilterPanel
            warehouses={data.warehouses}
            reps={visibleReps}
            trips={visibleTripsForFilter}
            selectedWarehouses={selectedWarehouses}
            setSelectedWarehouses={setSelectedWarehouses}
            selectedReps={selectedReps}
            setSelectedReps={setSelectedReps}
            selectedTrips={selectedTrips}
            setSelectedTrips={setSelectedTrips}
          />
          <StatsPanel data={data} filteredTrips={filteredTrips} />
        </aside>

        {/* Map Area */}
        <main className="app-main">
          <div className="map-wrapper">
            <RouteMap
              warehouses={data.warehouses}
              orders={data.served_orders}
              trips={filteredTrips}
              geoRoutes={data.geo_routes}
            />
          </div>
        </main>
      </div>
    </div>
  );
}

export default ResultsPage;
