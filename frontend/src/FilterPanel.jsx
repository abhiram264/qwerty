import React from 'react';
import './FilterPanel.css';

const FilterPanel = ({
    warehouses,
    reps,
    trips,
    selectedWarehouses,
    setSelectedWarehouses,
    selectedReps,
    setSelectedReps,
    selectedTrips,
    setSelectedTrips
}) => {
    const handleWarehouseToggle = (whId) => {
        setSelectedWarehouses(prev =>
            prev.includes(whId)
                ? prev.filter(id => id !== whId)
                : [...prev, whId]
        );
    };

    const handleRepToggle = (repId) => {
        setSelectedReps(prev =>
            prev.includes(repId)
                ? prev.filter(id => id !== repId)
                : [...prev, repId]
        );
    };

    const handleTripToggle = (tripKey) => {
        setSelectedTrips(prev =>
            prev.includes(tripKey)
                ? prev.filter(key => key !== tripKey)
                : [...prev, tripKey]
        );
    };

    const handleSelectAll = () => {
        setSelectedWarehouses(warehouses.map(w => w.id));
        setSelectedReps(reps);
        setSelectedTrips(trips.map(t => `${t.rep_id}-${t.trip_index_for_rep}`));
    };

    const handleClearAll = () => {
        setSelectedWarehouses([]);
        setSelectedReps([]);
        setSelectedTrips([]);
    };

    return (
        <div className="filter-panel">
            <div className="filter-header">
                <h3>Filters</h3>
                <div className="filter-actions">
                    <button onClick={handleSelectAll} className="btn-text">All</button>
                    <button onClick={handleClearAll} className="btn-text">None</button>
                </div>
            </div>

            {/* Warehouse Filter */}
            <div className="filter-section">
                <h4 className="filter-title">Warehouses</h4>
                <div className="filter-options">
                    {warehouses.map(wh => (
                        <label key={wh.id} className="filter-checkbox">
                            <input
                                type="checkbox"
                                checked={selectedWarehouses.includes(wh.id)}
                                onChange={() => handleWarehouseToggle(wh.id)}
                            />
                            <span className="checkbox-custom"></span>
                            <span className="checkbox-label">{wh.id}</span>
                        </label>
                    ))}
                </div>
            </div>

            {/* Rep Filter */}
            <div className="filter-section">
                <h4 className="filter-title">Representatives ({reps.length})</h4>
                <div className="filter-options scrollable">
                    {reps.map(repId => (
                        <label key={repId} className="filter-checkbox">
                            <input
                                type="checkbox"
                                checked={selectedReps.includes(repId)}
                                onChange={() => handleRepToggle(repId)}
                            />
                            <span className="checkbox-custom"></span>
                            <span className="checkbox-label">{repId}</span>
                        </label>
                    ))}
                </div>
            </div>

            {/* Trip Toggle - Removed from view as requested */ }
            {/* <div className="filter-section">
                <h4 className="filter-title">Trips ({trips.length})</h4>
                <div className="filter-options scrollable">
                    {trips.slice(0, 10).map(trip => {
                        const tripKey = `${trip.rep_id}-${trip.trip_index_for_rep}`;
                        return (
                            <label key={tripKey} className="filter-checkbox">
                                <input
                                    type="checkbox"
                                    checked={selectedTrips.includes(tripKey)}
                                    onChange={() => handleTripToggle(tripKey)}
                                />
                                <span className="checkbox-custom"></span>
                                <span className="checkbox-label">
                                    {trip.rep_id} - Trip {trip.trip_index_for_rep}
                                </span>
                            </label>
                        );
                    })}
                    {trips.length > 10 && (
                        <p className="filter-hint">+ {trips.length - 10} more trips</p>
                    )}
                </div>
            </div> */}
        </div>
    );
};

export default FilterPanel;
