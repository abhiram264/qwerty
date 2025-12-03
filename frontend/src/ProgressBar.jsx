import React from 'react';
import './ProgressBar.css';

const ProgressBar = ({ progress, status = 'Loading...' }) => {
    return (
        <div className="progress-container">
            <div className="progress-content">
                <div className="progress-icon">
                    <svg className="spinner" viewBox="0 0 50 50">
                        <circle
                            className="path"
                            cx="25"
                            cy="25"
                            r="20"
                            fill="none"
                            strokeWidth="4"
                        />
                    </svg>
                </div>
                <h2 className="progress-title">Loading Route Data</h2>
                <p className="progress-status">{status}</p>
                <div className="progress-bar-wrapper">
                    <div className="progress-bar">
                        <div
                            className="progress-bar-fill"
                            style={{ width: `${progress}%` }}
                        >
                            <div className="progress-bar-glow"></div>
                        </div>
                    </div>
                    <span className="progress-percentage">{progress}%</span>
                </div>
                <p className="progress-hint">
                    Fetching optimal routes from {progress < 30 ? 'warehouses' : progress < 60 ? 'Geoapify API' : 'database'}...
                </p>
            </div>
        </div>
    );
};

export default ProgressBar;
