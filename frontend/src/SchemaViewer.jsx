import React, { useState } from 'react';
import './SchemaViewer.css';

const SchemaViewer = () => {
  const [copiedType, setCopiedType] = useState(null);

  const warehousesSchema = {
    warehouses: [
      { id: "WH1", lat: 17.49, lng: 78.40 },
      { id: "WH2", lat: 17.44, lng: 78.45 }
    ],
    reps: [
      { id: "WH1-R1", warehouse_id: "WH1", max_trips_per_day: 4 },
      { id: "WH1-R2", warehouse_id: "WH1", max_trips_per_day: 4 },
      { id: "WH2-R1", warehouse_id: "WH2", max_trips_per_day: 4 }
    ]
  };

  const ordersSchema = {
    orders: [
      {
        id: "O1",
        store: { id: "S1", lat: 17.40, lng: 78.45 },
        warehouse_candidates: ["WH1", "WH2"]
      },
      {
        id: "O2",
        store: { id: "S2", lat: 17.42, lng: 78.48 },
        warehouse_candidates: ["WH1"]
      },
      {
        id: "O3",
        store: { id: "S3", lat: 17.46, lng: 78.42 },
        warehouse_candidates: ["WH2"]
      }
    ]
  };

  const copyToClipboard = (data, type) => {
    const jsonString = JSON.stringify(data, null, 2);
    navigator.clipboard.writeText(jsonString).then(() => {
      setCopiedType(type);
      setTimeout(() => setCopiedType(null), 2000);
    });
  };

  return (
    <div className="schema-viewer">
      <div className="schema-header">
        <h2>JSON Schema Reference</h2>
        <p>Use these examples as templates for your data files</p>
      </div>

      <div className="schema-sections">
        {/* Warehouses Schema */}
        <div className="schema-section">
          <div className="schema-title">
            <div>
              <h3>Warehouses & Representatives</h3>
              <p className="schema-description">
                Define your warehouse locations and sales representatives
              </p>
            </div>
            <button
              className={`btn-copy ${copiedType === 'warehouses' ? 'copied' : ''}`}
              onClick={() => copyToClipboard(warehousesSchema, 'warehouses')}
            >
              {copiedType === 'warehouses' ? (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12"></polyline>
                  </svg>
                  Copied!
                </>
              ) : (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                  </svg>
                  Copy
                </>
              )}
            </button>
          </div>
          <pre className="schema-code">
            <code>{JSON.stringify(warehousesSchema, null, 2)}</code>
          </pre>
          <div className="schema-notes">
            <h4>Field Descriptions:</h4>
            <ul>
              <li><strong>id:</strong> Unique identifier for warehouse/rep</li>
              <li><strong>lat, lng:</strong> Geographic coordinates (latitude, longitude)</li>
              <li><strong>warehouse_id:</strong> Which warehouse this rep is assigned to</li>
              <li><strong>max_trips_per_day:</strong> Maximum number of trips per rep per day</li>
            </ul>
          </div>
        </div>

        {/* Orders Schema */}
        <div className="schema-section">
          <div className="schema-title">
            <div>
              <h3>Orders</h3>
              <p className="schema-description">
                Define your delivery orders with store locations
              </p>
            </div>
            <button
              className={`btn-copy ${copiedType === 'orders' ? 'copied' : ''}`}
              onClick={() => copyToClipboard(ordersSchema, 'orders')}
            >
              {copiedType === 'orders' ? (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12"></polyline>
                  </svg>
                  Copied!
                </>
              ) : (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                  </svg>
                  Copy
                </>
              )}
            </button>
          </div>
          <pre className="schema-code">
            <code>{JSON.stringify(ordersSchema, null, 2)}</code>
          </pre>
          <div className="schema-notes">
            <h4>Field Descriptions:</h4>
            <ul>
              <li><strong>id:</strong> Unique identifier for the order</li>
              <li><strong>store.id:</strong> Store identifier</li>
              <li><strong>store.lat, store.lng:</strong> Store geographic coordinates</li>
              <li><strong>warehouse_candidates:</strong> List of warehouses that can fulfill this order</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SchemaViewer;
