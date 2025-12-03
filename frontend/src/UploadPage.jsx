import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import SchemaViewer from './SchemaViewer';
import './UploadPage.css';

const UploadPage = () => {
  const navigate = useNavigate();
  const [warehousesFile, setWarehousesFile] = useState(null);
  const [ordersFile, setOrdersFile] = useState(null);
  const [warehousesData, setWarehousesData] = useState(null);
  const [ordersData, setOrdersData] = useState(null);
  const [uploading, setUploading] = useState({ warehouses: false, orders: false });
  const [errors, setErrors] = useState({ warehouses: null, orders: null });
  const [calculating, setCalculating] = useState(false);
  const [dragOver, setDragOver] = useState({ warehouses: false, orders: false });

  const warehousesInputRef = useRef(null);
  const ordersInputRef = useRef(null);

  const handleFileRead = (file, type) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const json = JSON.parse(e.target.result);
        if (type === 'warehouses') {
          setWarehousesData(json);
          setErrors(prev => ({ ...prev, warehouses: null }));
        } else {
          setOrdersData(json);
          setErrors(prev => ({ ...prev, orders: null }));
        }
      } catch (err) {
        setErrors(prev => ({
          ...prev,
          [type]: 'Invalid JSON file. Please check the format.'
        }));
      }
    };
    reader.readAsText(file);
  };

  const handleFileSelect = (e, type) => {
    const file = e.target.files?.[0];
    if (file) {
      if (type === 'warehouses') {
        setWarehousesFile(file);
      } else {
        setOrdersFile(file);
      }
      handleFileRead(file, type);
    }
  };

  const handleDragOver = (e, type) => {
    e.preventDefault();
    setDragOver(prev => ({ ...prev, [type]: true }));
  };

  const handleDragLeave = (e, type) => {
    e.preventDefault();
    setDragOver(prev => ({ ...prev, [type]: false }));
  };

  const handleDrop = (e, type) => {
    e.preventDefault();
    setDragOver(prev => ({ ...prev, [type]: false }));
    
    const file = e.dataTransfer.files?.[0];
    if (file) {
      if (type === 'warehouses') {
        setWarehousesFile(file);
      } else {
        setOrdersFile(file);
      }
      handleFileRead(file, type);
    }
  };

  const uploadFile = async (data, type) => {
    setUploading(prev => ({ ...prev, [type]: true }));
    setErrors(prev => ({ ...prev, [type]: null }));

    try {
      const endpoint = type === 'warehouses' ? '/api/upload/warehouses' : '/api/upload/orders';
      await axios.post(`http://localhost:5000${endpoint}`, data);
      return true;
    } catch (err) {
      const errorMsg = err.response?.data?.error || `Failed to upload ${type}`;
      setErrors(prev => ({ ...prev, [type]: errorMsg }));
      return false;
    } finally {
      setUploading(prev => ({ ...prev, [type]: false }));
    }
  };

  const handleCalculateRoutes = async () => {
    // Upload files first
    const warehousesUploaded = await uploadFile(warehousesData, 'warehouses');
    const ordersUploaded = await uploadFile(ordersData, 'orders');

    if (!warehousesUploaded || !ordersUploaded) {
      return;
    }

    // Calculate routes
    setCalculating(true);
    try {
      const response = await axios.post('http://localhost:5000/api/calculate-routes', {
        service_date: new Date().toISOString().split('T')[0]
      });
      
      // Navigate to results page with data
      navigate('/results', { state: { data: response.data } });
    } catch (err) {
      setErrors(prev => ({
        ...prev,
        calculation: err.response?.data?.error || 'Failed to calculate routes'
      }));
    } finally {
      setCalculating(false);
    }
  };

  const canCalculate = warehousesData && ordersData && !calculating;

  return (
    <div className="upload-page">
      <header className="upload-header">
        <div className="header-content">
          <h1>Route Optimization Platform</h1>
          <p>Upload your data to calculate optimal delivery routes</p>
        </div>
      </header>

      <div className="upload-container">
        {/* Upload Cards */}
        <div className="upload-cards">
          {/* Warehouses Upload */}
          <div className="upload-card">
            <div className="card-header">
              <div className="card-icon warehouse-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                  <polyline points="9 22 9 12 15 12 15 22"></polyline>
                </svg>
              </div>
              <div>
                <h3>Warehouses & Representatives</h3>
                <p>Upload warehouse locations and sales rep data</p>
              </div>
            </div>

            <div
              className={`drop-zone ${dragOver.warehouses ? 'drag-over' : ''} ${warehousesFile ? 'has-file' : ''}`}
              onDragOver={(e) => handleDragOver(e, 'warehouses')}
              onDragLeave={(e) => handleDragLeave(e, 'warehouses')}
              onDrop={(e) => handleDrop(e, 'warehouses')}
              onClick={() => warehousesInputRef.current?.click()}
            >
              <input
                ref={warehousesInputRef}
                type="file"
                accept=".json"
                onChange={(e) => handleFileSelect(e, 'warehouses')}
                style={{ display: 'none' }}
              />
              
              {warehousesFile ? (
                <div className="file-selected">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
                    <polyline points="13 2 13 9 20 9"></polyline>
                  </svg>
                  <p className="file-name">{warehousesFile.name}</p>
                  <p className="file-size">{(warehousesFile.size / 1024).toFixed(2)} KB</p>
                  {uploading.warehouses && <div className="upload-spinner"></div>}
                </div>
              ) : (
                <div className="drop-prompt">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                  </svg>
                  <p>Drag & drop your JSON file here</p>
                  <p className="or-text">or</p>
                  <button className="btn-browse">Browse Files</button>
                </div>
              )}
            </div>

            {errors.warehouses && (
              <div className="error-message">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="12" y1="8" x2="12" y2="12"></line>
                  <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                {errors.warehouses}
              </div>
            )}
          </div>

          {/* Orders Upload */}
          <div className="upload-card">
            <div className="card-header">
              <div className="card-icon orders-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                  <circle cx="8.5" cy="7" r="4"></circle>
                  <polyline points="17 11 19 13 23 9"></polyline>
                </svg>
              </div>
              <div>
                <h3>Orders</h3>
                <p>Upload delivery orders with store locations</p>
              </div>
            </div>

            <div
              className={`drop-zone ${dragOver.orders ? 'drag-over' : ''} ${ordersFile ? 'has-file' : ''}`}
              onDragOver={(e) => handleDragOver(e, 'orders')}
              onDragLeave={(e) => handleDragLeave(e, 'orders')}
              onDrop={(e) => handleDrop(e, 'orders')}
              onClick={() => ordersInputRef.current?.click()}
            >
              <input
                ref={ordersInputRef}
                type="file"
                accept=".json"
                onChange={(e) => handleFileSelect(e, 'orders')}
                style={{ display: 'none' }}
              />
              
              {ordersFile ? (
                <div className="file-selected">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
                    <polyline points="13 2 13 9 20 9"></polyline>
                  </svg>
                  <p className="file-name">{ordersFile.name}</p>
                  <p className="file-size">{(ordersFile.size / 1024).toFixed(2)} KB</p>
                  {uploading.orders && <div className="upload-spinner"></div>}
                </div>
              ) : (
                <div className="drop-prompt">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                  </svg>
                  <p>Drag & drop your JSON file here</p>
                  <p className="or-text">or</p>
                  <button className="btn-browse">Browse Files</button>
                </div>
              )}
            </div>

            {errors.orders && (
              <div className="error-message">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="12" y1="8" x2="12" y2="12"></line>
                  <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                {errors.orders}
              </div>
            )}
          </div>
        </div>

        {/* Calculate Button */}
        <div className="calculate-section">
          {errors.calculation && (
            <div className="error-message calculation-error">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
              </svg>
              {errors.calculation}
            </div>
          )}
          
          <button
            className="btn-calculate"
            onClick={handleCalculateRoutes}
            disabled={!canCalculate}
          >
            {calculating ? (
              <>
                <div className="spinner"></div>
                Calculating Optimal Routes...
              </>
            ) : (
              <>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                </svg>
                Calculate Optimal Routes
              </>
            )}
          </button>
          
          {!canCalculate && !calculating && (
            <p className="help-text">
              Please upload both files to calculate routes
            </p>
          )}
        </div>

        {/* Schema Viewer */}
        <SchemaViewer />
      </div>
    </div>
  );
};

export default UploadPage;
