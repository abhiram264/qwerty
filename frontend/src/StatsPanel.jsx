import React from 'react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import './StatsPanel.css';


const StatsPanel = ({ data, filteredTrips }) => {
    // Base collections
    const allTrips = data.trips;
    const totalTripsAll = allTrips.length;
    const tripsForStats = filteredTrips;
    const visibleTrips = tripsForStats.length;
    const isFiltered = visibleTrips > 0 && visibleTrips < totalTripsAll;

    const totalDistanceAll = data.total_distance_km;
    const optimizationScore = data.optimization_score ?? null;
    const carriedOverOrders = data.carried_over_orders || [];
    const carriedOverCount = carriedOverOrders.length;

    // Group by warehouse using filtered trips
    const warehouseStats = {};
    tripsForStats.forEach(trip => {
        if (!warehouseStats[trip.warehouse_id]) {
            warehouseStats[trip.warehouse_id] = {
                trips: 0,
                stops: 0,
                distance: 0
            };
        }
        warehouseStats[trip.warehouse_id].trips++;
        warehouseStats[trip.warehouse_id].stops += trip.stops.length;
    });

    // Group by rep (filtered)
    const repStats = {};
    tripsForStats.forEach(trip => {
        if (!repStats[trip.rep_id]) {
            repStats[trip.rep_id] = { trips: 0, orders: 0 };
        }
        repStats[trip.rep_id].trips++;
        repStats[trip.rep_id].orders += trip.stops.length;
    });

    const totalRepsAll = new Set(allTrips.map(t => t.rep_id)).size;
    const visibleReps = Object.keys(repStats).length;

    // Orders covered by filtered trips
    const allOrdersCount = data.orders.length;
    const ordersCoveredSet = new Set();
    tripsForStats.forEach(trip => {
        trip.stops.forEach(stop => ordersCoveredSet.add(stop.order_id));
    });
    const visibleOrders = ordersCoveredSet.size;

    // Map order -> store for PDF export
    const orderIdToStore = {};
    data.orders.forEach(order => {
        orderIdToStore[order.id] = order.store;
    });

    // Time statistics for filtered trips
    const totalDuration = tripsForStats.reduce((sum, trip) => sum + (trip.duration_minutes || 0), 0);
    const avgDuration = visibleTrips > 0 ? totalDuration / visibleTrips : 0;

    // Find earliest start and latest end
    const startTimes = tripsForStats.map(t => t.start_time ? new Date(t.start_time) : null).filter(Boolean);
    const endTimes = tripsForStats.map(t => t.end_time ? new Date(t.end_time) : null).filter(Boolean);
    const earliestStart = startTimes.length > 0 ? new Date(Math.min(...startTimes)) : null;
    const latestEnd = endTimes.length > 0 ? new Date(Math.max(...endTimes)) : null;

    // Calculate utilization
    const totalCapacity = visibleTrips * 5; // 5 orders per trip
    const utilizationPercent = visibleOrders > 0 ? Math.round((visibleOrders / totalCapacity) * 100) : 0;

    const filteredTripKeys = new Set(
        tripsForStats.map(trip => `${trip.rep_id}-${trip.trip_index_for_rep}`)
    );
    const filteredGeoRoutes = (data.geo_routes || []).filter(route =>
        filteredTripKeys.has(`${route.rep_id}-${route.trip_index_for_rep}`)
    );

    const handleExportPdf = () => {
        const doc = new jsPDF('p', 'mm', 'a4');
        const pageWidth = doc.internal.pageSize.getWidth();
        const pageHeight = doc.internal.pageSize.getHeight();
        const marginLeft = 15;
        const marginRight = 15;
        const marginBottom = 20;
        const contentWidth = pageWidth - marginLeft - marginRight;

        let currentY = 20;

        // ============ COVER PAGE HEADER ============
        // Gradient-like top bar with branding
        doc.setFillColor(13, 27, 42); // Navy
        doc.rect(0, 0, pageWidth, 45, 'F');

        // Logo placeholder
        doc.setFillColor(59, 130, 246); // Blue
        doc.circle(marginLeft + 8, 15, 5, 'F');

        // Main title
        doc.setTextColor(255, 255, 255);
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(28);
        doc.text('Delivery Route Report', marginLeft + 20, 22);

        // Subtitle
        doc.setFontSize(11);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(203, 213, 225);
        doc.text('Optimized Delivery Network Analysis', marginLeft + 20, 30);

        currentY = 55;

        // ============ KEY METRICS CARDS ============
        const metrics = [
            {
                label: 'Total Trips',
                value: visibleTrips.toString(),
                subValue: isFiltered ? `of ${totalTripsAll}` : '',
                icon: 'truck',
                color: [59, 130, 246] // Blue
            },
            {
                label: 'Distance',
                value: `${totalDistanceAll.toFixed(1)}`,
                subValue: 'km',
                icon: 'route',
                color: [16, 185, 129] // Green
            },
            {
                label: 'Representatives',
                value: visibleReps.toString(),
                subValue: isFiltered ? `of ${totalRepsAll}` : '',
                icon: 'people',
                color: [249, 115, 22] // Orange
            },
            {
                label: 'Orders Covered',
                value: visibleOrders.toString(),
                subValue: `of ${allOrdersCount}`,
                icon: 'pin',
                color: [139, 92, 246] // Purple
            },
            ...(optimizationScore !== null
                ? [{
                    label: 'Optimization Score',
                    value: `${optimizationScore.toFixed(1)}%`,
                    subValue: 'OR-Tools solution quality',
                    icon: 'route',
                    color: [34, 197, 94] // Green
                }]
                : [])
        ];

        // Helper function to draw icons
        const drawIcon = (type, x, y) => {
            doc.setDrawColor(255, 255, 255);
            doc.setFillColor(255, 255, 255);
            doc.setLineWidth(0.4);

            switch(type) {
                case 'truck':
                    // Truck body
                    doc.rect(x - 2.5, y - 1, 3.5, 2, 'F');
                    // Truck cabin
                    doc.rect(x + 1.2, y - 1.5, 1.5, 2.5, 'F');
                    // Wheels
                    doc.circle(x - 1.5, y + 1.3, 0.5, 'F');
                    doc.circle(x + 1.5, y + 1.3, 0.5, 'F');
                    break;

                case 'route':
                    // Winding road
                    doc.setLineWidth(0.6);
                    doc.line(x - 2, y - 1.5, x - 0.5, y);
                    doc.line(x - 0.5, y, x + 0.5, y - 0.5);
                    doc.line(x + 0.5, y - 0.5, x + 2, y + 1.5);
                    // Dots along the path
                    doc.circle(x - 2, y - 1.5, 0.3, 'F');
                    doc.circle(x, y - 0.2, 0.3, 'F');
                    doc.circle(x + 2, y + 1.5, 0.3, 'F');
                    break;

                case 'people':
                    // Two people silhouettes
                    // Person 1 head
                    doc.circle(x - 1, y - 1, 0.6, 'F');
                    // Person 1 body
                    doc.setLineWidth(1);
                    doc.line(x - 1, y - 0.2, x - 1, y + 1);
                    doc.line(x - 1.8, y + 0.5, x - 0.2, y + 0.5);
                    
                    // Person 2 head
                    doc.circle(x + 1.2, y - 0.5, 0.6, 'F');
                    // Person 2 body
                    doc.line(x + 1.2, y + 0.3, x + 1.2, y + 1.3);
                    doc.line(x + 0.4, y + 0.8, x + 2, y + 0.8);
                    break;

                case 'pin':
                    // Location pin
                    doc.setLineWidth(0.5);
                    // Pin top (circle)
                    doc.circle(x, y - 0.5, 1.5, 'S');
                    // Pin inner circle
                    doc.circle(x, y - 0.5, 0.7, 'F');
                    // Pin bottom point
                    const pinPoints = [
                        [x - 1.5, y - 0.5],
                        [x, y + 2],
                        [x + 1.5, y - 0.5]
                    ];
                    doc.setFillColor(255, 255, 255);
                    doc.triangle(x - 1, y + 0.5, x, y + 2, x + 1, y + 0.5, 'F');
                    break;
            }
        };

        // Draw metrics in 2x2 grid
        const cardWidth = (contentWidth - 5) / 2; // 5mm gap
        const cardHeight = 30;

        metrics.forEach((metric, idx) => {
            const row = Math.floor(idx / 2);
            const col = idx % 2;
            const x = marginLeft + col * (cardWidth + 5);
            const y = currentY + row * (cardHeight + 5);

            // Card background
            doc.setFillColor(248, 250, 252);
            doc.setDrawColor(226, 232, 240);
            doc.setLineWidth(0.5);
            doc.rect(x, y, cardWidth, cardHeight, 'FD');

            // Icon background
            const [r, g, b] = metric.color;
            doc.setFillColor(r, g, b);
            doc.circle(x + 6, y + 8, 4, 'F');

            // Icon glyph
            drawIcon(metric.icon, x + 6, y + 8);

            // Value
            doc.setTextColor(15, 23, 42);
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(14);
            doc.text(`${metric.value}`, x + 18, y + 8);

            // Sub-value
            if (metric.subValue) {
                doc.setFontSize(8);
                doc.setTextColor(100, 116, 139);
                doc.setFont('helvetica', 'normal');
                doc.text(metric.subValue, x + 18, y + 13);
            }

            // Label
            doc.setFontSize(9);
            doc.setFont('helvetica', 'bold');
            doc.setTextColor(71, 85, 105);
            doc.text(metric.label, x + 18, y + 22);
        });

        currentY += 70;

        // ============ SERVICE INFO ============
        doc.setFillColor(241, 245, 249);
        doc.rect(marginLeft, currentY, contentWidth, 18, 'F');

        doc.setTextColor(30, 41, 59);
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(10);
        doc.text('Service Details', marginLeft + 5, currentY + 6);

        doc.setFont('helvetica', 'normal');
        doc.setFontSize(9);
        doc.setTextColor(71, 85, 105);
        doc.text(`Service Date: ${data.service_date}`, marginLeft + 5, currentY + 12);
        doc.text(`Generated: ${new Date().toLocaleString()}`, pageWidth - marginRight - 50, currentY + 12);

        currentY += 25;

        // ============ SUMMARY STATISTICS TABLE ============
        const summaryRows = [
            ['Metric', 'Value'],
            ['Total Delivery Time', `${(totalDuration / 60).toFixed(1)} hours`],
            ['Average Trip Duration', `${avgDuration.toFixed(0)} minutes`],
            ['Fleet Utilization', `${utilizationPercent}%`],
            ['Orders per Trip (Avg)', `${(visibleOrders / (visibleTrips || 1)).toFixed(1)}`],
            ['Operation Window', `${earliestStart ? earliestStart.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : 'N/A'} - ${latestEnd ? latestEnd.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : 'N/A'}`],
            ...(optimizationScore !== null
                ? [['Optimization Score', `${optimizationScore.toFixed(1)}% (OR-Tools)`]]
                : []),
            ...(carriedOverCount > 0
                ? [['Carried Over Orders', `${carriedOverCount} (next-day capacity)`]]
                : []),
        ];

        autoTable(doc, {
            startY: currentY,
            head: [['Metric', 'Value']],
            body: summaryRows.slice(1),
            theme: 'grid',
            styles: {
                fontSize: 9,
                cellPadding: 4,
                textColor: [15, 23, 42],
                lineColor: [226, 232, 240],
                lineWidth: 0.3
            },
            headStyles: {
                fillColor: [13, 27, 42],
                textColor: [255, 255, 255],
                fontStyle: 'bold',
                fontSize: 10
            },
            alternateRowStyles: {
                fillColor: [248, 250, 252]
            },
            columnStyles: {
                0: { fontStyle: 'bold', textColor: [59, 130, 246] },
                1: { textColor: [30, 41, 59] }
            }
        });

        currentY = doc.lastAutoTable.finalY + 12;

        const repRows = Object.entries(repStats).map(([repId, stats]) => [
            repId,
            stats.trips.toString(),
            stats.orders.toString()
        ]);

        if (repRows.length > 0) {
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(11);
            doc.setTextColor(30, 41, 59);
            doc.text('Representative Allocation', marginLeft, currentY);
            currentY += 8;

            autoTable(doc, {
                startY: currentY,
                head: [['Representative', 'Routes Assigned', 'Orders']],
                body: repRows,
                theme: 'grid',
                styles: {
                    fontSize: 9,
                    cellPadding: 3.5,
                    textColor: [15, 23, 42],
                    lineColor: [226, 232, 240],
                    lineWidth: 0.3
                },
                headStyles: {
                    fillColor: [139, 92, 246],
                    textColor: [255, 255, 255],
                    fontStyle: 'bold',
                    fontSize: 9
                },
                alternateRowStyles: {
                    fillColor: [248, 250, 252]
                }
            });

            currentY = doc.lastAutoTable.finalY + 12;
        }

        // ============ WAREHOUSE BREAKDOWN ============
        const warehouseRows = Object.entries(warehouseStats).map(([whId, stats]) => [
            whId,
            stats.trips.toString(),
            stats.stops.toString(),
            (stats.stops / stats.trips).toFixed(1)
        ]);

        if (warehouseRows.length > 0) {
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(11);
            doc.setTextColor(30, 41, 59);
            doc.text('Warehouse Performance', marginLeft, currentY);
            currentY += 8;

            autoTable(doc, {
                startY: currentY,
                head: [['Warehouse', 'Trips', 'Stops', 'Avg Stops/Trip']],
                body: warehouseRows,
                theme: 'grid',
                styles: {
                    fontSize: 9,
                    cellPadding: 3.5,
                    textColor: [15, 23, 42],
                    lineColor: [226, 232, 240],
                    lineWidth: 0.3
                },
                headStyles: {
                    fillColor: [59, 130, 246],
                    textColor: [255, 255, 255],
                    fontStyle: 'bold',
                    fontSize: 9
                },
                alternateRowStyles: {
                    fillColor: [248, 250, 252]
                }
            });

            currentY = doc.lastAutoTable.finalY + 12;
        }

        // ============ TRIPS TABLE ============
        if (currentY > pageHeight - marginBottom - 40) {
            doc.addPage();
            currentY = 15;
        }

        doc.setFont('helvetica', 'bold');
        doc.setFontSize(11);
        doc.setTextColor(30, 41, 59);
        doc.text('Trip Summary', marginLeft, currentY);
        currentY += 8;

        const tripTableRows = tripsForStats.map((trip, index) => {
            const start = trip.start_time ? new Date(trip.start_time).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : '--:--';
            const end = trip.end_time ? new Date(trip.end_time).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : '--:--';
            return [
                index + 1,
                trip.warehouse_id,
                trip.rep_id,
                trip.trip_index_for_rep,
                trip.stops.length,
                start,
                end,
                trip.duration_minutes ? `${Math.round(trip.duration_minutes)}` : '0'
            ];
        });

        autoTable(doc, {
            startY: currentY,
            head: [['#', 'WH', 'Rep', 'Trip', 'Stops', 'Start', 'End', 'Duration (min)']],
            body: tripTableRows,
            theme: 'grid',
            styles: {
                fontSize: 8,
                cellPadding: 2.5,
                textColor: [15, 23, 42],
                lineColor: [226, 232, 240],
                lineWidth: 0.3
            },
            headStyles: {
                fillColor: [13, 27, 42],
                textColor: [255, 255, 255],
                fontStyle: 'bold',
                fontSize: 8
            },
            alternateRowStyles: {
                fillColor: [248, 250, 252]
            },
            columnStyles: {
                0: { halign: 'center' },
                1: { halign: 'center' },
                2: { halign: 'center' },
                3: { halign: 'center' },
                4: { halign: 'center' },
                5: { halign: 'right' },
                6: { halign: 'right' },
                7: { halign: 'right' }
            }
        });

        currentY = doc.lastAutoTable.finalY + 15;

        // ============ TRIP DETAILS (MULTI-PAGE) ============
        tripsForStats.forEach((trip, tripIdx) => {
            const stops = trip.stops;
            const neededHeight = 20 + (stops.length * 8);

            if (currentY + neededHeight > pageHeight - marginBottom) {
                doc.addPage();
                currentY = 15;
            }

            // Trip header box
            doc.setFillColor(59, 130, 246);
            doc.rect(marginLeft, currentY, contentWidth, 10, 'F');

            doc.setTextColor(255, 255, 255);
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(10);
            doc.text(`${trip.warehouse_id} • ${trip.rep_id} • Trip ${trip.trip_index_for_rep}`, marginLeft + 4, currentY + 6.5);

            const metaText = `${stops.length} stops • ${Math.round(trip.duration_minutes || 0)} min`;
            doc.setFontSize(9);
            doc.text(metaText, pageWidth - marginRight - 5, currentY + 6.5, { align: 'right' });

            currentY += 14;

            // Timeline
            const timelineX = marginLeft + 8;
            const textX = timelineX + 12;

            stops.forEach((stop, idx) => {
                const isLast = idx === stops.length - 1;
                const store = orderIdToStore[stop.order_id];
                const eta = stop.eta
                    ? new Date(stop.eta).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
                    : '--:--';

                // Vertical connector line
                if (!isLast) {
                    doc.setDrawColor(203, 213, 225);
                    doc.setLineWidth(0.5);
                    doc.line(timelineX, currentY + 2, timelineX, currentY + 8.5);
                }

                // Dot
                doc.setFillColor(59, 130, 246);
                doc.circle(timelineX, currentY + 1.5, 1.8, 'F');

                // Border circle for visual depth
                doc.setDrawColor(203, 213, 225);
                doc.setLineWidth(0.3);
                doc.circle(timelineX, currentY + 1.5, 2.5, 'S');

                // Time (bold)
                doc.setFont('courier', 'bold');
                doc.setFontSize(9);
                doc.setTextColor(59, 130, 246);
                doc.text(eta, textX, currentY + 2);

                // Order ID
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(9);
                doc.setTextColor(15, 23, 42);
                doc.text(`Order ${stop.order_id}`, textX + 25, currentY + 2);

                // Store info
                if (store && store.id) {
                    doc.setFont('helvetica', 'normal');
                    doc.setFontSize(8);
                    doc.setTextColor(107, 114, 128);
                    doc.text(`(Store ${store.id})`, textX + 50, currentY + 2);
                }

                currentY += 8;
            });

            currentY += 10; // Space after trip
        });

        // ============ FOOTER ============
        doc.setTextColor(128, 128, 128);
        doc.setFontSize(8);
        doc.setFont('helvetica', 'normal');
        const totalPages = doc.internal.getNumberOfPages();
        doc.text(
            `Delivery Route Report • Generated ${new Date().toLocaleString()} • Page ${totalPages}`,
            pageWidth / 2,
            pageHeight - 10,
            { align: 'center' }
        );

        doc.save(`delivery-routes-${data.service_date}.pdf`);
    };

    const handleDownloadJson = () => {
        const payload = {
            service_date: data.service_date,
            warehouses: data.warehouses,
            orders: data.orders,
            trips: tripsForStats,
            geo_routes: filteredGeoRoutes,
            optimization_score: optimizationScore,
            optimality_gap: data.optimality_gap ?? null,
            objective_value: data.objective_value ?? null,
            best_objective_bound: data.best_objective_bound ?? null,
            carried_over_orders: carriedOverOrders,
        };

        const jsonString = JSON.stringify(payload, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = `delivery-routes-${data.service_date}.json`;
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        URL.revokeObjectURL(url);
    };

    return (
        <div className="stats-panel">
            <div className="stats-header">
                <div className="stats-header-left">
                    <h3>Statistics</h3>
                    <span className="stats-badge">{data.service_date}</span>
                </div>
                <div className="stats-export-group">
                    <button className="stats-export-btn" type="button" onClick={handleExportPdf}>
                        <span className="stats-export-icon">⬇</span>
                        PDF Report
                    </button>
                    <button
                        className="stats-export-btn stats-export-json"
                        type="button"
                        onClick={handleDownloadJson}
                    >
                        <span className="stats-export-icon">{"{ }"}</span>
                        JSON Export
                    </button>
                </div>
            </div>

            {/* Main Stats Grid */}
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)' }}>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                            <polyline points="9 22 9 12 15 12 15 22"></polyline>
                        </svg>
                    </div>
                    <div className="stat-content">
                        <div className="stat-value">
                            {visibleTrips}
                            {isFiltered && (
                                <span className="stat-value-sub">/{totalTripsAll}</span>
                            )}
                        </div>
                        <div className="stat-label">
                            {isFiltered ? 'Trips (filtered)' : 'Total Trips'}
                        </div>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #10b981 0%, #3b82f6 100%)' }}>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <polyline points="12 6 12 12 16 14"></polyline>
                        </svg>
                    </div>
                    <div className="stat-content">
                        <div className="stat-value">{totalDistanceAll.toFixed(1)} km</div>
                        <div className="stat-label">Distance (all routes)</div>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #f59e0b 0%, #ef4444 100%)' }}>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                            <circle cx="8.5" cy="7" r="4"></circle>
                            <line x1="20" y1="8" x2="20" y2="14"></line>
                            <line x1="23" y1="11" x2="17" y2="11"></line>
                        </svg>
                    </div>
                    <div className="stat-content">
                        <div className="stat-value">
                            {visibleReps}
                            {isFiltered && (
                                <span className="stat-value-sub">/{totalRepsAll}</span>
                            )}
                        </div>
                        <div className="stat-label">
                            {isFiltered ? 'Representatives (filtered)' : 'Representatives'}
                        </div>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%)' }}>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
                            <circle cx="12" cy="10" r="3"></circle>
                        </svg>
                    </div>
                    <div className="stat-content">
                        <div className="stat-value">
                            {visibleOrders}
                            <span className="stat-value-sub">/{allOrdersCount}</span>
                        </div>
                        <div className="stat-label">Orders Covered</div>
                    </div>
                </div>

                {optimizationScore !== null && (
                    <div className="stat-card">
                        <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)' }}>
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M3 3v18h18"></path>
                                <polyline points="6 15 10 11 13 14 18 9"></polyline>
                                <polyline points="18 9 18 13 14 13"></polyline>
                            </svg>
                        </div>
                        <div className="stat-content">
                            <div className="stat-value">
                                {optimizationScore.toFixed(1)}%
                            </div>
                            <div className="stat-label">Optimization Score</div>
                        </div>
                    </div>
                )}
            </div>

            {/* Warehouse Breakdown */}
            <div className="stats-section">
                <h4 className="section-title">Warehouse Breakdown</h4>
                {Object.entries(warehouseStats).map(([whId, stats]) => (
                    <div key={whId} className="stat-row">
                        <div className="stat-row-label">{whId}</div>
                        <div className="stat-row-values">
                            <span className="stat-pill">{stats.trips} trips</span>
                            <span className="stat-pill">{stats.stops} stops</span>
                        </div>
                    </div>
                ))}
            </div>

            {/* Time Statistics */}
            <div className="stats-section">
                <h4 className="section-title">Time Statistics</h4>
                <div className="stat-row">
                    <div className="stat-row-label">Avg Trip Duration</div>
                    <div className="stat-row-values">
                        <span className="stat-pill">{avgDuration.toFixed(0)} min</span>
                    </div>
                </div>
                <div className="stat-row">
                    <div className="stat-row-label">Total Delivery Time</div>
                    <div className="stat-row-values">
                        <span className="stat-pill">{(totalDuration / 60).toFixed(1)} hrs</span>
                    </div>
                </div>
                <div className="stat-row">
                    <div className="stat-row-label">Fleet Utilization</div>
                    <div className="stat-row-values">
                        <span className="stat-pill">{utilizationPercent}%</span>
                    </div>
                </div>
                <div className="stat-row">
                    <div className="stat-row-label">Operation Window</div>
                    <div className="stat-row-values">
                        <span className="stat-pill">
                            {earliestStart ? earliestStart.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : 'N/A'}
                            {' → '}
                            {latestEnd ? latestEnd.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : 'N/A'}
                        </span>
                    </div>
                </div>
                {carriedOverCount > 0 && (
                    <div className="stat-row">
                        <div className="stat-row-label">Carried Over Orders</div>
                        <div className="stat-row-values">
                            <span className="stat-pill">{carriedOverCount} orders (next day)</span>
                        </div>
                    </div>
                )}
            </div>

            {/* Filter Status */}
            {isFiltered && (
                <div className="filter-status">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon>
                    </svg>
                    Showing {visibleTrips} of {totalTripsAll} trips
                </div>
            )}
        </div>
    );
};

export default StatsPanel;