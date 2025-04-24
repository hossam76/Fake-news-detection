/**
 * Dashboard JavaScript for the Fake News Detection System
 * Enhances the dashboard with interactive features and real-time updates
 */

// Initialize all charts and interactive elements when the page loads
$(document).ready(function() {
    // Set up periodic data refresh
    setInterval(refreshDashboardData, 60000); // Refresh every minute
    
    // Set up interactive filters
    setupDateRangeFilter();
    setupModelFilters();
    
    // Initialize any tooltips or popovers
    initTooltips();
});

/**
 * Refreshes the dashboard data via AJAX
 */
function refreshDashboardData() {
    $.ajax({
        url: '/api/stats',
        method: 'GET',
        success: function(data) {
            // Update statistics cards
            updateStatCards(data);
            
            // Update charts
            updateTrendChart(data.time_series);
            updateDistributionChart(data.fake_count, data.real_count);
            updateModelPerformanceChart(data.model_performance);
            
            // Show refresh notification
            showRefreshNotification();
        },
        error: function(err) {
            console.error("Error refreshing dashboard data:", err);
        }
    });
}

/**
 * Updates the statistic cards with new data
 */
function updateStatCards(data) {
    $('#totalAnalysesValue').text(data.total_analyses);
    $('#fakeNewsValue').text(data.fake_count);
    $('#realNewsValue').text(data.real_count);
    $('#avgConfidenceValue').text(data.avg_confidence.toFixed(1) + '%');
}

/**
 * Updates the trend chart with new time series data
 */
function updateTrendChart(timeSeriesData) {
    const chart = Chart.getChart('trendChart');
    if (chart) {
        chart.data.labels = timeSeriesData.dates;
        chart.data.datasets[0].data = timeSeriesData.fake_trend;
        chart.data.datasets[1].data = timeSeriesData.real_trend;
        chart.update();
    }
}

/**
 * Updates the distribution pie chart
 */
function updateDistributionChart(fakeCount, realCount) {
    const chart = Chart.getChart('distributionChart');
    if (chart) {
        chart.data.datasets[0].data = [fakeCount, realCount];
        chart.update();
    }
}

/**
 * Updates the model performance comparison chart
 */
function updateModelPerformanceChart(modelData) {
    const chart = Chart.getChart('performanceChart');
    if (chart) {
        const models = Object.keys(modelData);
        chart.data.labels = models.map(m => m.charAt(0).toUpperCase() + m.slice(1).replace('_', ' '));
        
        chart.data.datasets[0].data = models.map(model => modelData[model].accuracy);
        chart.data.datasets[1].data = models.map(model => modelData[model].precision);
        chart.data.datasets[2].data = models.map(model => modelData[model].recall);
        
        chart.update();
    }
}

/**
 * Sets up date range filter interactivity
 */
function setupDateRangeFilter() {
    $('#dateRangeSelector').change(function() {
        const range = $(this).val();
        console.log("Date range changed to:", range);
        // In a real app, you would refresh the data with the new date range
        // For demo, just show a message
        showFilterNotification("Date range updated to " + range);
    });
}

/**
 * Sets up model filter interactivity
 */
function setupModelFilters() {
    $('.model-filter').click(function() {
        $(this).toggleClass('active');
        // In a real app, you would refresh the data with the new filters
        // For demo, just show a message
        const activeFilters = $('.model-filter.active').map(function() {
            return $(this).data('model');
        }).get().join(', ');
        showFilterNotification("Model filters updated: " + activeFilters);
    });
}

/**
 * Shows a temporary notification that the data was refreshed
 */
function showRefreshNotification() {
    const notificationHtml = `
        <div class="toast align-items-center text-white bg-success border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-sync-alt me-2"></i> Dashboard data refreshed
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;
    
    const toastContainer = $('#toastContainer');
    if (toastContainer.length === 0) {
        $('body').append('<div id="toastContainer" class="toast-container position-fixed bottom-0 end-0 p-3"></div>');
    }
    
    $('#toastContainer').append(notificationHtml);
    const toast = new bootstrap.Toast($('.toast:last-child'));
    toast.show();
}

/**
 * Shows a temporary notification for filter changes
 */
function showFilterNotification(message) {
    const notificationHtml = `
        <div class="toast align-items-center text-white bg-info border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-filter me-2"></i> ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;
    
    const toastContainer = $('#toastContainer');
    if (toastContainer.length === 0) {
        $('body').append('<div id="toastContainer" class="toast-container position-fixed bottom-0 end-0 p-3"></div>');
    }
    
    $('#toastContainer').append(notificationHtml);
    const toast = new bootstrap.Toast($('.toast:last-child'));
    toast.show();
}

/**
 * Initializes Bootstrap tooltips and popovers
 */
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}
