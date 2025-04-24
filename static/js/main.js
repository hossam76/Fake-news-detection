/**
 * Advanced JavaScript functionality for Fake News Detection System
 * Final Year Project - Enhanced UI
 */

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize theme preference
    initTheme();
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialize animations
    initAnimations();
    
    // Add loading state to form submissions
    initFormLoadingState();
    
    // Initialize charts if needed
    if (document.getElementById('modelComparisonChart')) {
        initModelComparisonChart();
    }
    
    if (document.getElementById('explanationChart')) {
        initExplanationChart();
    }
    
    if (document.getElementById('posChart')) {
        initPOSChart();
    }
    
    if (document.getElementById('wordFrequencyChart')) {
        initWordFrequencyChart();
    }
});

/**
 * Initialize theme based on saved preference or system setting
 */
function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
        if (document.querySelector('.theme-toggle i')) {
            document.querySelector('.theme-toggle i').classList.replace('fa-moon', 'fa-sun');
        }
    } else if (savedTheme === 'light') {
        document.body.classList.remove('dark-mode');
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        // Use system preference as fallback
        document.body.classList.add('dark-mode');
        if (document.querySelector('.theme-toggle i')) {
            document.querySelector('.theme-toggle i').classList.replace('fa-moon', 'fa-sun');
        }
    }
}

/**
 * Set up event listeners for interactive elements
 */
function setupEventListeners() {
    // Theme toggle functionality
    const themeToggle = document.querySelector('.theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            if (document.body.classList.contains('dark-mode')) {
                document.body.classList.remove('dark-mode');
                localStorage.setItem('theme', 'light');
                this.querySelector('i').classList.replace('fa-sun', 'fa-moon');
            } else {
                document.body.classList.add('dark-mode');
                localStorage.setItem('theme', 'dark');
                this.querySelector('i').classList.replace('fa-moon', 'fa-sun');
            }
        });
    }
    
    // Modern tabs functionality
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.modern-tab-pane').forEach(pane => pane.classList.remove('active'));
            
            // Add active class to current tab and pane
            this.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

/**
 * Initialize animations for UI elements
 */
function initAnimations() {
    // Animated counters
    const counters = document.querySelectorAll('.counter');
    
    if (counters.length > 0) {
        counters.forEach(counter => {
            const target = parseFloat(counter.getAttribute('data-target'));
            const duration = 2000; // ms
            const step = 60; // updates per second
            const increment = target / (duration / (1000 / step));
            
            let current = 0;
            const updateCounter = setInterval(() => {
                current += increment;
                
                if (current >= target) {
                    counter.textContent = target < 1 ? target.toFixed(1) : Math.round(target);
                    clearInterval(updateCounter);
                } else {
                    counter.textContent = current < 1 ? current.toFixed(1) : Math.round(current);
                }
            }, 1000 / step);
        });
    }
    
    // Enhance card hover effects
    const cards = document.querySelectorAll('.neo-card, .tech-card, .model-result-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 10px 30px rgba(0,0,0,0.15)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = '';
            this.style.boxShadow = '';
        });
    });
}

/**
 * Add loading state to form submissions
 */
function initFormLoadingState() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                const originalContent = submitBtn.innerHTML;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
                submitBtn.disabled = true;
                
                // Create a loading overlay
                const overlay = document.createElement('div');
                overlay.className = 'loading-overlay';
                overlay.innerHTML = `
                    <div class="spinner"></div>
                    <p class="mt-3">Analyzing content...</p>
                `;
                document.body.appendChild(overlay);
                
                // Restore button state in case of error
                setTimeout(() => {
                    if (document.querySelector('.loading-overlay')) {
                        submitBtn.innerHTML = originalContent;
                        submitBtn.disabled = false;
                        document.querySelector('.loading-overlay').remove();
                    }
                }, 30000); // 30 second timeout
            }
        });
    });
}

/**
 * Initialize model comparison chart
 */
function initModelComparisonChart() {
    const ctx = document.getElementById('modelComparisonChart').getContext('2d');
    
    // Get data from the results grid
    const models = [];
    const confidences = [];
    const colors = [];
    
    document.querySelectorAll('.model-result-card').forEach(card => {
        const model = card.querySelector('.model-result-header span').textContent;
        const confidence = parseFloat(card.querySelector('.confidence-value').textContent);
        const isPredictionReal = card.querySelector('.prediction').classList.contains('prediction-real');
        
        models.push(model);
        confidences.push(confidence);
        colors.push(isPredictionReal ? 'rgba(40, 167, 69, 0.7)' : 'rgba(220, 53, 69, 0.7)');
    });
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [{
                label: 'Confidence (%)',
                data: confidences,
                backgroundColor: colors,
                borderColor: colors.map(color => color.replace('0.7', '1')),
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Confidence: ${context.parsed.y.toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        display: true,
                        drawBorder: false,
                        color: 'rgba(200, 200, 200, 0.15)'
                    },
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif"
                        }
                    }
                },
                x: {
                    grid: {
                        display: false,
                        drawBorder: false
                    },
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif"
                        }
                    }
                }
            },
            animation: {
                duration: 2000,
                easing: 'easeOutQuart'
            }
        }
    });
}

/**
 * Initialize word frequency chart
 */
function initWordFrequencyChart() {
    const ctx = document.getElementById('wordFrequencyChart').getContext('2d');
    
    // Extract data from text analysis if available
    const textAnalysis = window.textAnalysis || {};
    const words = Object.keys(textAnalysis.top_words || {}).slice(0, 5);
    const frequencies = words.map(word => textAnalysis.top_words[word]);
    
    // Use placeholder data if no data is available
    const labels = words.length ? words : ['Sample', 'Example', 'Word', 'Test', 'Demo'];
    const data = frequencies.length ? frequencies : [24, 18, 16, 15, 12];
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Frequency',
                data: data,
                backgroundColor: 'rgba(67, 97, 238, 0.7)',
                borderColor: 'rgba(67, 97, 238, 1)',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: {
                        display: true,
                        drawBorder: false,
                        color: 'rgba(200, 200, 200, 0.15)'
                    }
                },
                y: {
                    grid: {
                        display: false,
                        drawBorder: false
                    }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeOutQuart'
            }
        }
    });
}

/**
 * Initialize POS chart
 */
function initPOSChart() {
    const ctx = document.getElementById('posChart').getContext('2d');
    
    // Extract data from text analysis if available
    const textAnalysis = window.textAnalysis || {};
    const posLabels = {
        'NOUN': 'Nouns',
        'VERB': 'Verbs',
        'ADJ': 'Adjectives',
        'ADV': 'Adverbs',
        'PRON': 'Pronouns',
        'DET': 'Determiners',
        'ADP': 'Adpositions',
        'CONJ': 'Conjunctions',
        'PRT': 'Particles'
    };
    
    const posCounts = textAnalysis.pos_counts || {};
    const labels = Object.keys(posCounts).map(pos => posLabels[pos] || pos);
    const counts = Object.values(posCounts);
    
    // Use placeholder data if no data is available
    const chartLabels = labels.length ? labels : ['Nouns', 'Verbs', 'Adjectives', 'Adverbs', 'Others'];
    const chartData = counts.length ? counts : [45, 28, 15, 12, 10];
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: chartLabels,
            datasets: [{
                data: chartData,
                backgroundColor: [
                    'rgba(67, 97, 238, 0.7)',
                    'rgba(77, 201, 240, 0.7)',
                    'rgba(114, 9, 183, 0.7)',
                    'rgba(6, 214, 160, 0.7)',
                    'rgba(255, 209, 102, 0.7)',
                    'rgba(239, 71, 111, 0.7)',
                    'rgba(17, 138, 178, 0.7)',
                    'rgba(7, 59, 76, 0.7)',
                    'rgba(6, 123, 194, 0.7)'
                ],
                borderColor: [
                    'rgba(67, 97, 238, 1)',
                    'rgba(77, 201, 240, 1)',
                    'rgba(114, 9, 183, 1)',
                    'rgba(6, 214, 160, 1)',
                    'rgba(255, 209, 102, 1)',
                    'rgba(239, 71, 111, 1)',
                    'rgba(17, 138, 178, 1)',
                    'rgba(7, 59, 76, 1)',
                    'rgba(6, 123, 194, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 11
                        },
                        boxWidth: 15
                    }
                }
            },
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 2000,
                easing: 'easeOutQuart'
            },
            cutout: '65%'
        }
    });
}

/**
 * Initialize explanation chart for AI explainability
 */
function initExplanationChart() {
    const ctx = document.getElementById('explanationChart').getContext('2d');
    
    // Extract data from explanation if available
    const explanation = window.explanation || {};
    const features = explanation.explanation || [];
    
    const words = features.map(item => item.word);
    const weights = features.map(item => item.weight);
    
    // Use placeholder data if no data is available
    const labels = words.length ? words : ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'];
    const data = weights.length ? weights : [0.4, 0.3, -0.2, -0.5, 0.2];
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Feature Impact',
                data: data,
                backgroundColor: data.map(value => value >= 0 ? 'rgba(40, 167, 69, 0.7)' : 'rgba(220, 53, 69, 0.7)'),
                borderColor: data.map(value => value >= 0 ? 'rgba(40, 167, 69, 1)' : 'rgba(220, 53, 69, 1)'),
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            return `Impact: ${value.toFixed(4)} (${value >= 0 ? 'Real' : 'Fake'})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: {
                        display: true,
                        drawBorder: false,
                        color: 'rgba(200, 200, 200, 0.15)'
                    }
                },
                y: {
                    grid: {
                        display: false,
                        drawBorder: false
                    }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeOutQuart'
            }
        }
    });
}
