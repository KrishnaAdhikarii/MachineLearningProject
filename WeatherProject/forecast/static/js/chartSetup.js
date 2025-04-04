document.addEventListener('DOMContentLoaded', () => {
    const chartElement = document.getElementById('chart');  // Ensure canvas exists
    if (!chartElement) {
        console.error('Canvas Element not found.');
        return;
    }

    const ctx = chartElement.getContext('2d');
    const gradient = ctx.createLinearGradient(0, -10, 0, 100);
    gradient.addColorStop(0, 'rgb(255, 0, 0)');
    gradient.addColorStop(1, 'rgba(136, 255, 0, 1)');

    const forecastItems = document.querySelectorAll('.forecast-item');
    if (!forecastItems || forecastItems.length === 0) {
        console.error('No forecast items found.');
        return;
    }

    const temps = [];
    const times = [];

    forecastItems.forEach(item => {
        const time = item.querySelector('.forecast-time')?.textContent;  // Check for existence before using
        const temp = item.querySelector('.forecast-temperatureValue')?.textContent;
        const hum = item.querySelector('.forecast-humidityValue')?.textContent;

        // Ensure temp is extracted correctly and is a valid number
        if (time && temp) {
            // Remove any non-numeric characters from the temperature value (like °C or °F)
            const tempValue = parseFloat(temp.replace(/[^0-9.-]/g, '')); 

            if (!isNaN(tempValue)) {
                times.push(time);
                temps.push(tempValue);
            } else {
                console.error(`Invalid temperature value: ${temp}`);
            }
        }
    });

    if (temps.length === 0 || times.length === 0) {
        console.error('Temp or time values are missing.');
        return;
    }

    // Initialize chart with valid data
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
            datasets: [
                {
                    label: 'Temperature (°C)',
                    data: temps,
                    borderColor: gradient,
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 2,
                }
            ]
        },
        options: {
            plugins: {
                legend: {
                    display: false,
                },
            },
            scales: {
                x: {
                    grid: {
                        drawOnChartArea: false,
                    },
                },
                y: {
                    grid: {
                        drawOnChartArea: false,
                    },
                },
            },
            animation: {
                duration: 750,
            },
        },
    });
});
