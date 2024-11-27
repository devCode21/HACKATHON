document.addEventListener("DOMContentLoaded", () => {
    const toggleBtn = document.getElementById("toggle-btn");
    const chartCanvas = document.getElementById("chart").getContext("2d");
    const Savings=document.getElementById('Savings')
    const totalexp=document.getElementById('totalexp')
   
    
    
    let chartInstance = null;
    let isChartEnabled = true;
  
    // Actual data fetch function
    async function fetchChartData() {
      try {
        const response = await fetch("http://localhost:3000/user"); // Correct URL without quotes
        const data = await response.json();
        console.log(data);
         // Log fetched data to see the structure
         Savings.innerText=(data.totalSum)*0.5;
         totalexp.innerText=data.totalSum
         
        return data; // Assuming the API returns { labels, data }
      } catch (error) {
        console.error("Error fetching data:", error);
        // Returning mock data in case of an error
        return new Promise((resolve) => {
          setTimeout(() => {
            resolve({
              labels: data.labels,
              data:( data.sav)*0.5 // Mock savings data
            });
          }, 1000);
        });
      }
    }
  
    // Render the chart
    async function renderChart() {
      const { label,sav} = await fetchChartData()
  
      if (chartInstance) {
        chartInstance.destroy(); // Clear the existing chart
      }
  
      chartInstance = new Chart(chartCanvas, {
        type: "line", // You can change to 'bar' or other chart types
        data: {
          labels: label,
          datasets: [
            {
              label: "Savings",
              data: sav,
              borderColor: "#007bff",
              backgroundColor: "rgba(0, 123, 255, 0.2)",
              borderWidth: 2,
              fill: true,
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            legend: {
              display: true,
            },
          },
        },
      });
    }
  
    // Toggle button logic
    toggleBtn.addEventListener("click", () => {
      isChartEnabled = !isChartEnabled;
      toggleBtn.textContent = isChartEnabled ? "ON" : "OFF";
      toggleBtn.classList.toggle("off");
  
      if (isChartEnabled) {
        renderChart();
      } else {
        if (chartInstance) {
          chartInstance.destroy(); // Destroy the chart
          chartInstance = null;
        }
      }
    });
  
    // Initial chart render
    renderChart();
  });
  