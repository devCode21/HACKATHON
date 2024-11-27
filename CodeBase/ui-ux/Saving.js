let goalData;

document.addEventListener('DOMContentLoaded', function() {
    // Initial empty chart setup
    const ctx = document.getElementById('expensesChart').getContext('2d');
    const expensesChart = new Chart(ctx, {
        type: 'bar',  // You can change this to 'pie', 'line', etc.
        data: {
            labels: ['Income', 'Goal Amount', 'Debts'], // Default categories
            datasets: [{
                label: 'Amount',
                data: [0, 0, 0], // Initial data points for Income, Goal Amount, Debts
                backgroundColor: ['#4CAF50', '#FFC107', '#F44336'],
                borderColor: ['#388E3C', '#FF9800', '#D32F2F'],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Event listener for form submission
    document.querySelector("form").addEventListener("submit", function(event) {
        event.preventDefault(); // Prevent the default form submission

        // Get the values from the form inputs
        const goalName = document.getElementById("goal-name").value;
        const goalAmount = document.getElementById("goal-amount").value;
        const goalCategory = document.getElementById("goal-category").value;
        const monthlyIncome = document.getElementById("income").value;
        const savingsRate = document.getElementById("savings-rate").value;
        const timePeriod = document.getElementById("time-period").value;
        const currentDebts = document.getElementById("debts").value;

        // Create an object with the form data
        goalData = {
            name: goalName,
            amount: goalAmount,
            category: goalCategory,
            income: monthlyIncome,
            savingsRate: savingsRate,
            timePeriod: timePeriod,
            debts: currentDebts
        };

        // Save the goal data to localStorage
        localStorage.setItem('savedGoal', JSON.stringify(goalData));

        // Update the chart data dynamically
        expensesChart.data.datasets[0].data = [monthlyIncome, goalAmount, currentDebts];
        expensesChart.data.labels = ['Income', 'Goal Amount', 'Current Debts'];
        expensesChart.update(); // Update the chart with new data

        // Provide feedback to the user (for testing purposes)
        alert(`Goal Saved! Here's the summary:\n
               Name: ${goalData.name}\n
               Amount: $${goalData.amount}\n
               Category: ${goalData.category}\n
               Monthly Income: $${goalData.income}\n
               Savings Rate: ${goalData.savingsRate}%\n
               Time Period: ${goalData.timePeriod} months\n
               Debts: $${goalData.debts}`);

       
        
        // Optionally reset the form
        document.querySelector("form").reset();
    });
});
