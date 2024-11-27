const mongoose = require('mongoose');
const Schema = mongoose.Schema;

// Define the schema for the expense categories
const expenseSchema = new Schema({
  category: { type: String, required: true },
  amount: { type: Number, required: true }
});

// Define the schema for the daily expenses document
const dailyExpenseSchema = new Schema({
  day: { type: String, required: true },            // Date of the expense
  weekday: { type: String, required: true },        // Day of the week (e.g., "Monday")
  expenses: [{
    category: { type: String, required: true },
    amount: { type: Number, required: true }
  }]                       // Array of expense categories
});

// Create a model based on the schema
const DailyExpense = mongoose.model('DailyExpense', dailyExpenseSchema);

// Export the model for use in other parts of the application
module.exports = DailyExpense;