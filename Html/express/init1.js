const mongoose=require('mongoose')

const connectDb= async()=>{
    try{
        const connectInstance=await  mongoose.connect('mongodb://127.0.0.1:27017/NewWea')
        console.log('connected')
    }
    catch(error){
        console.log('error', error)
    }
}

connectDb()

const  DailyExpense=require('./model')

const sampleData=[
    {
      "day": "2024-05-01",
      "weekday": "Wednesday",
      "expenses": [
        { "category": "Food", "amount": 25.50 },
        { "category": "Transportation", "amount": 15.00 },
        { "category": "Entertainment", "amount": 10.00 }
      ]
    },
    {
      "day": "2024-05-02",
      "weekday": "Thursday",
      "expenses": [
        { "category": "Food", "amount": 30.00 },
        { "category": "Utilities", "amount": 50.00 },
        { "category": "Transportation", "amount": 20.00 }
      ]
    },
    {
      "day": "2024-05-03",
      "weekday": "Friday",
      "expenses": [
        { "category": "Food", "amount": 10.00 },
        { "category": "Entertainment", "amount": 20.00 }
      ]
    },
    {
      "day": "2024-05-04",
      "weekday": "Saturday",
      "expenses": [
        { "category": "Food", "amount": 40.00 },
        { "category": "Transportation", "amount": 15.00 },
        { "category": "Entertainment", "amount": 25.00 }
      ]
    },
    {
      "day": "2024-05-05",
      "weekday": "Sunday",
      "expenses": [
        { "category": "Food", "amount": 35.00 },
        { "category": "Utilities", "amount": 30.00 }
      ]
    },
    {
      "day": "2024-05-06",
      "weekday": "Monday",
      "expenses": [
        { "category": "Food", "amount": 20.00 },
        { "category": "Transportation", "amount": 12.00 }
      ]
    },
    {
      "day": "2024-05-07",
      "weekday": "Tuesday",
      "expenses": [
        { "category": "Food", "amount": 22.50 },
        { "category": "Entertainment", "amount": 18.00 },
        { "category": "Transportation", "amount": 15.00 }
      ]
    }
  ]


  const Data=async()=>{
     DailyExpense.deleteMany({})
     const res=await DailyExpense.insertMany(sampleData)
     console.log(res[0].expenses[0].category)
     

  }
  
  Data()

