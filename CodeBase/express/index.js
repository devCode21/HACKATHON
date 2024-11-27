const express=require('express')
const app=express()
const mongoose=require('mongoose')
const port=3000
const DailyExpense=require('./model')
const  cors=require('cors')
app.use(cors())

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



app.get('/user',async(req,res)=>{
    const user=await DailyExpense.find({})
    let label=[]
    for(i of user){
        label=[...label,i.weekday]
    }
    let sav=[]
    let totalSum=0
    for( i of user){
         let sum=0
        for(l of i.expenses){
            sum=sum+l.amount
            sav=[...sav,sum]
            totalSum=totalSum+sum
        }
    }
    console.log(label,sav)
    res.status(200).json({
        label,
        sav,
        totalSum
    })
})
app.listen(port,()=>{
    console.log('listning')
})