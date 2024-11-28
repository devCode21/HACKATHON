const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const app = express();
const PORT = 3000;
const DailyExpense=require('./model')

const cors=require('cors')
app.use(cors())
// Middleware to parse URL-encoded data
app.use(express.urlencoded({ extended: true })); // Parses x-www-form-urlencoded data

// MongoDB connection setup


// MongoDB user schema and model
const userSchema = new mongoose.Schema({
    name: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true }
});

const User = mongoose.model('User', userSchema);

mongoose.connect('mongodb://localhost:27017/NewWea', {
    useNewUrlParser: true,
    useUnifiedTopology: true
})
    .then(() => console.log('Connected to  aMongoDB'))
    .catch((err) => console.error('Failed to connect to MongoDB:', err));

// Signup route
app.post('/signup', async (req, res) => {
    const { name, email, password } = req.body;

    try {
        // Check if the user already exists by email
        const userExists = await User.findOne({ email });
        if (userExists) {
            return res.status(400).json({ message: 'Email already exists', success: false });
        }

        // Create a new user
        const newUser = new User({
            name,
            email,
            password // Store the password as plain text (not recommended for production)
        });

        // Save the user to MongoDB
        await newUser.save();

        res.status(200).json({ message: 'Signup successful', success: true });
    } catch (err) {
        console.error(err);
        res.status(500).json({ message: 'Server error', success: false });
    }
});

// Login route
app.post('/login', async (req, res) => {
    const { email, password } = req.body;

    try {
        // Find the user by email
        const user = await User.findOne({ email });
        if (!user) {
            return res.status(400).json({ message: 'User not found', success: false });
        }

        // Compare the plain text password directly
        if (user.password !== password) {
            return res.status(400).json({ message: 'Invalid password', success: false });
        }

        res.status(200).json({ message: 'Login successful', success: true });
    } catch (err) {
        console.error(err);
        res.status(500).json({ message: 'Server error', success: false });
    }
});
app.get('/user',async(req,res)=>{
    const user=await DailyExpense.find({})
    console.log("user", user)
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
    console.log(label,sav,user,totalSum)
    res.status(200).json({
        user,
        label,
        sav,
        totalSum

    })
    
})
// Start the server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
