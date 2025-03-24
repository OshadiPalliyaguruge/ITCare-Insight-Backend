// require('dotenv').config(); // Load environment variables
// const express = require('express');
// const cors = require('cors');
// const dashboardRoutes = require('./routes/dashboardRoutes'); // Import routes

// const app = express();
// const port = process.env.PORT || 5000;

// // Middleware
// app.use(cors({ origin: 'http://localhost:3001' }));
// app.use(express.json());

// // Use routes
// app.use('/api', dashboardRoutes);

// // Root endpoint
// app.get('/', (req, res) => {
//     res.send('Welcome to the backend server!');
// });

// // Start server
// app.listen(port, () => {
//     console.log(`Server running on http://localhost:${port}`);
// });


//real time 
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const dotenv = require('dotenv');
const dashboardRoutes = require('./routes/DashboardRoutes');

// Load environment variables from .env file
dotenv.config();

const app = express();
const server = http.createServer(app);
const io = socketIo(server); // Initialize socket.io

const port = process.env.PORT || 5000;

// Enable CORS for all routes
// app.use(
//   cors({
//     origin: process.env.FRONTEND_URL,
//     methods: ['GET', 'POST', 'PUT', 'DELETE'],
//     allowedHeaders: ['Content-Type', 'Authorization'],
//   })
// );
app.use(cors({
  origin: process.env.FRONTEND_URL, // Ensure it matches your React frontend URL
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true
}));

// Middleware to parse JSON requests
app.use(express.json());

// Use dashboard routes
app.use('/api', dashboardRoutes);

// Emit real-time data updates to all connected clients
io.on('connection', (socket) => {
  console.log('A user connected');
  
  // Example of emitting real-time updates for incident data (can be based on specific events)
  setInterval(() => {
    // Simulating the retrieval of data from the database
    socket.emit('incidentUpdate', { incidentCount: Math.floor(Math.random() * 100) });
  }, 5000); // Sending updates every 5 seconds

  socket.on('disconnect', () => {
    console.log('User disconnected');
  });
});

// Route for the root URL
app.get('/', (req, res) => {
  res.send('Welcome to the Express server!');
});

// Start the server
server.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
