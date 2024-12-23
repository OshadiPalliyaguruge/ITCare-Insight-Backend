const express = require('express'); 
const mysql = require('mysql2');
const cors = require('cors'); // Import the CORS package

const app = express();
const port = 5000;

// Enable CORS for all routes with specific options
app.use(cors({
    origin: 'http://localhost:3000', // Allow only React app's origin
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));

// Handle preflight (OPTIONS) requests
app.options('*', cors());

// MySQL connection configuration
const connection = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'Osh@2000',
    database: 'helpdesk'
});

// Connect to MySQL database
connection.connect((err) => {
    if (err) {
        console.error('Error connecting to MySQL:', err.stack);
        return;
    }
    console.log('Connected to MySQL');
});

// Route for the root URL
app.get('/', (req, res) => {
    res.send('Welcome to the Express server!!!!');
});

// Endpoint to get grouped data for the dashboard
app.get('/api/data', (req, res) => {
    const query = `
        SELECT 
            column10 AS priority, 
            column9 AS status, 
            column11 AS slmStatus, 
            column13 AS department, 
            column7 AS assignedGroup, 
            COUNT(*) AS incidentCount 
        FROM csvData 
        GROUP BY priority, status, slmStatus, department, assignedGroup
    `;
    
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching data:', error);
            res.status(500).send('Server Error');
            return;
        }
        res.json(results);
    });
});

// Additional routes in Express backend
app.get('/api/slm-status', (req, res) => {
    const query = `SELECT column11 AS slmStatus, COUNT(*) AS incidentCount FROM csvData GROUP BY slmStatus`;
    
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching SLM status data:', error);
            res.status(500).send('Server Error');
            return;
        }
        res.json(results);
    });
});

app.get('/api/submit-date-trends', (req, res) => {
    const query = `SELECT column3 AS submitDate, COUNT(*) AS incidentCount FROM csvData GROUP BY submitDate ORDER BY submitDate`;
    
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching submit date data:', error);
            res.status(500).send('Server Error');
            return;
        }
        res.json(results);
    });
});

// Endpoint to get status data
app.get('/api/status-data', (req, res) => {
    const query = `SELECT column9 AS status, COUNT(*) AS incidentCount FROM csvData GROUP BY status`;
    
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching status data:', error);
            res.status(500).send('Server Error');
            return;
        }
        res.json(results);
    });
});


// Start the server
app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});

