// // const express = require('express');
// // const mysql = require('mysql2');
// // const cors = require('cors'); // Import the CORS package

// // const app = express();
// // const port = 5000;

// // // Enable CORS for all routes with specific options
// // app.use(cors({
// //     origin: 'http://localhost:3000', // Allow only React app's origin
// //     methods: ['GET', 'POST', 'PUT', 'DELETE'],
// //     allowedHeaders: ['Content-Type', 'Authorization']
// // }));

// // // Handle preflight (OPTIONS) requests
// // app.options('*', cors());

// // // MySQL connection configuration
// // const connection = mysql.createConnection({
// //     host: '',
// //     user: '',
// //     password: '',
// //     database: ''
// // });

// // // Connect to MySQL database
// // connection.connect((err) => {
// //     if (err) {
// //         console.error('Error connecting to MySQL:', err.stack);
// //         return;
// //     }
// //     console.log('Connected to MySQL');
// // });

// // // Route for the root URL
// // app.get('/', (req, res) => {
// //     res.send('Welcome to the Express server!!!!');
// // });

// // // Endpoint to get grouped data for the dashboard
// // app.get('/api/data', (req, res) => {
// //     const query = `
// //         SELECT 
// //             Priority AS priority, 
// //             Status AS status, 
// //             SLM_Real_Time_Status AS slmStatus, 
// //             Department AS department, 
// //             Assigned_Group AS assignedGroup, 
// //             COUNT(*) AS incidentCount 
// //         FROM incident_analysis
// //         GROUP BY priority, status, slmStatus, department, assignedGroup
// //     `;
    
// //     connection.query(query, (error, results) => {
// //         if (error) {
// //             console.error('Error fetching data:', error);
// //             res.status(500).send('Server Error');
// //             return;
// //         }
// //         res.json(results);
// //     });
// // });

// // // Additional routes in Express backend
// // app.get('/api/slm-status', (req, res) => {
// //     const query = `
// //         SELECT SLM_Real_Time_Status AS slmStatus, COUNT(*) AS incidentCount 
// //         FROM incident_data 
// //         GROUP BY slmStatus
// //     `;
    
// //     connection.query(query, (error, results) => {
// //         if (error) {
// //             console.error('Error fetching SLM status data:', error);
// //             res.status(500).send('Server Error');
// //             return;
// //         }
// //         res.json(results);
// //     });
// // });

// // app.get('/api/submit-date-trends', (req, res) => {
// //     const query = `
// //         SELECT Submit_Date AS submitDate, COUNT(*) AS incidentCount 
// //         FROM incident_data 
// //         GROUP BY submitDate 
// //         ORDER BY submitDate
// //     `;
    
// //     connection.query(query, (error, results) => {
// //         if (error) {
// //             console.error('Error fetching submit date data:', error);
// //             res.status(500).send('Server Error');
// //             return;
// //         }
// //         res.json(results);
// //     });
// // });

// // // Endpoint to get status data
// // app.get('/api/status-data', (req, res) => {
// //     const query = `
// //         SELECT Status AS status, COUNT(*) AS incidentCount 
// //         FROM incident_data 
// //         GROUP BY status
// //     `;
    
// //     connection.query(query, (error, results) => {
// //         if (error) {
// //             console.error('Error fetching status data:', error);
// //             res.status(500).send('Server Error');
// //             return;
// //         }
// //         res.json(results);
// //     });
// // });

// // app.get('/api/problems-solutions', (req, res) => {
// //     const query = `
// //         SELECT Summary AS question, 
// //                MAX(Resolution) AS answer 
// //         FROM incident_data 
// //         GROUP BY Summary 
// //         ORDER BY COUNT(*) DESC 
// //         LIMIT 10
// //     `;
    
// //     connection.query(query, (error, results) => {
// //         if (error) {
// //             console.error('Error fetching problems and solutions:', error);
// //             res.status(500).send('Server Error');
// //             return;
// //         }
// //         res.json(results);
// //     });
// // });


// // // Start the server
// // app.listen(port, () => {
// //     console.log(`Server running on port ${port}`);
// // });





// const express = require('express');
// const mysql = require('mysql2');
// const cors = require('cors');

// const app = express();
// const port = 5000;

// // Enable CORS for all routes with specific options
// app.use(cors({
//     origin: 'http://localhost:3000' , // Allow only React app's origin
//     methods: ['GET', 'POST', 'PUT', 'DELETE'],
//     allowedHeaders: ['Content-Type', 'Authorization']
// }));

// // MySQL connection configuration
// const connection = mysql.createConnection({
//     host: 'localhost',
//     user: 'root',
//     password: 'Osh@2000',
//     database: 'helpdesk'
// });

// // Connect to MySQL database
// connection.connect((err) => {
//     if (err) {
//         console.error('Error connecting to MySQL:', err.stack);
//         return;
//     }
//     console.log('Connected to MySQL');
// });

// // Route for the root URL
// app.get('/', (req, res) => {
//     res.send('Welcome to the Express server!!!!');
// });

// // Endpoint to get grouped data for the dashboard
// app.get('/api/data', (req, res) => {
//     const query = `
//         SELECT 
//             Priority AS priority, 
//             Status AS status, 
//             SLM_Real_Time_Status AS slmStatus, 
//             Department AS department, 
//             Assigned_Group AS assignedGroup, 
//             COUNT(*) AS incidentCount 
//         FROM incident_analysis
//         GROUP BY priority, status, slmStatus, department, assignedGroup
//     `;
    
//     connection.query(query, (error, results) => {
//         if (error) {
//             console.error('Error fetching data:', error);
//             res.status(500).send('Server Error');
//             return;
//         }
//         res.json(results);
//     });
// });

// // Endpoint for SLM Status distribution
// app.get('/api/slm-status', (req, res) => {
//     const query = `
//         SELECT SLM_Real_Time_Status AS slmStatus, COUNT(*) AS incidentCount 
//         FROM incident_data 
//         GROUP BY slmStatus
//     `;
    
//     connection.query(query, (error, results) => {
//         if (error) {
//             console.error('Error fetching SLM status data:', error);
//             res.status(500).send('Server Error');
//             return;
//         }
//         res.json(results);
//     });
// });

// // Endpoint for yearly trends
// app.get('/api/yearly-trend', (req, res) => {
//     const query = `
//         SELECT YEAR(Submit_Date) AS year, COUNT(*) AS incidentCount 
//         FROM incident_data 
//         GROUP BY year 
//         ORDER BY year
//     `;
    
//     connection.query(query, (error, results) => {
//         if (error) {
//             console.error('Error fetching yearly trend data:', error);
//             res.status(500).send('Server Error');
//             return;
//         }
//         res.json(results);
//     });
// });

// // Endpoint for monthly distribution
// app.get('/api/monthly-distribution', (req, res) => {
//     const query = `
//         SELECT MONTH(Submit_Date) AS month, COUNT(*) AS incidentCount 
//         FROM incident_data 
//         GROUP BY month 
//         ORDER BY month
//     `;
    
//     connection.query(query, (error, results) => {
//         if (error) {
//             console.error('Error fetching monthly distribution data:', error);
//             res.status(500).send('Server Error');
//             return;
//         }
//         res.json(results);
//     });
// });

// // Endpoint for weekday distribution
// app.get('/api/weekday-distribution', (req, res) => {
//     const query = `
//         SELECT DAYOFWEEK(Submit_Date) AS weekday, COUNT(*) AS incidentCount 
//         FROM incident_data 
//         GROUP BY weekday 
//         ORDER BY weekday
//     `;
    
//     connection.query(query, (error, results) => {
//         if (error) {
//             console.error('Error fetching weekday distribution data:', error);
//             res.status(500).send('Server Error');
//             return;
//         }
//         res.json(results);
//     });
// });

// // Endpoint for submit date trends
// app.get('/api/submit-date-trends', (req, res) => {
//     const query = `
//         SELECT Submit_Date AS submitDate, COUNT(*) AS incidentCount 
//         FROM incident_data 
//         GROUP BY submitDate 
//         ORDER BY submitDate
//     `;
    
//     connection.query(query, (error, results) => {
//         if (error) {
//             console.error('Error fetching submit date data:', error);
//             res.status(500).send('Server Error');
//             return;
//         }
//         res.json(results);
//     });
// });

// // Endpoint to get status data
// app.get('/api/status-data', (req, res) => {
//     const query = `
//         SELECT Status AS status, COUNT(*) AS incidentCount 
//         FROM incident_data 
//         GROUP BY status
//     `;
    
//     connection.query(query, (error, results) => {
//         if (error) {
//             console.error('Error fetching status data:', error);
//             res.status(500).send('Server Error');
//             return;
//         }
//         res.json(results);
//     });
// });

// // Endpoint to get problems and solutions
// app.get('/api/problems-solutions', (req, res) => {
//     const query = `
//         SELECT Summary AS question, 
//                MAX(Resolution) AS answer 
//         FROM incident_data 
//         GROUP BY Summary 
//         ORDER BY COUNT(*) DESC 
//         LIMIT 10
//     `;
    
//     connection.query(query, (error, results) => {
//         if (error) {
//             console.error('Error fetching problems and solutions:', error);
//             res.status(500).send('Server Error');
//             return;
//         }
//         res.json(results);
//     });
// });

// // Close MySQL connection when shutting down the server
// process.on('SIGINT', () => {
//     connection.end((err) => {
//         if (err) {
//             console.error('Error closing MySQL connection:', err.stack);
//         }
//         console.log('MySQL connection closed');
//         process.exit();
//     });
// });

// // Start the server
// app.listen(port, () => {
//     console.log(`Server running on port ${port}`);
// });


const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');

const app = express();
const port = 5000;

// Enable CORS for all routes
app.use(cors({
    origin: 'http://localhost:3000', // React app's origin
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));

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
    res.send('Welcome to the Express server!');
});

// Dashboard data endpoint
app.get('/api/data', (req, res) => {
    const query = `
        SELECT 
            Priority AS priority, 
            Status AS status, 
            SLM_Real_Time_Status AS slmStatus, 
            Department AS department, 
            Assigned_Group AS assignedGroup, 
            COUNT(*) AS incidentCount 
        FROM incident_data
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

// SLM Status distribution
app.get('/api/slm-status', (req, res) => {
    const query = `
        SELECT SLM_Real_Time_Status AS slmStatus, COUNT(*) AS incidentCount 
        FROM incident_data 
        GROUP BY slmStatus
    `;
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching SLM status data:', error);
            res.status(500).send('Server Error');
            return;
        }
        res.json(results);
    });
});

// Yearly trends
app.get('/api/yearly-trend', (req, res) => {
    const query = `
        SELECT YEAR(STR_TO_DATE(Submit_Date, '%d/%m/%Y %h:%i:%s %p')) AS year, COUNT(*) AS incidentCount 
        FROM incident_data 
        GROUP BY year 
        ORDER BY year
    `;
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching yearly trend data:', error);
            res.status(500).send('Server Error');
            return;
        }
        res.json(results);
    });
});

// Monthly distribution
app.get('/api/monthly-distribution', (req, res) => {
    const query = `
        SELECT MONTH(STR_TO_DATE(Submit_Date, '%d/%m/%Y %h:%i:%s %p')) AS month, COUNT(*) AS incidentCount 
        FROM incident_data 
        GROUP BY month 
        ORDER BY month
    `;
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching monthly distribution data:', error);
            res.status(500).send('Server Error');
            return;
        }
        res.json(results);
    });
});

// Weekday distribution
app.get('/api/weekday-distribution', (req, res) => {
    const query = `
        SELECT DAYOFWEEK(STR_TO_DATE(Submit_Date, '%d/%m/%Y %h:%i:%s %p')) AS weekday, COUNT(*) AS incidentCount 
        FROM incident_data 
        GROUP BY weekday 
        ORDER BY weekday
    `;
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching weekday distribution data:', error);
            res.status(500).send('Server Error');
            return;
        }
        res.json(results);
    });
});

// Submit date trends
app.get('/api/submit-date-trends', (req, res) => {
    const query = `
        SELECT Submit_Date AS submitDate, COUNT(*) AS incidentCount 
        FROM incident_data 
        GROUP BY submitDate 
        ORDER BY submitDate
    `;
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching submit date data:', error);
            res.status(500).send('Server Error');
            return;
        }
        res.json(results);
    });
});

// Status data
app.get('/api/status-data', (req, res) => {
    const query = `
        SELECT Status AS status, COUNT(*) AS incidentCount 
        FROM incident_data 
        GROUP BY status
    `;
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching status data:', error);
            res.status(500).send('Server Error');
            return;
        }
        res.json(results);
    });
});

// Problems and solutions
app.get('/api/problems-solutions', (req, res) => {
    const query = `
        SELECT Summary AS question, 
               MAX(Resolution) AS answer 
        FROM incident_data 
        GROUP BY Summary 
        ORDER BY COUNT(*) DESC 
        LIMIT 10
    `;
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching problems and solutions:', error);
            res.status(500).send('Server Error');
            return;
        }
        res.json(results);
    });
});

// Data quality insights
app.get('/api/data-quality', (req, res) => {
    const query = `
        SELECT
            SUM(CASE WHEN Service IS NULL THEN 1 ELSE 0 END) AS missingService,
            SUM(CASE WHEN Resolution = '' THEN 1 ELSE 0 END) AS missingResolution,
            COUNT(*) AS totalRecords
        FROM incident_data;
    `;
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching data quality insights:', error);
            res.status(500).send('Server Error');
            return;
        }
        res.json(results[0]);
    });
});

// Resolution times
app.get('/api/resolution-times', (req, res) => {
    const query = `
        SELECT TIMESTAMPDIFF(HOUR, STR_TO_DATE(Submit_Date, '%d/%m/%Y %h:%i:%s %p'), NOW()) AS resolutionTime
        FROM incident_data
        WHERE Status = 'Closed';
    `;
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching resolution times:', error);
            res.status(500).send('Server Error');
            return;
        }
        res.json(results);
    });
});

// Department resolution performance
app.get('/api/department-resolution-performance', (req, res) => {
    const query = `
        SELECT Department, 
               AVG(TIMESTAMPDIFF(HOUR, STR_TO_DATE(Submit_Date, '%d/%m/%Y %h:%i:%s %p'), 
               STR_TO_DATE(Resolve_Date, '%d/%m/%Y %h:%i:%s %p'))) AS avgResolutionTime 
        FROM incident_data 
        WHERE Status = 'Closed' 
        GROUP BY Department;
    `;
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching department resolution performance:', error);
            res.status(500).send('Server Error');
            return;
        }
        res.json(results);
    });
});

// Peak hours analysis
app.get('/api/peak-hours', (req, res) => {
    const query = `
        SELECT HOUR(STR_TO_DATE(Submit_Date, '%d/%m/%Y %h:%i:%s %p')) AS hour, 
               COUNT(*) AS incidentCount 
        FROM incident_data 
        GROUP BY hour 
        ORDER BY hour;
    `;
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching peak hours data:', error);
            res.status(500).send('Server Error');
            return;
        }
        res.json(results);
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
