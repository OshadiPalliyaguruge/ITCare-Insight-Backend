const express = require('express');
const router = express.Router();
const connection = require('../dbConnection');
const Sentiment = require('sentiment');
const sentiment = new Sentiment();
const axios = require('axios'); // Axios is used for HTTP requests


// Proxy predictions to Python Flask service
router.post('/predict', async (req, res) => {
  try {
    const response = await axios.post('http://127.0.0.1:5001/predict', req.body);
    res.json(response.data); // Return the prediction to the frontend
  } catch (error) {
    console.error('Error communicating with prediction service:', error.message);
    res.status(500).json({ error: 'Prediction service is unavailable.' });
  }
});

// Dashboard data endpoint
router.get('/data', (req, res) => {
    const query = `
        SELECT 
            Priority AS priority, 
            Status AS status, 
            SLM_Real_Time_Status AS slmStatus, 
            Department AS department, 
            Assigned_Group AS assignedGroup, 
            COUNT(*) AS incidentCount 
        FROM incident_reports
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
router.get('/slm-status', (req, res) => {
    const query = `
        SELECT SLM_Real_Time_Status AS slmStatus, COUNT(*) AS incidentCount 
        FROM incident_reports 
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

// Sentiment analysis of incident resolutions
router.get('/incident-resolution-sentiment', (req, res) => {
    const query = `
        SELECT Resolution AS resolution 
        FROM incident_reports
    `;
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching resolution:', error);
            return res.status(500).json({ error: 'Error fetching resolution' });
        }

        // Initialize sentiment counters
        let positiveCount = 0;
        let neutralCount = 0;
        let negativeCount = 0;

        // Analyze sentiment for each summary
        results.forEach((incident) => {
            const result = sentiment.analyze(incident.resolution);
            // console.log(result);  //show sentiment results

            // Categorize sentiment based on score
            if (result.score > 0) {
                positiveCount++;
            } else if (result.score < 0) {
                negativeCount++;
            } else {
                neutralCount++;
            }
        });

        // Send the sentiment counts as the response
        res.json({
            positive: positiveCount,
            negative: negativeCount,
            neutral: neutralCount,
        });
    });
});

// Yearly trends
router.get('/yearly-trend', (req, res) => {
    const query = `
        SELECT YEAR(STR_TO_DATE(Submit_Date, '%d/%m/%Y %h:%i:%s %p')) AS year, COUNT(*) AS incidentCount 
        FROM incident_reports 
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
router.get('/monthly-distribution', (req, res) => {
    const query = `
        SELECT MONTH(STR_TO_DATE(Submit_Date, '%d/%m/%Y %h:%i:%s %p')) AS month, COUNT(*) AS incidentCount 
        FROM incident_reports 
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
router.get('/weekday-distribution', (req, res) => {
    const query = `
        SELECT DAYOFWEEK(STR_TO_DATE(Submit_Date, '%d/%m/%Y %h:%i:%s %p')) AS weekday, COUNT(*) AS incidentCount 
        FROM incident_reports 
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
router.get('/submit-date-trends', (req, res) => {
    const query = `
        SELECT Submit_Date AS submitDate, COUNT(*) AS incidentCount 
        FROM incident_reports 
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
router.get('/status-data', (req, res) => {
    const query = `
        SELECT Status AS status, COUNT(*) AS incidentCount 
        FROM incident_reports 
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

// Problems and solutions endpoint
router.get('/problems-solutions', (req, res) => {
  const query = `
      SELECT Summary AS question, 
             MAX(Resolution) AS answer 
      FROM incident_reports 
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
router.get('/data-quality', (req, res) => {
    const query = `
        SELECT
            SUM(CASE WHEN Resolution = '' AND Assigned_Group = '' THEN 1 ELSE 0 END) AS newIssues,
            SUM(CASE WHEN Resolution = '' AND Assigned_Group <> '' THEN 1 ELSE 0 END) AS inProgressIssues,
            SUM(CASE WHEN Resolution <> '' AND Assigned_Group <> '' THEN 1 ELSE 0 END) AS resolvedIssues
        FROM incident_reports;

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
router.get('/resolution-times', (req, res) => {
    const query = `
        SELECT TIMESTAMPDIFF(HOUR, STR_TO_DATE(Submit_Date, '%d/%m/%Y %h:%i:%s %p'), NOW()) AS resolutionTime
        FROM incident_reports
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
router.get('/department-resolution-performance', (req, res) => {
    const query = `
        SELECT Department, COUNT(*) AS resolvedCount 
        FROM incident_reports 
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

router.get('/department-closure-rate', (req, res) => {
    const query = `
        SELECT Department, 
               COUNT(CASE WHEN Status = 'Closed' THEN 1 END) AS closedCount, 
               COUNT(*) AS totalCount, 
               (COUNT(CASE WHEN Status = 'Closed' THEN 1 END) / COUNT(*)) * 100 AS closureRate
        FROM incident_reports 
        GROUP BY Department;
    `;
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching department closure rate:', error);
            res.status(500).send('Server Error');
            return;
        }
        res.json(results);
    });
});


// Peak hours analysis
router.get('/peak-hours', (req, res) => {
    const query = `
        SELECT HOUR(STR_TO_DATE(Submit_Date, '%d/%m/%Y %h:%i:%s %p')) AS hour, 
               COUNT(*) AS incidentCount 
        FROM incident_reports 
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

// Peak days analysis
router.get('/peak-days', (req, res) => {
    const query = `
        SELECT 
            DAYNAME(STR_TO_DATE(Submit_Date, '%d/%m/%Y %h:%i:%s %p')) AS weekday, 
            COUNT(*) AS incidentCount 
        FROM incident_reports 
        GROUP BY weekday 
        ORDER BY FIELD(weekday, 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday');
    `;
    
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching peak days data:', error);
            res.status(500).send('Server Error');
            return;
        }

        // Prepare an object to hold the results for each weekday
        const weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
        const formattedResults = weekdays.map(day => {
            const found = results.find(result => result.weekday === day);
            return {
                weekday: day,
                incidentCount: found ? found.incidentCount : 0 // Default to 0 if not found
            };
        });

        res.json(formattedResults);
    });
});


router.get('/peak-months', (req, res) => {
    const query = `
        SELECT MONTH(STR_TO_DATE(Submit_Date, '%d/%m/%Y %h:%i:%s %p')) AS month,
               COUNT(*) AS incidentCount
        FROM incident_reports
        GROUP BY month
        ORDER BY month;
    `;
    
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching peak months data:', error);
            res.status(500).send('Server Error');
            return;
        }

        // Prepare an object to hold the results for each month
        const formattedResults = Array.from({ length: 12 }, (_, i) => {
            const monthNumber = i + 1; // Month numbers are 1-based (1 to 12)
            const found = results.find(result => result.month === monthNumber);
            return {
                month: monthNumber,
                incidentCount: found ? found.incidentCount : 0 // Default to 0 if not found
            };
        });

        res.json(formattedResults);
    });
});



module.exports = router;
 