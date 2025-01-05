const Sentiment = require('sentiment');
const sentiment = new Sentiment();

// Sample data - Replace with your actual dataset
const incidents = [
  { incidentID: 1, summary: 'The system is working fine, great job!' },
  { incidentID: 2, summary: 'There was an issue with the database.' },
  { incidentID: 3, summary: 'The application is crashing unexpectedly.' },
  // ... Add more incidents here
];

// Initialize counters
let positiveCount = 0;
let neutralCount = 0;
let negativeCount = 0;

// Analyze sentiment for each incident
incidents.forEach((incident) => {
  const result = sentiment.analyze(incident.summary);
  
  // Categorize sentiment based on score
  if (result.score > 0) {
    positiveCount++;
  } else if (result.score < 0) {
    negativeCount++;
  } else {
    neutralCount++;
  }
});

// Print the sentiment counts
console.log('Positive:', positiveCount);
console.log('Negative:', negativeCount);
console.log('Neutral:', neutralCount);
