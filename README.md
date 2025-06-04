# ITCare Insight Backend

This repository contains the backend services for **ITCare Insight**, an IT helpdesk incident management and analytics platform.

It includes the following components:

-  **Node.js/Express API** – RESTful API for frontend integration, including real-time updates via WebSocket.
-  **Flask ML Prediction Service** – Predicts the appropriate incident handling team based on ticket data.
-  **Flask Q&A Search Service** – Searches past incident resolutions to suggest answers for similar queries.
-  **CSV Import Scripts** – Python-based utility to load large preprocessed incident datasets into MySQL.
-  **SQL Schema Definition** – Scripts to initialize the MySQL database and create required tables.


## Table of Contents

- [Prerequisites](#prerequisites)  
- [Environment Variables](#environment-variables)  
- [Installation and Setup](#installation-and-setup)  
- [Database Setup](#database-setup)
- [Data Loading](#data-loading)  
- [Running Services](#running-services)  
- [API Endpoints](#api-endpoints)  
- [Logging and Monitoring](#logging-and-monitoring)
  

## Prerequisites
- Node.js (v14+) and npm
- Python (3.8+) and pip
- MySQL Server (8.0+)
- Git

## Environment Variables

#### Create .env and add the following

```
DB_HOST=your_db_host(ex: localhost)
DB_USER=your_db_root(ex: root)
DB_PASSWORD=your_db_password
DB_NAME=your_db_name
PORT=5000
FRONTEND_URL=your_frontend_url(ex: http://localhost:3000)
```

## Installation and Setup  

1. Clone the repo and enter the root directory:
   
   ```bash 
    git clone https://github.com/OshadiPalliyaguruge/ITCare-Insight-Backend.git && cd ITCare-Insight-Backend
   ```
   
2. Install dependencies:
   
   ```bash 
   # Install Node.js dependencies
   npm install express socket.io cors dotenv sentiment axios mysql
   
   # Install Python dependencies for both ML and Q&A services
   cd ..
   pip install flask joblib torch numpy pandas transformers nltk sentence-transformers mysql-connector-python flask-cors scikit-learn
   ```

## Database Setup

#### Load the preprocessed CSV  
Make sure your cleaned CSV is in MySQL’s upload directory (e.g. `c:\ProgramData\MySQL\MySQL Server 8.0\Uploads\`)  

---  

### **Setting Up MySQL Database**  

---  

### Option 1: Using SQL Command Line (MySQL Shell)  

1. **Open your SQL terminal or command prompt.**

2. **Login to MySQL as root (or your user) and run `setup_helpdesk.sql`:**

   ```bash
   mysql -u root -p < path\to\setup_helpdesk.sql
   ```

This will create a database `helpdesk` and a  table `incident_reports`.

---  

### Option 2: Using VS Code Extension for MySQL  

### 1. Install a MySQL Extension in VS Code ``MySQL``

- You can install them from the Extensions Marketplace in VS Code.

### 2. Configure the Connection

Once installed:

- Open the extension panel (usually a database icon on the sidebar).
- Click **Add New Connection** or **New Connection**.
- Fill in your MySQL connection details:
  - **Host:** e.g., `localhost`
  - **Port:** default is `3306`
  - **User:** e.g., `root` or your MySQL username
  - **Password:** your password
  - **Database:** leave blank if you want to create one later, or specify `helpdesk` if it exists.
- Save the connection.

### 3. Connect and Run Queries

- Connect to your database via the extension.
- Open a new SQL query window.
- Write SQL commands directly (e.g., `CREATE DATABASE helpdesk;`) and run them
  *or*
- Run scripts in `setup_helpdesk.sql` to create a database `helpdesk` and a  table `incident_reports`.
- The extension will show results inside VS Code.

---  

#### Verify the table:

   ``` sql
   SELECT COUNT(*) FROM incident_reports;
   ```

## Data Loading 

### Option 1:  
- Loads a CSV using Pandas and inserts rows into the `incident_reports` table using Python and `mysql.connector`.

#### Usage
1. Update credentials and CSV path inside the script `Database/automateCsvImport.py`
2. Run the script:

```bash
python Database/automateCsvImport.py
```

---  

### Option 2:  
- Load using SQL queries in `createTable.sql`  

---   

Add necessary SQL queries to `createTable.sql` to make modifications to the database, table and its content.  


## Running Services  

- Start each component in its terminal:  

**Express Server (with real-time updates)**  

``` bash
cd server
node index.js
```
- In the same directory, run the following two scripts as well.  

**Flask ML Prediction Service**  

``` bash
python predict_service.py
```
**Flask Q&A Search Service**  

``` bash
python QnA_search.py
```

## API Endpoints

### Dashboard Routes (Express API)

| Method | Path                                           | Description                                   |
|--------|------------------------------------------------|-----------------------------------------------|
| GET    | `/api/data`                                    | Aggregate incident counts by various fields   |
| GET    | `/api/slm-status`                              | Distribution of SLM real-time statuses        |
| GET    | `/api/incident-resolution-sentiment`           | Sentiment analysis of incident resolutions    |
| GET    | `/api/yearly-trend`                            | Annual incident counts                        |
| GET    | `/api/monthly-distribution`                    | Monthly incident counts                       |
| GET    | `/api/weekday-distribution`                    | Weekday incident counts                       |
| GET    | `/api/peak-hours`                              | Incident counts by hour                       |
| GET    | `/api/peak-days`                               | Incident counts by weekday name              |
| GET    | `/api/peak-months`                             | Incident counts by month                      |
| GET    | `/api/options`                                 | Distinct priorities, organizations, depts     |
| GET    | `/api/data-quality`                            | New/in-progress/resolved issue counts         |
| GET    | `/api/resolution-times`                        | Closed-status resolution times (hours)        |
| GET    | `/api/department-resolution-performance`       | Closed counts per department                  |
| GET    | `/api/department-closure-rate`                 | Closure rate (%) per department               |
| POST   | `/api/predict`                                 | Proxy to ML service, returns assigned group   |
| POST   | `/api/problems-solutions`                      | Find top solution for a given question        |  


## Logging and Monitoring   

- Express logs real-time connections and errors to console.
- Flask services write validation and runtime errors to log/app_errors.log.
- MySQL errors are returned as 500 responses by the router.


#### **With these components running, your frontend at FRONTEND_URL can consume all dashboard, prediction, and search capabilities in real time.**
