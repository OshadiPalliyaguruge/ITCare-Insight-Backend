# ITCare Insight Backend

This repository contains the backend services for ITCare Insight, an IT helpdesk incident management & analytics platform.  
It includes a `Node.js/Express API with real-time WebSocket updates, a Flask-based ML prediction service, a Q&A search service, data loading scripts, and SQL definitions to set up the MySQL database.`

## Table of Contents
I. Prerequisites  
II. Project Structure  
III. Environment Variables  
IV. Installation & Setup  
V. Database Setup  
VI. Data Loading  
VII. Running Services  
VIII. API Endpoints  
IX. Logging & Monitoring  


## Prerequisites
- Node.js (v14+) and npm
- Python (3.8+) and pip
- MySQL Server (8.0+)
- Git

## Installation & Setup
1. Clone the repo and enter the root directory:
   
   ```
   git clone <repo_url> && cd <repo_name>
2. Install Node.js dependencies:
   
    ```
    cd server
    npm install express socket.io cors dotenv sentiment axios mysql
    ```

3. Install Python dependencies in a virtualenv:

   ```
   cd ../ml_predict
   python3 -m venv venv && source venv/bin/activate
   pip install flask joblib torch numpy pandas transformers nltk sentence-transformers mysql-connector-python

4. Install Q&A search service dependencies:

   ```
   cd ../qna_search
   python3 -m venv venv && source venv/bin/activate
   pip install flask flask-cors nltk sentence-transformers scikit-learn
   ```
### Database Setup


### Create .env file

#### Create .env and add following

```
DB_HOST=your_db_host(ex: localhost)
DB_USER=your_db_root(ex: root)
DB_PASSWORD=your_db_password
DB_NAME=your_db_name
PORT=5000
FRONTEND_URL=your_frontend_url(ex: http://localhost:3000)
```

# Run the backend

Run node index.js
