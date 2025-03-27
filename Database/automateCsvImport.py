import pandas as pd
import mysql.connector

def load_csv_to_database(csv_file_path, db_config, table_name):
    try:
        # Read the CSV file
        data = pd.read_csv(csv_file_path)
        
        # Establish a connection to the database
        connection = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        cursor = connection.cursor()

        # Generate the INSERT query dynamically based on CSV columns
        columns = ", ".join(data.columns)
        placeholders = ", ".join(["%s"] * len(data.columns))
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        # Insert rows into the database
        for _, row in data.iterrows():
            cursor.execute(insert_query, tuple(row))

        # Commit the transaction
        connection.commit()

        print(f"Data from {csv_file_path} has been successfully loaded into {table_name}.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the database connection
        if connection.is_connected():
            cursor.close()
            connection.close()

# Configuration for your MySQL database
db_config = {
    'host': 'localhost',
    'user': 'your_username',
    'password': 'your_password',
    'database': 'your_database_name'
}

# Path to the CSV file to be imported
csv_file_path = 'path_to_your_csv_file.csv'

# Name of the table to insert data into
table_name = 'incident_reports'

# Load data into the database
load_csv_to_database(csv_file_path, db_config, table_name)
