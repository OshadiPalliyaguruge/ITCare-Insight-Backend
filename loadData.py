import pandas as pd
import mysql.connector

# Load the CSV file with the correct encoding
df = pd.read_csv('c:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\question_and_answers.csv', encoding='latin1')

# Fill NaN values with empty strings (optional)
df['Summary'] = df['Summary'].fillna('')
df['Resolution'] = df['Resolution'].fillna('')

# Define the maximum length for the Resolution column
max_length_resolution = 500  # Adjust this according to your database column size

# Connect to MySQL
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Osh@2000',
    database='helpdesk'
)

# Insert data into MySQL
cursor = conn.cursor()
for index, row in df.iterrows():
    summary = row['Summary'] if pd.notna(row['Summary']) else ''
    resolution = row['Resolution'] if pd.notna(row['Resolution']) else ''
    
    # Trim the resolution if it's too long
    if len(resolution) > max_length_resolution:
        resolution = resolution[:max_length_resolution]  # Trim to the first max_length_resolution characters
    
    cursor.execute("INSERT INTO csvData (Summary, Resolution) VALUES (%s, %s)", (summary, resolution))

# Commit changes and close the connection
conn.commit()
cursor.close()
conn.close()