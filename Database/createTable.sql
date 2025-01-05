#Create database table
CREATE TABLE IF NOT EXISTS incident_reports(
    Operational_Categorization_Tier_1 VARCHAR(255),
    Incident_ID varchar(255),
    Submit_Date varchar(255),
    Summary TEXT,
    Resolution TEXT,
    Assigned_Group VARCHAR(255),
    Service VARCHAR(255),
    Submitter VARCHAR(50),
    Status VARCHAR(50),
    Priority VARCHAR(50),
    SLM_Real_Time_Status VARCHAR(255),
    Organization VARCHAR(255),
    Department VARCHAR(255)
);

#Load data to the table
#Ensure the CSV file is uploaded to a directory accessible by MySQL(c:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\)
LOAD DATA INFILE 'c:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\incident_report_preprocessed_final_98000_cleaned.csv'
INTO TABLE incident_reports
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

#Add new data to the table
#Step 1: 
#Use previous load data method with new datacsv file
#Both files should be in same format

#Step 2:
INSERT INTO incident_reports (
    Operational_Categorization_Tier_1,
    Incident_ID,
    Submit_Date,
    Summary,
    Resolution,
    Assigned_Group,
    Service,
    Submitter,
    Status,
    Priority,
    SLM_Real_Time_Status,
    Organization,
    Department
)
VALUES
(
    'Failure',
    'INC666666666',
    '01/11/2024 12:27:07 PM',
    'Cloud issue - Desktop resource not available',
    '',
    '',
    'Desktop Service',
    '20508',
    'Closed',
    'High',
    '',
    'ENGINEERING',
    'ENGINEERING MATERIALS & SUPPLY CHAIN MANAGEMENT'
);

#Delete table
DROP TABLE incident_reports;

DELETE FROM incident_reports WHERE Incident_ID = 'INC666666666';

SELECT * FROM incident_reports WHERE Incident_ID = 'INC666666666';

