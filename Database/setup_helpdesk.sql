CREATE DATABASE IF NOT EXISTS helpdesk;
USE helpdesk;

CREATE TABLE IF NOT EXISTS incident_reports (
    Operational_Categorization_Tier_1 VARCHAR(255),
    Incident_ID VARCHAR(255),
    Submit_Date VARCHAR(255),
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
