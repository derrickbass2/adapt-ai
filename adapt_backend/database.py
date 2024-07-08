import sqlite3

# Function to connect to the database
def connect_to_db():
    conn = sqlite3.connect('your_database.db')
    return conn

# Function to execute a SQL query
def execute_query(conn, query, params=None):
    cursor = conn.cursor()
    cursor.execute(query, params)
    return cursor

# Function to fetch query results
def fetch_results(cursor):
    return cursor.fetchall()