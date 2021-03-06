import psycopg2
from settings import DATABASE_NAME, DATABASE_USER, DATABASE_PASS


def run_query(query, db_schema=DATABASE_NAME, db_username=DATABASE_USER, db_password=DATABASE_PASS):
    # Connect to an existing database
    conn = psycopg2.connect("dbname={} user={} password={}".format(db_schema, db_username, db_password))

    # Open a cursor to perform database operations
    cur = conn.cursor()

    # Execute a command: this creates a new table
    cur.execute(query)
    data = cur.fetchall()


    # Close communication with the database
    cur.close()
    conn.close()
    return data
