import sqlite3
import os

# Path to the SQLite database
DATABASE_DIR = "database"
DATABASE_PATH = os.path.join(DATABASE_DIR, "attendance.db")

# Function to initialize the database
def initialize_database():
    """
    Ensures the database and required tables are created.
    """
    # Create the database directory if it doesn't exist
    os.makedirs(DATABASE_DIR, exist_ok=True)

    # Connect to the SQLite database (creates file if it doesn't exist)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Create the 'students' table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            prnno TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"Database initialized at {DATABASE_PATH}")

# Function to insert or update student records
def insert_or_update_student(prnno, name, embedding_blob):
    """
    Inserts a new student or updates an existing one in the database.
    :param prnno: Unique student identifier.
    :param name: Name of the student.
    :param embedding_blob: Face embedding as a binary BLOB.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Insert or update the record
    cursor.execute("""
        INSERT INTO students (prnno, name, embedding)
        VALUES (?, ?, ?)
        ON CONFLICT(prnno) DO UPDATE SET name=excluded.name, embedding=excluded.embedding
    """, (prnno, name, embedding_blob))

    conn.commit()
    conn.close()
    print(f"Student {name} (PRN: {prnno}) successfully added/updated.")
