import sqlite3

def migrate_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Step 1: Rename the old table
    cursor.execute("ALTER TABLE processing_status RENAME TO old_processing_status")

    # Step 2: Create the new table with the updated schema
    cursor.execute('''
        CREATE TABLE processing_status (
            model_id INTEGER,
            filename TEXT,
            nsfw_status TEXT,
            file_status TEXT,
            processing_status TEXT,
            new_filename TEXT,
            last_updated TEXT,
            PRIMARY KEY (model_id, filename)
        )
    ''')

    # Step 3: Migrate data from the old table to the new table
    cursor.execute('''
        INSERT INTO processing_status (model_id, filename, nsfw_status, file_status, processing_status, new_filename, last_updated)
        SELECT 
            model_id, 
            original_filename, 
            nsfw_status, 
            file_status, 
            processing_status, 
            new_filename, 
            last_updated
        FROM old_processing_status
    ''')

    # Step 4: Drop the old table
    cursor.execute("DROP TABLE old_processing_status")

    # Commit changes and close the connection
    conn.commit()
    conn.close()

    print("Database migration completed successfully.")

if __name__ == "__main__":
    migrate_database('lora_models.db')