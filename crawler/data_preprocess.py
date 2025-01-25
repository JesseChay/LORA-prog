import sqlite3
import os
import imghdr
from PIL import Image
import hashlib
import csv
import re
from datetime import datetime

class DataProcessor:
    def __init__(self, db_path, image_dir):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.image_dir = image_dir
        self.nsfw_threshold = 0.7  # Adjust this threshold as needed

    def create_processing_status_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_status (
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
        self.conn.commit()

    def get_image_path(self, model_id, filename):
        model_id_str = str(model_id).zfill(3)
        dir_path = os.path.join(self.image_dir, model_id_str[-3:], str(model_id))
        safe_filename = self.sanitize_filename(filename)
        return os.path.join(dir_path, safe_filename)

    def sanitize_filename(self, filename):
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        filename = filename.strip('. ')
        if not filename:
            filename = 'unnamed_file'
        max_length = 255
        if len(filename) > max_length:
            name, ext = os.path.splitext(filename)
            filename = name[:max_length-len(ext)] + ext
        return filename

    def process_data(self):
        self.create_processing_status_table()
        
        self.cursor.execute('''
            SELECT m.id, m.name, m.nsfw, i.filename, i.nsfw_level, i.width, i.height
            FROM models m
            JOIN images i ON m.id = i.model_id
            WHERE i.downloaded = TRUE
        ''')
        
        for row in self.cursor.fetchall():
            model_id, model_name, model_nsfw, filename, nsfw_level, width, height = row
            image_path = self.get_image_path(model_id, filename)
            
            nsfw_status = self.determine_nsfw_status(model_nsfw, nsfw_level)
            file_status = self.check_file_status(image_path)
            processing_status = self.process_image(image_path, nsfw_status, file_status)
            
            self.update_processing_status(model_id, filename, nsfw_status, file_status, processing_status)

    def determine_nsfw_status(self, model_nsfw, nsfw_level):
        if model_nsfw or (nsfw_level and nsfw_level.lower() != 'none'):
            return 'NSFW'
        return 'SFW'

    def check_file_status(self, image_path):
        if not os.path.exists(image_path):
            return 'Missing'
        
        file_type = imghdr.what(image_path)
        if file_type not in ['jpeg', 'png', 'gif']:
            return 'Invalid Format'
        
        try:
            with Image.open(image_path) as img:
                img.verify()
            return 'Valid'
        except:
            return 'Corrupted'

    def process_image(self, image_path, nsfw_status, file_status):
        if file_status != 'Valid':
            return 'Skipped'
        
        if nsfw_status == 'NSFW' and self.nsfw_threshold < 1:
            return 'Filtered'
        
        # Add more processing steps here if needed
        return 'Processed'

    def update_processing_status(self, model_id, filename, nsfw_status, file_status, processing_status):
        new_filename = filename if file_status == 'Valid' else None
        last_updated = datetime.now().isoformat()
        
        self.cursor.execute('''
            INSERT OR REPLACE INTO processing_status
            (model_id, filename, nsfw_status, file_status, processing_status, new_filename, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (model_id, filename, nsfw_status, file_status, processing_status, new_filename, last_updated))
        self.conn.commit()

    def export_results(self, output_file):
        self.cursor.execute('''
            SELECT 
                m.id as model_id, 
                m.name as model_name, 
                m.creator, 
                m.downloads, 
                m.likes,
                m.type as model_type,
                i.filename,
                i.name as image_name, 
                i.nsfw_level as original_nsfw_level,
                ps.nsfw_status, 
                ps.file_status, 
                ps.processing_status, 
                ps.new_filename,
                ps.last_updated
            FROM processing_status ps
            JOIN models m ON ps.model_id = m.id
            JOIN images i ON ps.model_id = i.model_id AND ps.filename = i.filename
        ''')
        
        results = self.cursor.fetchall()
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Model ID', 'Model Name', 'Creator', 'Downloads', 'Likes', 'Model Type',
                'Image Filename', 'Image Name', 'Original NSFW Level', 'NSFW Status',
                'File Status', 'Processing Status', 'New Filename', 'Last Updated', 'File Path'
            ])
            for row in results:
                model_id, model_name, creator, downloads, likes, model_type, \
                filename, image_name, original_nsfw_level, nsfw_status, \
                file_status, processing_status, new_filename, last_updated = row

                file_path = self.get_image_path(model_id, new_filename or filename)

                writer.writerow([
                    model_id, model_name, creator, downloads, likes, model_type,
                    filename, image_name, original_nsfw_level, nsfw_status,
                    file_status, processing_status, new_filename, last_updated, file_path
                ])
        
        print(f"Results exported to {output_file}")

    def process_and_export(self, output_file):
        self.process_data()
        self.export_results(output_file)

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    processor = DataProcessor('lora_models.db', 'images')
    processor.process_and_export('dataset_preparation_results.csv')
    processor.close()