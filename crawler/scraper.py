import sqlite3
import requests
import json
from datetime import datetime
import os
import time
import re

# Database operations
class Database:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY,
                name TEXT,
                creator TEXT,
                downloads INTEGER,
                likes INTEGER,
                created_at TEXT,
                last_version_at TEXT,
                type TEXT,
                nsfw BOOLEAN,
                tags TEXT,
                images_downloaded BOOLEAN DEFAULT FALSE
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                filename TEXT,
                name TEXT,
                nsfw_level TEXT,
                width INTEGER,
                height INTEGER,
                downloaded BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (model_id) REFERENCES models (id)
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS scrape_status (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_cursor TEXT
            )
        ''')
        self.conn.commit()

    def insert_model(self, model_data):
        self.cursor.execute('''
            INSERT OR REPLACE INTO models 
            (id, name, creator, downloads, likes, created_at, last_version_at, type, nsfw, tags, images_downloaded) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_data['id'], model_data['name'], model_data['creator'], model_data['downloads'], 
              model_data['likes'], model_data['created_at'], model_data['last_version_at'], 
              model_data['type'], model_data['nsfw'], model_data['tags'], False))
        self.conn.commit()

    def insert_image(self, image_data):
        self.cursor.execute('''
            INSERT OR REPLACE INTO images 
            (model_id, filename, name, nsfw_level, width, height, downloaded) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (image_data['model_id'], image_data['filename'], image_data['name'], 
              image_data['nsfw_level'], image_data['width'], image_data['height'], False))
        self.conn.commit()

    def model_exists(self, model_id):
        self.cursor.execute('SELECT id FROM models WHERE id = ?', (model_id,))
        return self.cursor.fetchone() is not None

    def get_last_cursor(self):
        self.cursor.execute('SELECT last_cursor FROM scrape_status WHERE id = 1')
        result = self.cursor.fetchone()
        return result[0] if result else None

    def update_last_cursor(self, cursor):
        self.cursor.execute('INSERT OR REPLACE INTO scrape_status (id, last_cursor) VALUES (1, ?)', (cursor,))
        self.conn.commit()

    def mark_image_downloaded(self, model_id, filename):
        self.cursor.execute('UPDATE images SET downloaded = TRUE WHERE model_id = ? AND filename = ?', (model_id, filename))
        self.conn.commit()

    def check_model_images_downloaded(self, model_id):
        self.cursor.execute('SELECT COUNT(*) FROM images WHERE model_id = ? AND downloaded = FALSE', (model_id,))
        count = self.cursor.fetchone()[0]
        if count == 0:
            self.cursor.execute('UPDATE models SET images_downloaded = TRUE WHERE id = ?', (model_id,))
            self.conn.commit()

    def get_stats(self):
        self.cursor.execute('SELECT COUNT(*) FROM models')
        total_models = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT COUNT(*) FROM images WHERE downloaded = TRUE')
        total_images = self.cursor.fetchone()[0]
        
        return total_models, total_images

# API operations
class CivitaiAPI:
    def __init__(self):
        self.base_url = 'https://civitai.com/api/trpc/model.getAll'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
            'content-type': 'application/json',
            'Referer': 'https://civitai.com/models',
            'x-client': 'web',
            'x-client-version': '3.0.234'
        }

    def fetch_lora_models(self, cursor=None):
        # Fetch LoRA models from the API (implementation remains the same)

        url = 'https://civitai.com/api/trpc/model.getAll'

        inputs = {
                "json": {
                    "period": "Month",
                    "periodMode": "published",
                    "sort": "Most Downloaded",
                    "types": ["LORA"],
                    "pending": False,
                    "browsingLevel": 1,
                    "cursor": cursor
                },
                "meta": {
                    "values": {
                        "cursor": ["undefined"]
                    }
                }
            }

        if cursor is not None:
            del inputs['meta']
        
        params = {
            'input': json.dumps(inputs)
        }
        response = requests.get(url, headers=self.headers, params=params)
        return response.json()

# Image operations
class ImageDownloader:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def get_image_path(self, model_id, filename):
        model_id_str = str(model_id).zfill(3)
        dir_path = os.path.join(self.base_dir, model_id_str[-3:], str(model_id))
        os.makedirs(dir_path, exist_ok=True)
        safe_filename = self.sanitize_filename(filename)
        return os.path.join(dir_path, safe_filename)

    def sanitize_filename(self, filename):
        # Remove any character that isn't alphanumeric, dash, underscore, or dot
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        # Remove any leading or trailing dots or spaces
        filename = filename.strip('. ')
        # Ensure the filename isn't empty after sanitization
        if not filename:
            filename = 'unnamed_file'
        # Limit filename length (optional, adjust as needed)
        max_length = 255  # Maximum filename length in most file systems
        if len(filename) > max_length:
            name, ext = os.path.splitext(filename)
            filename = name[:max_length-len(ext)] + ext
        return filename

    def download_image(self, image_url, model_id, image_filename):
        try:
            image_path = self.get_image_path(model_id, image_filename)
            if not os.path.exists(image_path):
                response = requests.get(image_url)
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    return True
            return False
        except Exception as e:
            print(f"Error downloading image {image_filename} for model {model_id}: {str(e)}")
            return False

# Main scraper logic
class LoraScraper:
    def __init__(self, db_name, save_dir):
        self.db = Database(db_name)
        self.api = CivitaiAPI()
        self.image_downloader = ImageDownloader(save_dir)
        self.models_processed = 0
        self.images_downloaded = 0
        self.start_time = time.time()

    def scrape(self, max_models=None):
        cursor = self.db.get_last_cursor()
        
        while True:
            data = self.api.fetch_lora_models(cursor)

            for model in data['result']['data']['json']['items']:
                if not self.db.model_exists(model['id']) or not self.check_model_images_downloaded(model['id']):
                    self.process_model(model)
                    self.models_processed += 1
                    self.display_progress()
                
                if max_models and self.models_processed >= max_models:
                    self.display_final_stats()
                    return

            cursor = data['result']['data']['json']['nextCursor']
            if not cursor:
                break

            self.db.update_last_cursor(cursor)
            time.sleep(5)

        self.display_final_stats()

    def process_model(self, model):
        model_info = {
            'id': model['id'],
            'name': model['name'],
            'creator': model['user']['username'],
            'downloads': model['rank']['downloadCount'],
            'likes': model['rank']['thumbsUpCount'],
            'created_at': model['createdAt'],
            'last_version_at': model['lastVersionAt'],
            'type': model['type'],
            'nsfw': model['nsfw'],
            'tags': ', '.join(str(tag) for tag in model['tags']),
        }
        self.db.insert_model(model_info)

        for image in model['images']:
            image_filename = f"{model['id']}_{image['name']}.jpg"
            image_url = f"https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/{image['url']}/width=450"
            
            image_info = {
                'model_id': model['id'],
                'filename': image_filename,
                'name': image['name'],
                'nsfw_level': image['nsfwLevel'],
                'width': image['width'],
                'height': image['height'],
            }
            self.db.insert_image(image_info)

            if self.image_downloader.download_image(image_url, model['id'], image_filename):
                sanitized_filename = self.image_downloader.sanitize_filename(image_filename)
                self.db.mark_image_downloaded(model['id'], sanitized_filename)
                self.images_downloaded += 1

        self.db.check_model_images_downloaded(model['id'])

    def check_model_images_downloaded(self, model_id):
        # Check if all images for a model have been downloaded
        # This method can be implemented in the Database class if preferred
        pass

    def display_progress(self):
        total_models, total_images = self.db.get_stats()
        elapsed_time = time.time() - self.start_time
        
        print(f"\rProcessed models: {self.models_processed} | "
              f"Downloaded images: {self.images_downloaded} | "
              f"Total models in DB: {total_models} | "
              f"Total images in DB: {total_images} | "
              f"Elapsed time: {elapsed_time:.2f}s", end="")

    def display_final_stats(self):
        total_models, total_images = self.db.get_stats()
        elapsed_time = time.time() - self.start_time
        
        print("\n\nFinal Statistics:")
        print(f"Total models processed: {self.models_processed}")
        print(f"Total images downloaded: {self.images_downloaded}")
        print(f"Total models in database: {total_models}")
        print(f"Total images in database: {total_images}")
        print(f"Total elapsed time: {elapsed_time:.2f} seconds")
        print(f"Average processing speed: {self.models_processed / elapsed_time:.2f} models/second")
        print(f"Average download speed: {self.images_downloaded / elapsed_time:.2f} images/second")


if __name__ == "__main__":
    scraper = LoraScraper('lora_models.db', 'images')
    scraper.scrape()