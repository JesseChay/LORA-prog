import requests
import json
from datetime import datetime
import os
import csv
import time

def fetch_lora_models(cursor=None):
    url = 'https://civitai.com/api/trpc/model.getAll'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        'content-type': 'application/json',
        'Referer': 'https://civitai.com/models',
        'x-client': 'web',
        'x-client-date': str(int(datetime.now().timestamp() * 1000)),
        'x-client-version': '3.0.234'
    }
    
    params = {
        'input': json.dumps({
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
                    "cursor": ["undefined"] if cursor is None else [cursor]
                }
            }
        })
    }
    
    response = requests.get(url, headers=headers, params=params)
    return response.json()

def process_model(model, writer):
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
        'images': []
    }

    # Process images
    if not os.path.exists('images'):
        os.makedirs('images')
    
    for i, image in enumerate(model['images']):
        image_url = f"https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/{image['url']}/width=450"
        image_filename = f"{model['id']}_{i}.jpg"
        image_path = os.path.join('images', image_filename)
        
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            with open(image_path, 'wb') as f:
                f.write(image_response.content)
            
            image_info = {
                'filename': image_filename,
                'name': image['name'],
                'nsfw_level': image['nsfwLevel'],
                'width': image['width'],
                'height': image['height'],
            }
            model_info['images'].append(image_info)

    # Write model info to CSV
    writer.writerow(model_info)

    print(f"Processed model: {model_info['name']}")
    print(f"Saved {len(model_info['images'])} images")
    print("---")

def main():
    cursor = None
    models_processed = 0
    max_models = 100  # Set this to the number of models you want to process

    with open('lora_models.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'name', 'creator', 'downloads', 'likes', 'created_at', 'last_version_at', 'type', 'nsfw', 'tags', 'images']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while models_processed < max_models:
            data = fetch_lora_models(cursor)
            
            models = data['result']['data']['json']['items']
            for model in models:
                process_model(model, writer)
                models_processed += 1
                
                if models_processed >= max_models:
                    break

                time.sleep(5)
            
            if not data['result']['data']['json']['nextCursor']:
                break
            
            cursor = data['result']['data']['json']['nextCursor']

    print(f"Processed {models_processed} models in total.")

if __name__ == "__main__":
    main()