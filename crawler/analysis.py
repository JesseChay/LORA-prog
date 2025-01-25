import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def load_data(file_path):
    return pd.read_csv(file_path)

def basic_stats(df):
    total_models = df['Model ID'].nunique()
    total_images = len(df)
    creators = df['Creator'].nunique()
    
    print(f"Total unique models: {total_models}")
    print(f"Total images: {total_images}")
    print(f"Number of unique creators: {creators}")
    
    print("\nTop 5 creators by number of images:")
    print(df['Creator'].value_counts().head())

def nsfw_analysis(df):
    nsfw_counts = df['NSFW Status'].value_counts()
    print("\nNSFW Status Distribution:")
    print(nsfw_counts)
    
    plt.figure(figsize=(8, 6))
    nsfw_counts.plot(kind='bar')
    plt.title('NSFW Status Distribution')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('nsfw_distribution.png')
    plt.close()

def file_status_analysis(df):
    status_counts = df['File Status'].value_counts()
    print("\nFile Status Distribution:")
    print(status_counts)
    
    plt.figure(figsize=(8, 6))
    status_counts.plot(kind='bar')
    plt.title('File Status Distribution')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('file_status_distribution.png')
    plt.close()

def processing_status_analysis(df):
    processing_counts = df['Processing Status'].value_counts()
    print("\nProcessing Status Distribution:")
    print(processing_counts)
    
    plt.figure(figsize=(8, 6))
    processing_counts.plot(kind='bar')
    plt.title('Processing Status Distribution')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('processing_status_distribution.png')
    plt.close()

def popularity_analysis(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Downloads'], df['Likes'])
    plt.title('Downloads vs Likes')
    plt.xlabel('Downloads')
    plt.ylabel('Likes')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('downloads_vs_likes.png')
    plt.close()

def main(file_path):
    df = load_data(file_path)
    
    basic_stats(df)
    nsfw_analysis(df)
    file_status_analysis(df)
    processing_status_analysis(df)
    popularity_analysis(df)

if __name__ == "__main__":
    main('dataset_preparation_results.csv')