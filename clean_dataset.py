from datasets import load_dataset, Dataset
import random
import json

def clean_and_save_dataset():
    print("Loading original dataset...")
    dataset = load_dataset("skvarre/movie-posters", cache_dir="data")["train"]
    
    print("\nCleaning dataset...")
    clean_data = []
    for movie in dataset:
        # Stricter filtering for 1M dataset
        if (movie['revenue'] >= 1_000_000 and     # At least $1M revenue
            movie['image'] is not None and         # Has image
            movie['budget'] > 100_000 and          # Real budget
            movie['release_date'] >= '2000-01-01'  # Modern movies
            ):
            
            clean_movie = {
                'id': movie['id'],
                'image': movie['image'],
                'title': movie['title'],
                'revenue': movie['revenue'],
                'budget': movie['budget'],
                'release_date': movie['release_date']
            }
            clean_data.append(clean_movie)
    
    print(f"\nOriginal size: {len(dataset):,d} movies")
    print(f"Cleaned size: {len(clean_data):,d} movies")
    
    # Save cleaned dataset
    clean_dataset = Dataset.from_list(clean_data)
    clean_dataset.save_to_disk("clean_movies_1M_modern")
    print("\nSaved cleaned dataset to: clean_movies_1M_modern/")
    
    # Save stats
    revenues = [m['revenue'] for m in clean_data]
    stats = {
        'total_movies': len(clean_data),
        'revenue_stats': {
            'min': min(revenues),
            'max': max(revenues),
            'avg': sum(revenues)/len(revenues),
            'median': sorted(revenues)[len(revenues)//2]
        }
    }
    
    with open('dataset_stats_1M_modern.json', 'w') as f:
        json.dump(stats, f, indent=4)
    print("\nSaved stats to: dataset_stats_1M_modern.json")

if __name__ == "__main__":
    clean_and_save_dataset() 