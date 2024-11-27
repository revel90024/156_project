from datasets import load_dataset, Dataset
import random
import json

def clean_and_save_dataset():
    print("Loading original dataset...")
    dataset = load_dataset("skvarre/movie-posters", cache_dir="data")["train"]
    
    print("\nCleaning dataset...")
    clean_data = []
    for movie in dataset:
        # Keep only if:
        if (movie['revenue'] >= 10_000_000 and    # Has meaningful revenue
            movie['image'] is not None and      # Has image
            movie['budget'] > 0):               # Has budget info
            
            # Keep only the fields we need
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
    clean_dataset.save_to_disk("clean_movies_10M")
    print("\nSaved cleaned dataset to: clean_movies_10M/")
    
    # Save some stats as JSON
    revenues = [m['revenue'] for m in clean_data]
    stats = {
        'total_movies': len(clean_data),
        'revenue_stats': {
            'min': min(revenues),
            'max': max(revenues),
            'avg': sum(revenues)/len(revenues)
        }
    }
    
    with open('dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Print stats
    print(f"\nDataset stats:")
    print(f"Total movies: {stats['total_movies']:,d}")
    print(f"Revenue range: ${stats['revenue_stats']['min']:,.2f} - ${stats['revenue_stats']['max']:,.2f}")
    print(f"Average revenue: ${stats['revenue_stats']['avg']:,.2f}")

if __name__ == "__main__":
    clean_and_save_dataset() 