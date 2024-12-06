from datasets import load_dataset, load_from_disk, Dataset
import os
"""
We loaded the massive dataset and filtered out movies with 10M which got from 
100k to about 14k and then filtered out non unique movies down to 6137 movies
"""

print("Loading filtered dataset...")
dataset = load_from_disk("large_10M_torch")
print(f"\nStarting size: {len(dataset):,d}")
print("\nRemoving duplicates...")
filtered_data = []
seen_ids = set()
count = 0
for item in dataset:
    count += 1
    if count % 100 == 0:
        print(f"Processed: {count:,d} | Found {len(seen_ids):,d} unique movies")
    
    if item['id'] not in seen_ids:
        seen_ids.add(item['id'])
        filtered_data.append(item)
        print(f"Kept: {item['title']} (${item['revenue']:,.0f})")
# Convert back to Dataset
print("\nConverting to final dataset...")
final_dataset = Dataset.from_list(filtered_data)
print(f"\nStats:")
print(f"Original size: {len(dataset):,d}")
print(f"After deduping: {len(final_dataset):,d}")
print("\nSaving final dataset...")
final_dataset.save_to_disk("large_10M_torch2")
print("\nDone! Saved to: large_10M_torch2/")