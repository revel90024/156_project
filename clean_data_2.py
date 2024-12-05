from datasets import load_dataset, load_from_disk, Dataset
import os

print("Loading filtered dataset...")
dataset = load_from_disk("large_10M_torch2")

print(f"\nStarting size: {len(dataset):,d}")

print("\nFiltering for movies with revenue >= $50M...")
final_dataset = dataset.filter(lambda x: x['revenue'] >= 50_000_000)

print(f"\nStats:")
print(f"Original size: {len(dataset):,d}")
print(f"After filtering: {len(final_dataset):,d}")

print("\nSaving final dataset...")
final_dataset.save_to_disk("large_50M_torch")
print("\nDone! Saved to: large_50M_torch/")