import pandas as pd
import os

# Define chunk size (number of rows per chunk)
CHUNK_SIZE = 100000  # Adjust this based on your needs

# Path to the large CSV file
file_path = 'P6269_1_50_DMK_Sample_Elek/P6269_1_50_DMK_Sample_Elek.csv'

print(f"Processing {file_path}...")

# Create output directory for chunks if it doesn't exist
output_dir = 'P6269_1_50_DMK_Sample_Elek/chunks'
os.makedirs(output_dir, exist_ok=True)

# Use pandas read_csv with chunksize parameter
# This processes the CSV in chunks without loading the entire file into memory
for i, chunk in enumerate(pd.read_csv(file_path, chunksize=CHUNK_SIZE)):
    chunk_file = os.path.join(output_dir, f'chunk_{i}.csv')
    print(f"Writing chunk {i} to {chunk_file}")
    chunk.to_csv(chunk_file, index=False)
    
    # Stop after creating 2 chunks
    if i == 1:
        print("Two chunks created, stopping as requested.")
        break

# Combine the first two chunks into a single file
print("Combining chunks 0 and 1...")
chunk0_path = os.path.join(output_dir, 'chunk_0.csv')
chunk1_path = os.path.join(output_dir, 'chunk_1.csv')
combined_path = os.path.join(output_dir, 'chunckCombined1_2.csv')

# Read both chunks
chunk0 = pd.read_csv(chunk0_path)
chunk1 = pd.read_csv(chunk1_path)

# Combine chunks
combined = pd.concat([chunk0, chunk1], ignore_index=True)

# Save combined file
combined.to_csv(combined_path, index=False)
print(f"Combined file saved to {combined_path}")

print("Chunking completed!")