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
    
    # If you only want the first chunk, uncomment this:
    if i == 0:
        print("First chunk created, stopping as requested.")
        break

print("Chunking completed!") 