import os
import pandas as pd

chunks_dir = '/data/group_data/rag-robust-eval/data/policy/chunks'
chunks_files = os.listdir(chunks_dir)

# combine all the chunks into a single dataframe
df = pd.DataFrame()

for file in chunks_files:
    file_path = os.path.join(chunks_dir, file)
    chunk_df = pd.read_json(file_path)
    df = pd.concat([df, chunk_df], ignore_index=True)
    print(f"Loaded {file_path} with {len(chunk_df)} rows. Total rows: {len(df)}")

# save the combined dataframe to a single file
output_file = '/data/group_data/rag-robust-eval/data/policy/chunks.json'
df.to_json(output_file, orient='records')
print(f"Combined dataframe saved to {output_file} with {len(df)} rows.")