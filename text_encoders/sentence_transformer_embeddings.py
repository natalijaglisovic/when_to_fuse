"""python sentence_transformer_embeddings.py --csv_file ../Data/games/amazon_games_user_item_image_text.csv --dataset_name amazon_games
python sentence_transformer_embeddings.py --csv_file ../Data/baby/amazon_baby_products_user_item_image_text.csv --dataset_name amazon_baby"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

MODEL_NAME = "all-MiniLM-L6-v2"  # Fast and efficient sentence transformer model
BATCH_SIZE = 32  # Process multiple texts at once for efficiency


def encode_text(model, text):
    """Encode a single text string into an embedding"""
    try:
        if pd.isna(text) or text == "" or text.strip() == "":
            print(f"  Warning: Empty or NaN text, skipping")
            return None

        print(f"  Encoding text (length: {len(text)} chars)...")

        with torch.no_grad():
            embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)

        if device == "cuda":
            torch.cuda.empty_cache()

        return embedding

    except Exception as e:
        print(f"  Error encoding text: {e}")
        return None


def main(args):
    output_folder = os.path.join("embeddings", args.dataset_name, "sentence_transformer")
    os.makedirs(output_folder, exist_ok=True)

    print(f"Loading CSV from {args.csv_file}...")
    try:
        df = pd.read_csv(args.csv_file)
        print(f"Loaded CSV with {len(df)} rows")

        if 'combined_text' not in df.columns:
            print(f"Error: CSV must have 'combined_text' column. Found columns: {df.columns.tolist()}")
            return

        if 'item_id' not in df.columns:
            print(f"Error: CSV must have 'item_id' column. Found columns: {df.columns.tolist()}")
            return

    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Loading {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME, device=device)

        if device == "cuda":
            print(f"GPU memory after model load: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

        print("Model loaded successfully!")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get unique item_id and combined_text pairs
    unique_items = df[['item_id', 'combined_text']].drop_duplicates(subset=['item_id'])
    print(f"Found {len(unique_items)} unique items to process")

    processed = 0
    errors = 0
    skipped = 0

    for i, row in unique_items.iterrows():
        item_id = row['item_id']
        text = row['combined_text']

        out_path = os.path.join(output_folder, f"{item_id}.npz")
        if os.path.exists(out_path):
            print(f"[{processed+errors+skipped+1}/{len(unique_items)}] Skipping {item_id} (already exists)")
            skipped += 1
            continue

        print(f"[{processed+errors+skipped+1}/{len(unique_items)}] Encoding {item_id}...")

        if device == "cuda":
            print(f"  GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

        emb = encode_text(model, text)

        if emb is not None:
            np.savez(out_path, sentence_transformer=emb)
            processed += 1
            print(f"  Saved embedding for {item_id} (shape: {emb.shape})")
        else:
            errors += 1
            print(f"  Failed to process {item_id}")

        if (processed + errors) % 10 == 0:
            print("  🧹 Running garbage collection...")
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print(f"Successfully processed: {processed} items")
    print(f"Errors: {errors} items")
    print(f"Skipped (already exists): {skipped} items")
    print(f"Embeddings saved in: {output_folder}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", required=True, help="Path to CSV file with combined_text column")
    parser.add_argument("--dataset_name", required=True, help="Dataset name for output folder")
    args = parser.parse_args()
    main(args)
