"""
Multimodal Amazon Data Processing

This script combines image and text processing to create a multimodal dataset from Amazon review and metadata.

Pipeline:
1. Load and filter review data (k-core filtering, k given by --min-interactions)
2. Optionally sample down to a target number of users/items ("top" = most
   active/popular, "random" = uniform random with a fixed seed) and re-apply
   k-core filtering so the sampled set stays consistent
3. Load and filter metadata with images and text
4. Merge and create final CSV with user_id, item_id, image, and text features

A JSON manifest recording every parameter used (including the random seed)
is written alongside the output CSV so a run can be reproduced exactly.
"""

import os
import pandas as pd
import numpy as np
import json
from typing import Any, Optional
import argparse


def has_valid_image(image_data: Any) -> bool:
    """Check if image data is not empty"""
    # Handle numpy arrays
    if isinstance(image_data, np.ndarray):
        return image_data.size > 0

    # Handle pandas NA values
    try:
        if pd.isna(image_data):
            return False
    except (ValueError, TypeError):
        # If pd.isna raises an error, continue with other checks
        pass

    # Handle empty strings
    if image_data == '' or image_data == '[]' or image_data == '{}':
        return False

    # Handle lists
    if isinstance(image_data, list) and len(image_data) == 0:
        return False

    # Handle dicts
    if isinstance(image_data, dict) and len(image_data) == 0:
        return False

    # Handle None
    if image_data is None:
        return False

    return True


def concatenate_text_fields(title: str, category: str, description: str) -> str:
    """Concatenate text fields with proper handling of missing values"""
    # Replace NaN with empty strings
    title = '' if pd.isna(title) else str(title)
    category = '' if pd.isna(category) else str(category)
    description = '' if pd.isna(description) else str(description)

    # Concatenate and clean
    text = f"{title} {category} {description}"
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def _apply_k_core(reviews: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    """Iteratively filter users/items with fewer than min_interactions interactions until stable."""
    prev_len = 0
    iteration = 0

    while len(reviews) != prev_len:
        prev_len = len(reviews)
        iteration += 1
        print(f"Iteration {iteration}:")

        item_counts = reviews['asin'].value_counts()
        popular_items = item_counts[item_counts >= min_interactions].index
        reviews = reviews[reviews['asin'].isin(popular_items)]
        print(f"  After item filtering: {len(reviews)} interactions, {reviews['asin'].nunique()} items")

        user_counts = reviews['user_id'].value_counts()
        active_users = user_counts[user_counts >= min_interactions].index
        reviews = reviews[reviews['user_id'].isin(active_users)]
        print(f"  After user filtering: {len(reviews)} interactions, {reviews['user_id'].nunique()} users")

    print(f"Converged after {iteration} iterations")
    return reviews


def load_and_filter_reviews(
    review_file: str,
    min_interactions: int = 5,
    target_users: Optional[int] = None,
    target_items: Optional[int] = None,
    sampling_strategy: str = "top",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load review data and apply iterative k-core filtering, optionally sampling to target size

    Args:
        review_file: Path to review JSONL file
        min_interactions: Minimum interactions per user/item (the "k" in k-core)
        target_users: Target number of users (optional)
        target_items: Target number of items (optional)
        sampling_strategy: "top" selects the most active users / most popular
            items (deterministic). "random" draws a uniform random sample
            using `seed`, for domains where we don't want to bias towards
            power users/items.
        seed: Random seed used only when sampling_strategy == "random".
            Fixed and recorded in the output manifest so results are
            reproducible.

    Returns:
        Filtered DataFrame with user_id and asin columns
    """
    if sampling_strategy not in ("top", "random"):
        raise ValueError(f"Unknown sampling_strategy: {sampling_strategy!r}")

    # Load review data
    print(f"Loading review data from {review_file}...")
    reviews = []
    with open(review_file, "r") as f:
        for line in f:
            reviews.append(json.loads(line))

    reviews = pd.DataFrame(reviews)
    print(f"Loaded {len(reviews)} reviews")

    # Keep necessary columns including review text
    reviews = reviews[['asin', 'user_id', 'title', 'text']]
    reviews.rename(columns={'title': 'review_title', 'text': 'review_text'}, inplace=True)

    # For duplicates, concatenate review texts and keep first review_title
    print("Aggregating multiple reviews per user-item pair...")
    reviews = reviews.groupby(['user_id', 'asin'], as_index=False).agg({
        'review_title': 'first',  # Keep first review title
        'review_text': lambda x: ' '.join(str(i) for i in x if pd.notna(i))  # Concatenate all review texts
    })
    print(f"Initial dataset: {len(reviews)} unique user-item pairs")

    # Apply iterative k-core filtering
    print(f"Filtering items and users with at least {min_interactions} interactions...")
    reviews = _apply_k_core(reviews, min_interactions)

    # Report statistics after k-core filtering
    num_users = reviews['user_id'].nunique()
    num_items = reviews['asin'].nunique()
    print(f"After k-core filtering: {num_users} unique users and {num_items} unique items.")

    # Sample to target size if specified
    if target_items is not None or target_users is not None:
        print(f"\nSampling to target size (strategy={sampling_strategy}, seed={seed})...")
        rng = np.random.default_rng(seed) if sampling_strategy == "random" else None

        if target_items is not None and num_items > target_items:
            item_counts = reviews['asin'].value_counts()
            if sampling_strategy == "random":
                selected_items = rng.choice(item_counts.index.values, size=target_items, replace=False)
            else:
                selected_items = item_counts.nlargest(target_items).index
            reviews = reviews[reviews['asin'].isin(selected_items)]
            print(f"  Sampled {target_items} items ({sampling_strategy})")
            print(f"  After item sampling: {len(reviews)} interactions, {reviews['user_id'].nunique()} users")

        if target_users is not None and reviews['user_id'].nunique() > target_users:
            user_counts = reviews['user_id'].value_counts()
            if sampling_strategy == "random":
                selected_users = rng.choice(user_counts.index.values, size=target_users, replace=False)
            else:
                selected_users = user_counts.nlargest(target_users).index
            reviews = reviews[reviews['user_id'].isin(selected_users)]
            print(f"  Sampled {target_users} users ({sampling_strategy})")
            print(f"  After user sampling: {len(reviews)} interactions, {reviews['asin'].nunique()} items")

        # Re-apply k-core filtering after sampling to maintain min_interactions
        print(f"\nRe-applying {min_interactions}-core filtering after sampling...")
        reviews = _apply_k_core(reviews, min_interactions)

    # Final statistics
    num_users = reviews['user_id'].nunique()
    num_items = reviews['asin'].nunique()
    print(f"\nFinal filtered dataset contains {num_users} unique users and {num_items} unique items.")

    # Verify the filtering
    min_user_interactions = reviews['user_id'].value_counts().min()
    min_item_interactions = reviews['asin'].value_counts().min()
    print(f"Verification - Min interactions per user: {min_user_interactions}")
    print(f"Verification - Min interactions per item: {min_item_interactions}")

    return reviews


def load_and_filter_metadata(metadata_file: str, target_asins: set) -> pd.DataFrame:
    """
    Load metadata and extract images and text features for filtered items

    Args:
        metadata_file: Path to metadata JSONL file
        target_asins: Set of ASINs to keep

    Returns:
        DataFrame with parent_asin, image, title, main_category, description columns
    """
    print(f"Loading and filtering metadata from {metadata_file}...")

    filtered_metadata = []
    with open(metadata_file, "r") as f:
        for line in f:
            record = json.loads(line)
            parent_asin = record.get("parent_asin")

            if parent_asin in target_asins:
                # Extract images
                images = record.get("images", [])
                image_data = images[0] if isinstance(images, list) and len(images) > 0 else images

                # Only keep records with valid images
                if has_valid_image(image_data):
                    # Extract text fields
                    title = record.get("title", "")
                    main_category = record.get("main_category", "")
                    description = record.get("description", "")

                    # Handle description if it's a list
                    if isinstance(description, list):
                        description = ' '.join(description)

                    filtered_metadata.append({
                        'parent_asin': parent_asin,
                        'image': image_data,
                        'title': title,
                        'main_category': main_category,
                        'description': description
                    })

    metadata_df = pd.DataFrame(filtered_metadata)
    print(f"Collected {len(metadata_df)} relevant metadata entries with images and text.")

    return metadata_df


def merge_and_create_text(reviews_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge reviews with metadata and create combined text column

    Args:
        reviews_df: Filtered reviews DataFrame (with review_title and review_text)
        metadata_df: Filtered metadata DataFrame

    Returns:
        Final DataFrame with multimodal features including combined text from reviews and metadata
    """
    print("Merging review data with metadata...")
    final_df = reviews_df.merge(
        metadata_df,
        left_on='asin',
        right_on='parent_asin',
        how='inner'
    )
    print(f"After merge: {len(final_df)} records")

    # Create combined text column: review title + review text + product title + category + description
    print("Creating combined text column (review + metadata)...")
    final_df['combined_text'] = final_df.apply(
        lambda row: concatenate_text_fields(
            f"{str(row['review_title']) if pd.notna(row['review_title']) else ''} {str(row['review_text']) if pd.notna(row['review_text']) else ''}",
            row['title'],
            f"{row['main_category']} {row['description']}"
        ),
        axis=1
    )

    # Select and rename final columns
    result_df = final_df[[
        'user_id', 'asin', 'image',
        'review_title', 'review_text',
        'title', 'main_category', 'description',
        'combined_text'
    ]].copy()
    result_df.rename(columns={
        'asin': 'item_id',
        'title': 'product_title'
    }, inplace=True)

    return result_df


def print_statistics(df: pd.DataFrame):
    """Print statistics about the final dataset"""
    final_users = df['user_id'].nunique()
    final_items = df['item_id'].nunique()

    print(f"\n{'='*60}")
    print("FINAL DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total interactions: {len(df)}")
    print(f"Unique users: {final_users}")
    print(f"Unique items: {final_items}")
    print(f"User/Item ratio: {final_users/final_items:.2f}")
    print(f"Avg interactions per user: {len(df)/final_users:.2f}")
    print(f"Avg interactions per item: {len(df)/final_items:.2f}")

    # Check for any missing text
    empty_text_count = (df['combined_text'].str.strip() == '').sum()
    print(f"\nRecords with empty combined text: {empty_text_count}")

    # Text length analysis
    df['text_length'] = df['combined_text'].str.len()
    print(f"\nCombined text length statistics:")
    print(f"  Mean: {df['text_length'].mean():.2f} characters")
    print(f"  Median: {df['text_length'].median():.2f} characters")
    print(f"  Min: {df['text_length'].min()} characters")
    print(f"  Max: {df['text_length'].max()} characters")

    # Review text statistics
    if 'review_text' in df.columns:
        df['review_text_length'] = df['review_text'].str.len()
        print(f"\nReview text length statistics:")
        print(f"  Mean: {df['review_text_length'].mean():.2f} characters")
        print(f"  Median: {df['review_text_length'].median():.2f} characters")
        df.drop('review_text_length', axis=1, inplace=True)

    # User/Item interaction distributions
    user_interaction_counts = df['user_id'].value_counts()
    item_interaction_counts = df['item_id'].value_counts()

    print(f"\nUser interactions per user:")
    print(f"  Mean: {user_interaction_counts.mean():.2f}")
    print(f"  Median: {user_interaction_counts.median():.2f}")
    print(f"  Min: {user_interaction_counts.min()}")
    print(f"  Max: {user_interaction_counts.max()}")

    print(f"\nItem interactions per item:")
    print(f"  Mean: {item_interaction_counts.mean():.2f}")
    print(f"  Median: {item_interaction_counts.median():.2f}")
    print(f"  Min: {item_interaction_counts.min()}")
    print(f"  Max: {item_interaction_counts.max()}")
    print(f"{'='*60}\n")

    # Remove temporary column
    df.drop('text_length', axis=1, inplace=True)


def write_manifest(manifest_path: str, args: argparse.Namespace, df: pd.DataFrame):
    """
    Record every parameter used to produce this CSV (including the random
    seed) so the run can be reproduced exactly. This is what makes sampled
    datasets (e.g. the random Baby sample) reproducible.
    """
    manifest = {
        'category': args.category,
        'review_file': f"{args.category}.jsonl",
        'metadata_file': f"meta_{args.category}.jsonl",
        'min_interactions': args.min_interactions,
        'target_users': args.target_users,
        'target_items': args.target_items,
        'sampling_strategy': args.sampling_strategy,
        'seed': args.seed if args.sampling_strategy == 'random' else None,
        'final_num_interactions': len(df),
        'final_num_users': int(df['user_id'].nunique()),
        'final_num_items': int(df['item_id'].nunique()),
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved reproducibility manifest to {manifest_path}")


def main():
    """Main processing pipeline"""
    parser = argparse.ArgumentParser(
        description='Process Amazon review and metadata into multimodal dataset'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='Video_Games',
        help='Dataset category (e.g., Video_Games, Industrial_and_Scientific, Baby_Products)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory containing <category>.jsonl and meta_<category>.jsonl (default: current directory)'
    )
    parser.add_argument(
        '--min-interactions',
        type=int,
        default=5,
        help='Minimum interactions per user/item for k-core filtering (e.g. 4 for science, 5 for games/baby)'
    )
    parser.add_argument(
        '--target-users',
        type=int,
        default=None,
        help='Target number of users to sample down to (optional)'
    )
    parser.add_argument(
        '--target-items',
        type=int,
        default=None,
        help='Target number of items to sample down to (optional)'
    )
    parser.add_argument(
        '--sampling-strategy',
        type=str,
        choices=['top', 'random'],
        default='top',
        help="'top' keeps the most active users / most popular items (deterministic). "
             "'random' draws a uniform random sample using --seed (used for Baby)."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for --sampling-strategy random (recorded in the output manifest)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: amazon_{category}_user_item_image_text.csv)'
    )

    args = parser.parse_args()

    # Configuration
    category = args.category
    min_interactions = args.min_interactions
    target_users = args.target_users
    target_items = args.target_items

    # File paths
    review_file = os.path.join(args.data_dir, f"{category}.jsonl")
    metadata_file = os.path.join(args.data_dir, f"meta_{category}.jsonl")
    output_file = args.output or os.path.join(args.data_dir, f"amazon_{category.lower()}_user_item_image_text.csv")

    print(f"\n{'='*60}")
    print("MULTIMODAL AMAZON DATA PROCESSING")
    print(f"{'='*60}")
    print(f"Category: {category}")
    print(f"Min interactions (k-core): {min_interactions}")
    print(f"Target users: {target_users if target_users else 'No limit'}")
    print(f"Target items: {target_items if target_items else 'No limit'}")
    print(f"Sampling strategy: {args.sampling_strategy}")
    if args.sampling_strategy == 'random':
        print(f"Random seed: {args.seed}")
    print(f"Review file: {review_file}")
    print(f"Metadata file: {metadata_file}")
    print(f"Output file: {output_file}")
    print(f"{'='*60}\n")

    # Step 1: Load and filter reviews
    filtered_reviews = load_and_filter_reviews(
        review_file,
        min_interactions,
        target_users=target_users,
        target_items=target_items,
        sampling_strategy=args.sampling_strategy,
        seed=args.seed,
    )

    # Step 2: Load and filter metadata
    target_asins = set(filtered_reviews['asin'].unique())
    metadata_df = load_and_filter_metadata(metadata_file, target_asins)

    # Step 3: Merge and create text column
    result_df = merge_and_create_text(filtered_reviews, metadata_df)

    # Step 4: Print statistics
    print_statistics(result_df)

    # Step 5: Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Saved dataset to {output_file}")
    print(f"Dataset shape: {result_df.shape}")

    # Step 6: Save reproducibility manifest
    manifest_path = f"{output_file}.manifest.json"
    write_manifest(manifest_path, args, result_df)

    # Display preview
    print("\nFirst 3 rows preview:")
    print(result_df.head(3).to_string())

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
