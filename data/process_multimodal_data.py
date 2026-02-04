"""
Multimodal Amazon Data Processing

This script combines image and text processing to create a multimodal dataset from Amazon review and metadata.

Pipeline:
1. Load and filter review data (5-core filtering)
2. Load and filter metadata with images and text
3. Merge and create final CSV with user_id, item_id, image, and text features
"""

import pandas as pd
import json
from typing import Any
import argparse


def has_valid_image(image_data: Any) -> bool:
    """Check if image data is not empty"""
    import numpy as np

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


def load_and_filter_reviews(
    review_file: str,
    min_interactions: int = 5,
    target_users: int = None,
    target_items: int = None
) -> pd.DataFrame:
    """
    Load review data and apply iterative k-core filtering, optionally sampling to target size

    Args:
        review_file: Path to review JSONL file
        min_interactions: Minimum interactions per user/item
        target_users: Target number of users (optional, will sample most active users)
        target_items: Target number of items (optional, will sample most popular items)

    Returns:
        Filtered DataFrame with user_id and asin columns
    """
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
    prev_len = 0
    iteration = 0

    while len(reviews) != prev_len:
        prev_len = len(reviews)
        iteration += 1
        print(f"Iteration {iteration}:")

        # Filter items with >= min_interactions interactions
        item_counts = reviews['asin'].value_counts()
        popular_items = item_counts[item_counts >= min_interactions].index
        reviews = reviews[reviews['asin'].isin(popular_items)]
        print(f"  After item filtering: {len(reviews)} interactions, {reviews['asin'].nunique()} items")

        # Filter users with >= min_interactions interactions
        user_counts = reviews['user_id'].value_counts()
        active_users = user_counts[user_counts >= min_interactions].index
        reviews = reviews[reviews['user_id'].isin(active_users)]
        print(f"  After user filtering: {len(reviews)} interactions, {reviews['user_id'].nunique()} users")

    print(f"\nConverged after {iteration} iterations")

    # Report statistics after k-core filtering
    num_users = reviews['user_id'].nunique()
    num_items = reviews['asin'].nunique()
    print(f"After k-core filtering: {num_users} unique users and {num_items} unique items.")

    # Sample to target size if specified
    if target_items is not None or target_users is not None:
        print(f"\nSampling to target size...")

        if target_items is not None and num_items > target_items:
            # Select top N most popular items
            item_counts = reviews['asin'].value_counts()
            top_items = item_counts.nlargest(target_items).index
            reviews = reviews[reviews['asin'].isin(top_items)]
            print(f"  Sampled top {target_items} most popular items")
            print(f"  After item sampling: {len(reviews)} interactions, {reviews['user_id'].nunique()} users")

        if target_users is not None and reviews['user_id'].nunique() > target_users:
            # Select top N most active users
            user_counts = reviews['user_id'].value_counts()
            top_users = user_counts.nlargest(target_users).index
            reviews = reviews[reviews['user_id'].isin(top_users)]
            print(f"  Sampled top {target_users} most active users")
            print(f"  After user sampling: {len(reviews)} interactions, {reviews['asin'].nunique()} items")

        # Re-apply k-core filtering after sampling to maintain min_interactions
        print(f"\nRe-applying {min_interactions}-core filtering after sampling...")
        prev_len = 0
        iteration = 0

        while len(reviews) != prev_len:
            prev_len = len(reviews)
            iteration += 1
            print(f"Iteration {iteration}:")

            # Filter items with >= min_interactions interactions
            item_counts = reviews['asin'].value_counts()
            popular_items = item_counts[item_counts >= min_interactions].index
            reviews = reviews[reviews['asin'].isin(popular_items)]
            print(f"  After item filtering: {len(reviews)} interactions, {reviews['asin'].nunique()} items")

            # Filter users with >= min_interactions interactions
            user_counts = reviews['user_id'].value_counts()
            active_users = user_counts[user_counts >= min_interactions].index
            reviews = reviews[reviews['user_id'].isin(active_users)]
            print(f"  After user filtering: {len(reviews)} interactions, {reviews['user_id'].nunique()} users")

        print(f"Converged after {iteration} iterations")

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


def main():
    """Main processing pipeline"""
    parser = argparse.ArgumentParser(
        description='Process Amazon review and metadata into multimodal dataset'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='Video_Games',
        help='Dataset category (e.g., Video_Games, Industrial_and_Scientific)'
    )
    parser.add_argument(
        '--min-interactions',
        type=int,
        default=5,
        help='Minimum interactions per user/item for filtering'
    )
    parser.add_argument(
        '--target-users',
        type=int,
        default=None,
        help='Target number of users to sample (selects most active users)'
    )
    parser.add_argument(
        '--target-items',
        type=int,
        default=None,
        help='Target number of items to sample (selects most popular items)'
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
    review_file = f"{category}.jsonl"
    metadata_file = f"meta_{category}.jsonl"
    output_file = args.output or f"amazon_{category.lower()}_user_item_image_text.csv"

    print(f"\n{'='*60}")
    print("MULTIMODAL AMAZON DATA PROCESSING")
    print(f"{'='*60}")
    print(f"Category: {category}")
    print(f"Min interactions: {min_interactions}")
    print(f"Target users: {target_users if target_users else 'No limit'}")
    print(f"Target items: {target_items if target_items else 'No limit'}")
    print(f"Review file: {review_file}")
    print(f"Metadata file: {metadata_file}")
    print(f"Output file: {output_file}")
    print(f"{'='*60}\n")

    # Step 1: Load and filter reviews
    filtered_reviews = load_and_filter_reviews(
        review_file,
        min_interactions,
        target_users=target_users,
        target_items=target_items
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

    # Display preview
    print("\nFirst 3 rows preview:")
    print(result_df.head(3).to_string())

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
