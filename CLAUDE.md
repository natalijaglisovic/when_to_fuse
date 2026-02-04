# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for multimodal recommendation systems using Amazon product data. The project processes Amazon review and metadata datasets to create user-item interaction matrices enriched with multimodal features (images and text).

## Data Processing Pipeline

The codebase implements a two-stage data processing pipeline:

### Stage 1: User-Item Filtering with Images ([query_data_amazon.py](query_data_amazon.py))
1. Loads Amazon review data from JSONL files (e.g., `Video_Games.jsonl`, `Industrial_and_Scientific.jsonl`)
2. Extracts user-item interactions (user_id, asin pairs)
3. Applies iterative 5-core filtering: removes users with <5 interactions and items with <5 interactions until convergence
4. Loads metadata from corresponding JSONL files (e.g., `meta_Video_Games.jsonl`, `meta_Industrial_and_Scientific.jsonl`)
5. Filters metadata to include only items that:
   - Survived the 5-core filtering
   - Have valid image URLs in the metadata
6. Outputs CSV with columns: `user_id`, `item_id`, `image` (containing dict with image URLs)

### Stage 2: Text Feature Addition ([data_preprocessing_text.py](data_preprocessing_text.py))
1. Takes the CSV output from Stage 1 (which should have additional metadata columns: `title`, `main_category`, `description`)
2. Concatenates `title`, `main_category`, and `description` into a single `text` column
3. Handles missing values by replacing with empty strings
4. Cleans up whitespace in the concatenated text
5. Overwrites the input CSV with the new `text` column added

## Dataset Structure

### Input Files
- `Video_Games.jsonl` / `Industrial_and_Scientific.jsonl`: Review data with user-item interactions
- `meta_Video_Games.jsonl` / `meta_Industrial_and_Scientific.jsonl`: Product metadata with images, titles, descriptions

### Output Files
- `amazon_games_user_item_image.csv` / `amazon_science_user_item_image.csv`: User-item pairs with image URLs (after Stage 1)
- `amazon_games_user_item_image_text.csv`: Final dataset with images and text features (after Stage 2)

### Image Directories
- `item_images_games/`: Downloaded product images for Video Games category (17,935 images)
- `item_images_science/`: Downloaded product images for Industrial & Scientific category (12,911 images)
- Images are named using ASIN identifiers (e.g., `B0863MT183.jpg`)

## Running the Code

### Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Processing Workflow
```bash
# Stage 1: Filter user-item interactions and extract images
python query_data_amazon.py

# Stage 2: Add text features (requires CSV from Stage 1 with title/description columns)
python data_preprocessing_text.py
```

## Key Dependencies

- **pandas**: DataFrame operations and CSV I/O
- **torch, torchvision, torchaudio**: PyTorch for deep learning models
- **sentence-transformers**: Text embedding models
- **transformers**: Hugging Face transformers library
- **scikit-learn**: Machine learning utilities
- **matplotlib, seaborn**: Visualization (likely for result analysis)

## Important Implementation Details

### 5-Core Filtering ([query_data_amazon.py](query_data_amazon.py))
The iterative filtering process ensures both users and items have at least 5 interactions:
- Continues until convergence (no more users/items filtered)
- Reports statistics at each iteration
- Verifies minimum interaction counts at the end

### Image Validation ([query_data_amazon.py](query_data_amazon.py:91-101))
The `has_valid_image()` function filters out records with:
- NaN/null values
- Empty strings, lists, or dicts
- Ensures only records with valid image data are included

### Text Concatenation ([data_preprocessing_text.py](data_preprocessing_text.py:31))
Text fields are combined with space separators and cleaned to remove extra whitespace, creating a unified text representation for each item.

## Data Characteristics

Based on the processed data:
- Video Games dataset: Filtered to ensure quality user-item interactions with multimodal features
- Industrial & Scientific dataset: Similar filtering applied
- Final datasets contain: user_id, item_id, image URLs, title, category, description, and concatenated text

## Notes for Future Development

- The pipeline is designed for two Amazon categories but can be extended to other categories
- Image URLs are preserved in dictionary format with multiple resolutions (thumb, large, hi_res)
- The iterative filtering approach ensures data density for collaborative filtering models
- Text features are pre-concatenated to simplify downstream processing for text encoders
