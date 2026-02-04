import pandas as pd
import ast
from collections import defaultdict
import requests
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class RecommenderDataPreprocessor:
    def __init__(self, csv_path: str, images_dir: str = "item_images_baby", max_workers: int = 20):
        """
        Initialize preprocessor
        
        Args:
            csv_path: Path to your CSV file
            images_dir: Directory to save downloaded images
            max_workers: Number of threads for parallel downloading
        """
        self.csv_path = csv_path
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy and proper headers"""
        session = requests.Session()
        
        # Amazon-friendly headers to avoid bot detection
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'image',
            'Sec-Fetch-Mode': 'no-cors',
            'Sec-Fetch-Site': 'cross-site'
        }
        session.headers.update(headers)
        
        # Retry strategy for failed requests
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1,
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
        
    def load_and_parse_data(self) -> pd.DataFrame:
        """Load CSV and parse the image column"""
        print("Loading data...")
        df = pd.read_csv(self.csv_path)
        
        # Parse the image column (it's a string representation of a dict)
        def parse_image_dict(img_str):
            try:
                if pd.isna(img_str) or img_str == '':
                    return {}
                return ast.literal_eval(img_str)
            except:
                return {}
        
        df['image_dict'] = df['image'].apply(parse_image_dict)
        print(f"Loaded {len(df)} interactions for {df['user_id'].nunique()} users and {df['item_id'].nunique()} items")
        return df
    
    def create_user_sequences(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Create user -> sequence of item IDs mapping"""
        print("Creating user sequences...")
        
        user_sequences = defaultdict(list)
        
        # Group by user and create sequences
        for _, row in df.iterrows():
            user_sequences[row['user_id']].append(row['item_id'])
        
        # Remove duplicates while preserving order
        for user_id in user_sequences:
            seen = set()
            deduplicated = []
            for item in user_sequences[user_id]:
                if item not in seen:
                    seen.add(item)
                    deduplicated.append(item)
            user_sequences[user_id] = deduplicated
        
        user_sequences = dict(user_sequences)
        
        print(f"Created sequences for {len(user_sequences)} users")
        print(f"Average sequence length: {sum(len(seq) for seq in user_sequences.values()) / len(user_sequences):.2f}")
        
        return user_sequences
    
    def create_item_to_image_mapping(self, df: pd.DataFrame, prioritize_high_res: bool = True) -> Dict[str, Dict[str, str]]:
        """
        Create item_id -> image URLs mapping with high-res priority
        
        Args:
            df: DataFrame with image data
            prioritize_high_res: If True, prioritize hi_res > large > thumb
        """
        print("Creating item to image mapping with high-res priority...")
        
        # Define image type priority (best to worst)
        if prioritize_high_res:
            priority_order = ['hi_res', 'large', 'thumb']
        else:
            priority_order = ['large', 'hi_res', 'thumb']
        
        item_to_images = {}
        hi_res_count = 0
        large_count = 0 
        thumb_count = 0
        no_image_count = 0
        
        for _, row in df.iterrows():
            if row['item_id'] not in item_to_images:
                image_dict = row['image_dict']
                
                # Find the best available image type
                best_url = None
                best_type = None
                
                for img_type in priority_order:
                    if img_type in image_dict and image_dict[img_type]:
                        url = image_dict[img_type]
                        if isinstance(url, str) and url.startswith('http'):
                            best_url = url
                            best_type = img_type
                            break
                
                if best_url:
                    item_to_images[row['item_id']] = {
                        'url': best_url,
                        'type': best_type,
                        'all_available': image_dict  # Keep all for reference
                    }
                    
                    # Track statistics
                    if best_type == 'hi_res':
                        hi_res_count += 1
                    elif best_type == 'large':
                        large_count += 1
                    elif best_type == 'thumb':
                        thumb_count += 1
                else:
                    no_image_count += 1
        
        print(f"Image mapping statistics:")
        print(f"  - High-res images: {hi_res_count}")
        print(f"  - Large images: {large_count}")
        print(f"  - Thumbnail images: {thumb_count}")
        print(f"  - No valid images: {no_image_count}")
        print(f"  - Total with images: {len(item_to_images)}")
        
        return item_to_images
    
    def _download_single_image(self, args: Tuple[str, str, str]) -> Tuple[str, bool, str]:
        """Download a single image - used for parallel processing"""
        item_id, image_url, image_type = args
        
        image_path = self.images_dir / f"{item_id}.jpg"  # Simplified naming
        
        # Skip if already exists
        if image_path.exists():
            return item_id, True, str(image_path), image_type
        
        try:
            # Small delay to be nice to the server
            time.sleep(0.1)
            
            response = self.session.get(image_url, timeout=15)
            response.raise_for_status()
            
            # Basic validation
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                raise ValueError(f"Not an image: {content_type}")
            
            # Save image
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            # Quick validation that it's a real image
            with Image.open(image_path) as img:
                width, height = img.size
                if width < 10 or height < 10:
                    raise ValueError(f"Image too small: {width}x{height}")
            
            return item_id, True, str(image_path), image_type
            
        except Exception as e:
            if image_path.exists():
                image_path.unlink()
            return item_id, False, str(e), image_type
    
    def download_images(self, item_to_images: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        """
        Download images with parallel processing, prioritizing high-res
        
        Returns:
            Dict mapping item_id to {'path': str, 'type': str, 'url': str}
        """
        print(f"Downloading images using {self.max_workers} workers...")
        
        item_to_path_info = {}
        
        # Prepare download tasks
        download_tasks = []
        for item_id, image_info in item_to_images.items():
            download_tasks.append((item_id, image_info['url'], image_info['type']))
        
        print(f"Starting download of {len(download_tasks)} images...")
        
        # Track image type statistics
        success_stats = {'hi_res': 0, 'large': 0, 'thumb': 0}
        failed_stats = {'hi_res': 0, 'large': 0, 'thumb': 0}
        
        # Download with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {
                executor.submit(self._download_single_image, task): task[0] 
                for task in download_tasks
            }
            
            completed = 0
            for future in as_completed(future_to_item):
                completed += 1
                
                if completed % 200 == 0:  # Progress update every 200 items
                    print(f"Downloaded {completed}/{len(download_tasks)} images...")
                
                try:
                    item_id, success, result, img_type = future.result()
                    
                    if success:
                        item_to_path_info[item_id] = {
                            'path': result,
                            'type': img_type,
                            'url': item_to_images[item_id]['url']
                        }
                        success_stats[img_type] += 1
                    else:
                        failed_stats[img_type] += 1
                        print(f"Failed to download {item_id} ({img_type}): {result}")
                    
                except Exception as e:
                    print(f"Unexpected error processing item: {e}")
        
        print(f"\nDownload Summary:")
        print(f"Successfully downloaded {len(item_to_path_info)} images:")
        for img_type, count in success_stats.items():
            if count > 0:
                print(f"  - {img_type}: {count}")
        
        if any(failed_stats.values()):
            print(f"Failed downloads:")
            for img_type, count in failed_stats.items():
                if count > 0:
                    print(f"  - {img_type}: {count}")
        
        return item_to_path_info
    
    def create_train_test_split(self, user_sequences: Dict[str, List[str]], 
                               test_size: float = 0.2) -> Tuple[Dict, Dict]:
        """
        Create train/test split using leave-one-out or leave-k-out strategy
        """
        print("Creating train/test split...")
        
        train_sequences = {}
        test_sequences = {}
        
        for user_id, items in user_sequences.items():
            if len(items) < 2:  # Skip users with only 1 item
                continue
                
            test_items_count = max(1, int(len(items) * test_size))
            # Ensure we have at least 1 item for training
            test_items_count = min(test_items_count, len(items) - 1)
            
            train_sequences[user_id] = items[:-test_items_count]
            test_sequences[user_id] = items[-test_items_count:]
        
        print(f"Train sequences: {len(train_sequences)} users")
        print(f"Test sequences: {len(test_sequences)} users")
        
        return train_sequences, test_sequences
    
    def save_preprocessed_data(self, user_sequences: Dict, item_to_path_info: Dict, 
                              train_sequences: Dict, test_sequences: Dict,
                              output_dir: str = "preprocessed_data"):
        """Save all preprocessed data"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save mappings
        with open(output_path / "user_sequences.pkl", 'wb') as f:
            pickle.dump(user_sequences, f)
            
        with open(output_path / "item_to_image_info.pkl", 'wb') as f:
            pickle.dump(item_to_path_info, f)
            
        with open(output_path / "train_sequences.pkl", 'wb') as f:
            pickle.dump(train_sequences, f)
            
        with open(output_path / "test_sequences.pkl", 'wb') as f:
            pickle.dump(test_sequences, f)
        
        # Create vocab mappings for model training
        all_items = set()
        for items in user_sequences.values():
            all_items.update(items)
        
        # Create item to index mapping
        item_to_idx = {item: idx + 1 for idx, item in enumerate(sorted(all_items))}  # +1 for padding
        item_to_idx['<PAD>'] = 0
        idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        
        with open(output_path / "item_to_idx.pkl", 'wb') as f:
            pickle.dump(item_to_idx, f)
            
        with open(output_path / "idx_to_item.pkl", 'wb') as f:
            pickle.dump(idx_to_item, f)
        
        # Save image type statistics
        image_stats = {}
        for item_id, info in item_to_path_info.items():
            img_type = info['type']
            if img_type not in image_stats:
                image_stats[img_type] = 0
            image_stats[img_type] += 1
        
        with open(output_path / "image_stats.txt", 'w') as f:
            f.write("Image Resolution Statistics:\n")
            f.write("=" * 30 + "\n")
            for img_type, count in sorted(image_stats.items()):
                f.write(f"{img_type}: {count}\n")
            f.write(f"\nTotal images: {sum(image_stats.values())}\n")
        
        print(f"Saved preprocessed data to {output_path}")
        print(f"Vocabulary size: {len(item_to_idx)} items")
        print(f"Image resolution breakdown: {image_stats}")
    
    def run_full_preprocessing(self, prioritize_high_res: bool = True):
        """Run the complete preprocessing pipeline with high-res priority"""
        print("Starting full preprocessing pipeline with high-res priority...")
        start_time = time.time()
        
        # Step 1: Load and parse data
        df = self.load_and_parse_data()
        
        # Step 2: Create user sequences
        user_sequences = self.create_user_sequences(df)
        
        # Step 3: Create item to image mapping (prioritizing high-res)
        item_to_images = self.create_item_to_image_mapping(df, prioritize_high_res)
        
        # Step 4: Download images
        item_to_path_info = self.download_images(item_to_images)
        
        # Step 5: Filter sequences to only include items with successfully downloaded images
        filtered_sequences = {}
        for user_id, items in user_sequences.items():
            filtered_items = [item for item in items if item in item_to_path_info]
            if len(filtered_items) > 0:
                filtered_sequences[user_id] = filtered_items
        
        print(f"After filtering for available images: {len(filtered_sequences)} users")
        
        # Step 6: Create train/test split
        train_sequences, test_sequences = self.create_train_test_split(filtered_sequences)
        
        # Step 7: Save everything
        self.save_preprocessed_data(filtered_sequences, item_to_path_info, 
                                   train_sequences, test_sequences)
        
        total_time = time.time() - start_time
        print(f"Preprocessing complete in {total_time:.1f} seconds!")
        return filtered_sequences, item_to_path_info, train_sequences, test_sequences


# Usage example
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = RecommenderDataPreprocessor(
        csv_path="amazon_baby_products_user_item_image_text.csv",
        images_dir="item_images_baby",  # Updated directory name
        max_workers=20  # Adjust based on your system
    )
    
    # Run preprocessing with high-res priority
    user_sequences, item_to_path_info, train_seq, test_seq = preprocessor.run_full_preprocessing(
        prioritize_high_res=True  # This will prioritize hi_res > large > thumb
    )