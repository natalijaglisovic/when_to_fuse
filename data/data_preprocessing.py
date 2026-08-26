import json
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import pickle


class RecommenderDataPreprocessor:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load_and_parse_data(self) -> pd.DataFrame:
        """Load CSV"""
        print("Loading data...")
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} interactions for {df['user_id'].nunique()} users and {df['item_id'].nunique()} items")
        return df

    def apply_k_core_filter(self, df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        """Iteratively remove users and items with fewer than k interactions until stable."""
        print(f"Applying {k}-core filter...")
        while True:
            prev_len = len(df)
            item_counts = df['item_id'].value_counts()
            df = df[df['item_id'].isin(item_counts[item_counts >= k].index)]
            user_counts = df['user_id'].value_counts()
            df = df[df['user_id'].isin(user_counts[user_counts >= k].index)]
            if len(df) == prev_len:
                break
        df = df.reset_index(drop=True)
        print(f"After {k}-core: {len(df)} interactions, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
        return df

    def create_user_sequences(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Create user -> sequence of item IDs mapping"""
        print("Creating user sequences...")

        user_sequences = defaultdict(list)
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

    def create_text_dataframe(self, jsonl_path: str) -> pd.DataFrame:
        """
        Create a DataFrame with itemid, userid, and text (from the title field of the review JSONL).

        Args:
            jsonl_path: Path to e.g. Baby_Products.jsonl
        """
        print(f"Loading text data from {jsonl_path}...")
        records = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records.append({
                    'itemid': record.get('asin'),
                    'userid': record.get('user_id'),
                    'timestamp': record.get('timestamp'),
                    'text': record.get('title'),
                })
        df = pd.DataFrame(records, columns=['itemid', 'userid', 'timestamp', 'text'])
        df.insert(0, 'review_id', range(len(df)))
        print(f"Created text DataFrame with {len(df)} rows, {df['itemid'].nunique()} unique items, {df['userid'].nunique()} unique users")
        return df

    def create_train_test_split(self, user_sequences: Dict[str, List[str]],
                               test_size: float = 0.2) -> Tuple[Dict, Dict]:
        """Create train/test split using leave-k-out strategy"""
        print("Creating train/test split...")

        train_sequences = {}
        test_sequences = {}

        for user_id, items in user_sequences.items():
            if len(items) < 2:
                continue
            test_items_count = max(1, int(len(items) * test_size))
            test_items_count = min(test_items_count, len(items) - 1)
            train_sequences[user_id] = items[:-test_items_count]
            test_sequences[user_id] = items[-test_items_count:]

        print(f"Train sequences: {len(train_sequences)} users")
        print(f"Test sequences: {len(test_sequences)} users")
        return train_sequences, test_sequences

    def save_preprocessed_data(self, user_sequences: Dict, train_sequences: Dict,
                               test_sequences: Dict, text_df: pd.DataFrame = None,
                               output_dir: str = "preprocessed_data"):
        """Save all preprocessed data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "user_sequences.pkl", 'wb') as f:
            pickle.dump(user_sequences, f)
        with open(output_path / "train_sequences.pkl", 'wb') as f:
            pickle.dump(train_sequences, f)
        with open(output_path / "test_sequences.pkl", 'wb') as f:
            pickle.dump(test_sequences, f)

        all_items = set()
        for items in user_sequences.values():
            all_items.update(items)

        item_to_idx = {item: idx + 1 for idx, item in enumerate(sorted(all_items))}
        item_to_idx['<PAD>'] = 0
        idx_to_item = {idx: item for item, idx in item_to_idx.items()}

        with open(output_path / "item_to_idx.pkl", 'wb') as f:
            pickle.dump(item_to_idx, f)
        with open(output_path / "idx_to_item.pkl", 'wb') as f:
            pickle.dump(idx_to_item, f)

        if text_df is not None:
            text_df.to_csv(output_path / "text_data.csv", index=False)
            print(f"Saved text_data.csv ({len(text_df)} rows)")

        print(f"Saved preprocessed data to {output_path}")
        print(f"Vocabulary size: {len(item_to_idx)} items")

    def run_full_preprocessing(self, jsonl_path: str, output_dir: str = "preprocessed_data", k: int = 5):
        """Run the complete preprocessing pipeline

        Args:
            jsonl_path: Path to the raw review JSONL for this domain (used to pull review titles as text)
            output_dir: Where to write the preprocessed pickles/CSV
            k: k-core threshold to re-apply here. The CSV coming out of
               process_multimodal_data.py is already k-core filtered (and,
               for sampled domains like Baby, re-filtered after sampling),
               so this is a no-op in the normal pipeline; it exists so this
               script is reproducible on its own if run against a CSV that
               hasn't already been through that filtering.
        """
        print("Starting full preprocessing pipeline...")

        df = self.load_and_parse_data()
        df = self.apply_k_core_filter(df, k=k)
        user_sequences = self.create_user_sequences(df)
        train_sequences, test_sequences = self.create_train_test_split(user_sequences)

        text_df = self.create_text_dataframe(jsonl_path)
        valid_users = set(df['user_id'])
        valid_items = set(df['item_id'])
        text_df = text_df[
            text_df['userid'].isin(valid_users) & text_df['itemid'].isin(valid_items)
        ].reset_index(drop=True)
        text_df['review_id'] = range(len(text_df))
        print(f"After filtering to interaction users/items: {len(text_df)} reviews")

        self.save_preprocessed_data(user_sequences, train_sequences, test_sequences, text_df, output_dir)

        return user_sequences, train_sequences, test_sequences, text_df


if __name__ == "__main__":
    # k mirrors the k-core threshold used upstream in process_multimodal_data.py
    # for each domain: 4-core for science, 5-core for games, 5-core for baby
    # (baby is additionally random-sampled to ~9k users / ~2k items upstream,
    # see process_multimodal_data.py --sampling-strategy random --seed 42).
    DOMAINS = {
        "baby":    ("data/baby/amazon_baby_products_user_item_image_text.csv", "data/baby/Baby_Products.jsonl", 5),
        "games":   ("data/games/amazon_games_user_item_image_text.csv",        "data/games/Video_Games.jsonl", 5),
        "science": ("data/science/amazon_science_user_item_image.csv",          "data/science/Industrial_and_Scientific.jsonl", 4),
    }

    for domain, (csv_path, jsonl_path, k) in DOMAINS.items():
        print(f"\n{'='*60}\nDomain: {domain}\n{'='*60}")
        preprocessor = RecommenderDataPreprocessor(csv_path=csv_path)
        user_sequences, train_seq, test_seq, text_df = preprocessor.run_full_preprocessing(
            jsonl_path=jsonl_path,
            output_dir=f"preprocessed_data/{domain}",
            k=k,
        )
        print(text_df.head())
