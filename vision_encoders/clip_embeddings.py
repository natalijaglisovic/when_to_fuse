"""python extract_embeddings_clip.py --image_folder ../item_images --dataset_name amazon
python extract_embeddings_clip.py --image_folder ../ikea_images --dataset_name ikea"""

import os
import argparse
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPVisionModel  
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_IMAGE_SIZE = 512


def resize_image_if_needed(image, max_size=MAX_IMAGE_SIZE):
    """Resize image if it's too large"""
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def encode_image(processor, model, image_path):
    try:
        print(f"  Loading image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image = resize_image_if_needed(image)
        
        print(f"  Processing image size: {image.size}")
        inputs = processor(images=image, return_tensors="pt") #only images

        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        emb = outputs.pooler_output.squeeze(0)
        emb_cpu = emb.cpu().numpy()
        
        del inputs, outputs, emb
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return emb_cpu
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main(args):
    output_folder = os.path.join("embeddings", args.dataset_name, "clip_science") #change here to e.g. cliphh or clip_art depending on amazon dataset
    os.makedirs(output_folder, exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    try:
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        model = CLIPVisionModel.from_pretrained(MODEL_NAME)  # Vision model only
        
        if device == "cuda":
            model = model.to(device)
            print(f"GPU memory after model load: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            
        model.eval()
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    image_files = sorted([f for f in os.listdir(args.image_folder) 
                         if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
    
    print(f"Found {len(image_files)} images to process")
    
    if not image_files:
        print("No image files found!")
        return

    processed = 0
    errors = 0
    
    for i, img_file in enumerate(image_files):
        product_id = os.path.splitext(img_file)[0]
        image_path = os.path.join(args.image_folder, img_file)
        
        out_path = os.path.join(output_folder, f"{product_id}.npz")
        if os.path.exists(out_path):
            print(f"[{i+1}/{len(image_files)}] Skipping {product_id} (already exists)")
            continue
            
        print(f"[{i+1}/{len(image_files)}] Encoding {product_id}...")
        
        if device == "cuda":
            print(f"  GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

        emb = encode_image(processor, model, image_path)
        
        if emb is not None:
            np.savez(out_path, clip=emb)
            processed += 1
            print(f"Saved embedding for {product_id}")
        else:
            errors += 1
            print(f"Failed to process {product_id}")
        
        if (i + 1) % 5 == 0:
            print("  🧹 Running garbage collection...")
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    print(f"Successfully processed: {processed} images")
    print(f"Errors: {errors} images")
    print(f"Embeddings saved in: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", required=True, help="Path to image folder")
    parser.add_argument("--dataset_name", required=True, help="Dataset name")
    args = parser.parse_args()
    main(args)