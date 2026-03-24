#!/usr/bin/env python3
"""
SAM3 Dataset Visualization Script (Grid Layout)

Creates a combined image per galaxy/orientation optimized for the refactored data pipeline.
Layout: 4 Columns grid.
- Columns: Original, Streams, Satellites, Combined
- Rows: Preprocessing variants (e.g., asinh_stretch, linear_magnitude)

Usage:
    python scripts/visualize_sam3.py [--max N] [--galaxy NAME] [--num_proc N]

Environment:
    Requires PIL, numpy, multiprocessing.
"""

import argparse
import json
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.coco_utils import decode_rle

# ================= Configuration =================
BASE_DIR = Path("/home/yuqyan/Yuqi/LSB-AI-Detection/data/02_processed/sam3_prepared")
JSON_PATH = BASE_DIR / "annotations.json"
IMAGES_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "visualizations_grid"
# =================================================

def draw_rectangle(img, x, y, w, h, color, thickness=2):
    """Draw a rectangle inside img array inplace."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    img[max(0,y):min(img.shape[0],y+thickness), max(0,x):min(img.shape[1],x+w)] = color
    img[max(0,y+h-thickness):min(img.shape[0],y+h), max(0,x):min(img.shape[1],x+w)] = color
    img[max(0,y):min(img.shape[0],y+h), max(0,x):min(img.shape[1],x+thickness)] = color
    img[max(0,y):min(img.shape[0],y+h), max(0,x+w-thickness):min(img.shape[1],x+w)] = color

def add_text_label(img_array, text, position=(10, 10), font_size=20, color=(255, 255, 255), bg_color=None):
    """Adds text to an image array."""
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    x, y = position
    if bg_color:
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle((bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2), fill=bg_color)

    # Shadow
    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
        draw.text((x+dx, y+dy), text, font=font, fill=(0, 0, 0))
    
    draw.text(position, text, font=font, fill=color)
    return np.array(img)

def create_mask_visualization(original_img, anns, cat_id_to_info, target_type):
    """
    target_type: 'streams', 'satellites', or 'combined'
    """
    h, w = original_img.shape[:2]
    overlay = original_img.copy()
    mask_layer = np.zeros((h, w, 3), dtype=np.uint8)
    
    instance_count = 0
    cmap = plt.get_cmap('tab20')
    
    for ann in anns:
        cat_name = cat_id_to_info[ann['category_id']]['name']
        is_stream = 'stream' in cat_name
        is_sat = 'satellite' in cat_name
        
        if target_type == 'streams' and not is_stream: continue
        if target_type == 'satellites' and not is_sat: continue
        
        # Get color from colormap and convert to 8-bit RGB
        color = np.array(cmap(instance_count % 20)[:3]) * 255
        color = color.astype(np.uint8)
        
        if 'segmentation' in ann and ann['segmentation']:
            seg = ann['segmentation']
            mask_binary = decode_rle(seg)
            
            mask_region = mask_binary > 0
            # Broadcast color fast
            mask_layer[mask_region] = color
            instance_count += 1
            
            bbox = ann.get('bbox')
            if bbox:
                draw_rectangle(overlay, *bbox, color, thickness=2)
    
    alpha = 0.6
    mask_region = np.any(mask_layer > 0, axis=-1)
    overlay[mask_region] = (overlay[mask_region] * (1-alpha) + mask_layer[mask_region] * alpha).astype(np.uint8)
    
    return overlay, instance_count

def process_galaxy_orient(args):
    """Multiprocessing worker for a single galaxy_orient."""
    galaxy_orient, variants, images_map, img_to_anns, cat_id_to_info, thumb_size = args
    
    sorted_variants = sorted(variants.keys())
    
    cols = ['Original', 'Streams', 'Satellites', 'Combined']
    rows = len(sorted_variants)
    if rows == 0:
        return True
        
    gap = 5
    title_height = 40
    header_height = 30
    
    canvas_w = len(cols) * thumb_size + (len(cols) - 1) * gap + 40
    canvas_h = title_height + header_height + rows * thumb_size + (rows - 1) * gap + 20
    
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas = add_text_label(canvas, f"Galaxy: {galaxy_orient}", position=(20, 10), font_size=24)
    
    start_x = 20
    start_y = title_height
    
    # Draw headers
    for c_idx, col_name in enumerate(cols):
        px = start_x + c_idx * (thumb_size + gap)
        canvas = add_text_label(canvas, col_name, position=(px + 10, start_y + 5), font_size=18, color=(100, 200, 255))
        
    start_y += header_height
    
    for r_idx, variant_name in enumerate(sorted_variants):
        img_id = variants[variant_name]
        img_file = BASE_DIR / images_map[img_id]['file_name']
        
        if not img_file.exists():
            continue
            
        orig_img = np.array(Image.open(img_file))
        if len(orig_img.shape) == 2:
            orig_img = np.stack([orig_img]*3, axis=-1)
            
        anns = img_to_anns.get(img_id, [])
        
        for c_idx, col_name in enumerate(cols):
            px = start_x + c_idx * (thumb_size + gap)
            py = start_y + r_idx * (thumb_size + gap)
            
            if col_name == 'Original':
                tile_arr = orig_img.copy()
                inst_count = 0
            else:
                target_map = {'Streams': 'streams', 'Satellites': 'satellites', 'Combined': 'combined'}
                tile_arr, inst_count = create_mask_visualization(orig_img, anns, cat_id_to_info, target_map[col_name])
                
            pil_tile = Image.fromarray(tile_arr).resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            tile_vis = np.array(pil_tile)
            
            # Label
            tile_vis = add_text_label(tile_vis, variant_name, position=(5, 5), font_size=14)
            if inst_count > 0:
                tile_vis = add_text_label(tile_vis, f"N={inst_count}", position=(5, thumb_size-20), font_size=12, color=(255, 255, 0))
                
            canvas[py:py+thumb_size, px:px+thumb_size] = tile_vis

    out_path = OUTPUT_DIR / f"{galaxy_orient}_grid.jpg"
    Image.fromarray(canvas).save(str(out_path), quality=95)
    return True

def main():
    parser = argparse.ArgumentParser(description="Visualize SAM3 dataset - Grid Layout")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of galaxies")
    parser.add_argument("--galaxy", type=str, help="Filter by galaxy name/orient (e.g., 00011_eo)")
    parser.add_argument("--thumb_size", type=int, default=320, help="Thumbnail size")
    parser.add_argument("--num_proc", type=int, default=max(1, os.cpu_count() - 1), help="Number of parallel processes")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading annotations from {JSON_PATH}...")
    with open(JSON_PATH, 'r') as f:
        coco_data = json.load(f)

    images_map = {img['id']: img for img in coco_data['images']}
    cat_id_to_info = {cat['id']: cat for cat in coco_data['categories']}
    
    img_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    galaxy_orient_to_variants = defaultdict(dict)
    for img_id, img_info in images_map.items():
        file_name = img_info['file_name']
        base_name = Path(file_name).stem
        parts = base_name.split('_')
        if len(parts) >= 3:
            go = f"{parts[0]}_{parts[1]}"
            var = "_".join(parts[2:])
            galaxy_orient_to_variants[go][var] = img_id
        else:
            galaxy_orient_to_variants[base_name]["default"] = img_id

    all_go = list(galaxy_orient_to_variants.keys())
    if args.galaxy:
        all_go = [go for go in all_go if args.galaxy in go]
    if args.max:
        all_go = all_go[:args.max]

    if not all_go:
        print("No matching records found. Exit.")
        return

    print(f"Processing {len(all_go)} galaxy orientations using {args.num_proc} workers...")
    
    tasks = []
    for go in all_go:
        tasks.append((
            go, 
            galaxy_orient_to_variants[go], 
            images_map, 
            img_to_anns, 
            cat_id_to_info, 
            args.thumb_size
        ))
        
    if args.num_proc > 1:
        with mp.Pool(args.num_proc) as pool:
            pool.map(process_galaxy_orient, tasks)
    else:
        for t in tasks:
            process_galaxy_orient(t)

    print(f"Done! Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()