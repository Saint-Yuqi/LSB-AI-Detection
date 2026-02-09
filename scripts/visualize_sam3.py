#!/usr/bin/env python3
"""
SAM3 Dataset Visualization Script (16:9 Layout)

Creates a combined image per galaxy optimized for 16:9 screens.
Layout: 6 Columns grid.
- Streams section (2 rows)
- Satellites section (2 rows)
Center-aligned on a 16:9 black canvas.
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re
from collections import defaultdict
import math

# ================= Configuration =================
BASE_DIR = Path("/home/yuqyan/Yuqi/LSB-AI-Detection/data/02_processed/sam3_unified")
JSON_PATH = BASE_DIR / "annotations.json"
IMAGES_DIR = BASE_DIR / "images"
MASKS_DIR = BASE_DIR / "masks"
OUTPUT_DIR = BASE_DIR / "visualizations_16_9"

# Color palette for instances
INSTANCE_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 255, 128),    # Spring green
    (255, 0, 128),    # Rose
    (128, 255, 0),    # Lime
    (0, 128, 255),    # Sky blue
]

# SB thresholds to display
SB_THRESHOLDS = ['27.0', '27.5', '28.0', '28.5', '29.0', '29.5', '30.0', '30.5', '31.0', '31.5', '32.0']
# =================================================


def decode_rle_simple(counts, size):
    """Simple RLE decoder for COCO format."""
    h, w = size
    if isinstance(counts, str):
        m = 0
        p = 0
        decoded = []
        while p < len(counts):
            x = 0
            k = 0
            more = True
            while more:
                c = ord(counts[p]) - 48
                x |= (c & 0x1f) << (5 * k)
                more = c & 0x20
                p += 1
                k += 1
                if not more and (c & 0x10):
                    x |= (-1) << (5 * k)
            if m > 2:
                x += decoded[m - 2]
            decoded.append(x)
            m += 1
        counts = decoded
    
    if isinstance(counts, list):
        mask = np.zeros(h * w, dtype=np.uint8)
        pos = 0
        val = 0
        for count in counts:
            count = int(count)
            if pos + count > h * w:
                count = h * w - pos
            if count > 0:
                mask[pos:pos + count] = val
            pos += count
            val = 1 - val
        return mask.reshape((h, w), order='F')
    else:
        raise ValueError(f"Unknown counts format: {type(counts)}")


def normalize_16bit_to_8bit(img_16bit):
    if img_16bit.dtype == np.uint8:
        return img_16bit
    min_val, max_val = img_16bit.min(), img_16bit.max()
    if max_val == min_val:
        return np.zeros_like(img_16bit, dtype=np.uint8)
    return ((img_16bit - min_val) / (max_val - min_val) * 255).astype(np.uint8)


def draw_rectangle(img, x, y, w, h, color, thickness=2):
    x, y, w, h = int(x), int(y), int(w), int(h)
    img[max(0,y):min(img.shape[0],y+thickness), max(0,x):min(img.shape[1],x+w)] = color
    img[max(0,y+h-thickness):min(img.shape[0],y+h), max(0,x):min(img.shape[1],x+w)] = color
    img[max(0,y):min(img.shape[0],y+h), max(0,x):min(img.shape[1],x+thickness)] = color
    img[max(0,y):min(img.shape[0],y+h), max(0,x+w-thickness):min(img.shape[1],x+w)] = color


def parse_category_info(cat_name):
    if 'stream' in cat_name:
        simple_type = 'streams'
    elif 'satellite' in cat_name:
        simple_type = 'satellites'
    else:
        simple_type = 'unknown'
    
    match = re.search(r'SB\s*(\d+\.?\d*)', cat_name)
    if match:
        sb_str = match.group(1)
        if '.' not in sb_str:
            sb_str += ".0"
    else:
        sb_str = "0.0"
    
    return simple_type, sb_str


def get_clean_base_name(file_name):
    base = os.path.splitext(file_name)[0]
    base = base.replace("_streams", "").replace("_satellites", "")
    return base


def add_text_label(img_array, text, position=(10, 10), font_size=20, color=(255, 255, 255), bg_color=None):
    """
    Adds text to image.
    If bg_color is provided (e.g. (0,0,0)), draws a small background box behind text.
    """
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    try:
        # Try to use a better font if available
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    x, y = position
    
    # Calculate text size for background box
    if bg_color:
        bbox = draw.textbbox((x, y), text, font=font)
        # Add slight padding
        draw.rectangle((bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2), fill=bg_color)

    # Shadow/Outline for readability
    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
        draw.text((x+dx, y+dy), text, font=font, fill=(0, 0, 0))
    
    draw.text(position, text, font=font, fill=color)
    return np.array(img)


def create_mask_visualization(original_img, anns, cat_id_to_info, target_type, target_sb):
    h, w = original_img.shape[:2]
    overlay = original_img.copy()
    mask_layer = np.zeros((h, w, 3), dtype=np.uint8)
    
    instance_count = 0
    # 1. Draw Masks
    for ann in anns:
        cat = cat_id_to_info[ann['category_id']]
        simple_type, sb_str = parse_category_info(cat['name'])
        
        if simple_type != target_type or sb_str != target_sb:
            continue
        
        color = INSTANCE_COLORS[instance_count % len(INSTANCE_COLORS)]
        
        if 'segmentation' in ann and ann['segmentation']:
            seg = ann['segmentation']
            if isinstance(seg, dict) and 'counts' in seg:
                rle_size = seg.get('size', [h, w])
                mask_binary = decode_rle_simple(seg['counts'], rle_size)
                
                if np.sum(mask_binary > 0) > 0:
                    for c in range(3):
                        mask_layer[:, :, c][mask_binary > 0] = color[c]
                    instance_count += 1
    
    alpha = 0.6
    mask_region = np.any(mask_layer > 0, axis=-1)
    overlay[mask_region] = (overlay[mask_region] * (1-alpha) + mask_layer[mask_region] * alpha).astype(np.uint8)
    
    # 2. Draw Bboxes
    idx = 0
    for ann in anns:
        cat = cat_id_to_info[ann['category_id']]
        simple_type, sb_str = parse_category_info(cat['name'])
        if simple_type != target_type or sb_str != target_sb:
            continue
        bbox = ann['bbox']
        x, y, bw, bh = bbox
        color = INSTANCE_COLORS[idx % len(INSTANCE_COLORS)]
        draw_rectangle(overlay, x, y, bw, bh, color, thickness=2)
        idx += 1
    
    return overlay, instance_count


def generate_tile(img_type, label, original_img, anns, cat_id_to_info, sb_val=None, thumb_size=180):
    """Generates a single square tile with label overlay."""
    if sb_val is None:
        # Original Image Tile
        img_array = original_img.copy()
        inst_count = 0
    else:
        # Visualization Tile
        img_array, inst_count = create_mask_visualization(original_img, anns, cat_id_to_info, img_type, sb_val)

    pil_img = Image.fromarray(img_array)
    pil_thumb = pil_img.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
    vis_array = np.array(pil_thumb)

    # Add Labels
    # Top Left: Tile Name (e.g., SB27.0)
    vis_array = add_text_label(vis_array, label, position=(5, 5), font_size=14, color=(255, 255, 255))
    
    # Bottom Left: Instance Count (if > 0)
    if inst_count > 0:
        vis_array = add_text_label(vis_array, f"N={inst_count}", position=(5, thumb_size-20), font_size=12, color=(255, 255, 0))
        
    return vis_array


def main():
    parser = argparse.ArgumentParser(description="Visualize SAM3 dataset - 16:9 Grid Layout")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of galaxies")
    parser.add_argument("--galaxy", type=str, help="Filter by galaxy name")
    parser.add_argument("--thumb_size", type=int, default=240, help="Thumbnail size (larger for 16:9)")
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

    galaxy_to_images = defaultdict(dict)
    for img_id, img_info in images_map.items():
        file_name = img_info['file_name']
        base_name = get_clean_base_name(file_name)
        if '_streams' in file_name:
            galaxy_to_images[base_name]['streams'] = img_id
        elif '_satellites' in file_name:
            galaxy_to_images[base_name]['satellites'] = img_id

    all_galaxies = list(galaxy_to_images.keys())
    if args.galaxy:
        all_galaxies = [g for g in all_galaxies if args.galaxy in g]
    if args.max:
        all_galaxies = all_galaxies[:args.max]

    print(f"Processing {len(all_galaxies)} galaxies...")
    thumb = args.thumb_size
    
    # Layout Constants
    GRID_COLS = 6
    GRID_GAP = 5
    SECTION_HEADER_HEIGHT = 40
    TITLE_HEIGHT = 60
    
    for galaxy_name in all_galaxies:
        print(f"  Processing {galaxy_name}...")
        img_ids = galaxy_to_images[galaxy_name]
        
        # Load Originals
        original_images = {}
        for t in ['streams', 'satellites']:
            if t in img_ids:
                img_path = IMAGES_DIR / images_map[img_ids[t]]['file_name']
                if img_path.exists():
                    img = np.array(Image.open(img_path))
                    if img.dtype == np.uint16: img = normalize_16bit_to_8bit(img)
                    if len(img.shape) == 2: img = np.stack([img]*3, axis=-1)
                    original_images[t] = img[:,:,:3]

        # Prepare Tile Lists
        sections = [] # List of (Section Title, [Tile Arrays])
        
        for img_type in ['streams', 'satellites']:
            if img_type not in original_images:
                continue
            
            tiles = []
            orig_img = original_images[img_type]
            anns = img_to_anns[img_ids[img_type]]
            
            # 1. Original
            tiles.append(generate_tile(img_type, "Original", orig_img, anns, cat_id_to_info, None, thumb))
            
            # 2. Thresholds
            for sb in SB_THRESHOLDS:
                tiles.append(generate_tile(img_type, f"SB{sb}", orig_img, anns, cat_id_to_info, sb, thumb))
            
            sections.append((img_type.capitalize(), tiles))

        if not sections:
            continue

        # --- Calculate Dimensions for 16:9 Canvas ---
        
        # Calculate Content Height
        content_width = GRID_COLS * thumb + (GRID_COLS - 1) * GRID_GAP
        current_y = TITLE_HEIGHT
        
        for _, tiles in sections:
            rows = math.ceil(len(tiles) / GRID_COLS)
            section_h = SECTION_HEADER_HEIGHT + rows * thumb + (rows - 1) * GRID_GAP
            current_y += section_h + 20 # 20px padding between sections
        
        total_content_height = current_y
        
        # Determine 16:9 Dimensions
        # Target Width based on 16:9 ratio
        target_aspect = 16 / 9
        calculated_width = int(total_content_height * target_aspect)
        
        # Ensure canvas is at least as wide as the grid content
        final_width = max(calculated_width, content_width + 40) # 40px side padding minimum
        final_height = total_content_height
        
        # If the grid is wider than 16:9 (unlikely with 6 cols), adjust height to maintain ratio
        if final_width < content_width + 40:
             final_width = content_width + 40
             final_height = int(final_width / target_aspect)

        # Create Black Canvas
        canvas = np.zeros((final_height, final_width, 3), dtype=np.uint8)
        
        # Draw Title
        canvas = add_text_label(canvas, f"Galaxy: {galaxy_name}", 
                                position=(final_width//2 - 100, 15), font_size=24, color=(255, 255, 255))

        # Draw Grid
        start_x = (final_width - content_width) // 2
        cursor_y = TITLE_HEIGHT
        
        for section_title, tiles in sections:
            # Draw Section Header
            canvas = add_text_label(canvas, section_title, 
                                    position=(start_x, cursor_y + 10), font_size=18, color=(100, 200, 255))
            cursor_y += SECTION_HEADER_HEIGHT
            
            # Draw Tiles
            for i, tile in enumerate(tiles):
                r = i // GRID_COLS
                c = i % GRID_COLS
                
                px = start_x + c * (thumb + GRID_GAP)
                py = cursor_y + r * (thumb + GRID_GAP)
                
                canvas[py:py+thumb, px:px+thumb] = tile
            
            # Update cursor for next section
            rows = math.ceil(len(tiles) / GRID_COLS)
            cursor_y += rows * (thumb + GRID_GAP) + 20

        # Save
        out_path = OUTPUT_DIR / f"{galaxy_name}_16_9.jpg"
        Image.fromarray(canvas).save(str(out_path), quality=95)

    print(f"\nDone! Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()