#!/usr/bin/env python3
"""
SAM2 Dataset Visualization Script (16:9 Layout)

Creates a combined image per galaxy for SAM2 folder-based format.
Layout: 6 Columns grid showing Original + SB thresholds.
- Streams section (2 rows)
- Satellites section (2 rows)
Center-aligned on a 16:9 black canvas.

SAM2 folder structure:
  img_folder/{sample}/0000.png
  gt_folder/{sample}/0000.png
  
Sample naming: {galaxy_id:05d}_{orientation}_SB{threshold}_{type}
"""

import argparse
import os
import re
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import math

# ================= Configuration =================
BASE_DIR = Path("/home/yuqyan/Yuqi/LSB-AI-Detection/data/02_processed/sam2_prepared")
IMG_FOLDER = BASE_DIR / "img_folder"
GT_FOLDER = BASE_DIR / "gt_folder"
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
SB_THRESHOLDS = ['27', '27.5', '28', '28.5', '29', '29.5', '30', '30.5', '31', '31.5', '32']
# =================================================


def parse_folder_name(folder_name: str) -> dict:
    """
    Parse SAM2 folder name like '00011_eo_SB27.5_streams'.
    
    Returns dict with:
        galaxy_id: int
        orientation: str
        sb_threshold: str
        feature_type: str
    """
    pattern = r'^(\d+)_([a-z]+)_SB([\d.]+)_(\w+)$'
    match = re.match(pattern, folder_name)
    if not match:
        return None
    
    return {
        'galaxy_id': int(match.group(1)),
        'orientation': match.group(2),
        'sb_threshold': match.group(3),
        'feature_type': match.group(4)
    }


def add_text_label(img_array, text, position=(10, 10), font_size=20, color=(255, 255, 255), bg_color=None):
    """Adds text to image with optional background box."""
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

    # Shadow for readability
    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
        draw.text((x+dx, y+dy), text, font=font, fill=(0, 0, 0))
    
    draw.text(position, text, font=font, fill=color)
    return np.array(img)


def create_mask_visualization(original_img: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Overlay mask on original image with instance-colored visualization.
    
    Returns:
        (overlay_image, instance_count)
    """
    h, w = original_img.shape[:2]
    overlay = original_img.copy()
    mask_layer = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Get unique instance IDs
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]  # Skip background
    
    instance_count = len(instance_ids)
    
    # Color each instance
    for idx, inst_id in enumerate(instance_ids):
        color = INSTANCE_COLORS[idx % len(INSTANCE_COLORS)]
        inst_mask = mask == inst_id
        for c in range(3):
            mask_layer[:, :, c][inst_mask] = color[c]
    
    # Blend overlay
    alpha = 0.6
    mask_region = np.any(mask_layer > 0, axis=-1)
    overlay[mask_region] = (overlay[mask_region] * (1-alpha) + mask_layer[mask_region] * alpha).astype(np.uint8)
    
    return overlay, instance_count


def generate_tile(img_path: Path, mask_path: Path, label: str, thumb_size: int = 240) -> np.ndarray:
    """
    Generate a single square tile showing image with mask overlay.
    
    Returns:
        RGB numpy array of size (thumb_size, thumb_size, 3)
    """
    # Check if image exists
    if not img_path.exists():
        # Return black tile with label
        tile = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
        tile = add_text_label(tile, "No Image", position=(5, thumb_size//2), font_size=12, color=(128, 128, 128))
        tile = add_text_label(tile, label, position=(5, 5), font_size=14, color=(255, 255, 255))
        return tile
    
    # Load image
    img = np.array(Image.open(img_path))
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]
    
    # Check if this is original (no mask) or visualization
    if mask_path is None:
        vis_array = img.copy()
        inst_count = 0
    elif mask_path.exists():
        mask = np.array(Image.open(mask_path))
        vis_array, inst_count = create_mask_visualization(img, mask)
    else:
        vis_array = img.copy()
        inst_count = 0
    
    # Resize to thumbnail
    pil_thumb = Image.fromarray(vis_array).resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
    vis_array = np.array(pil_thumb)
    
    # Add Labels
    vis_array = add_text_label(vis_array, label, position=(5, 5), font_size=14, color=(255, 255, 255))
    
    if inst_count > 0:
        vis_array = add_text_label(vis_array, f"N={inst_count}", position=(5, thumb_size-20), font_size=12, color=(255, 255, 0))
    
    return vis_array


def main():
    parser = argparse.ArgumentParser(description="Visualize SAM2 dataset - 16:9 Grid Layout")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of galaxies")
    parser.add_argument("--galaxy", type=str, help="Filter by galaxy ID (e.g., '11' or '00011')")
    parser.add_argument("--thumb_size", type=int, default=240, help="Thumbnail size")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    print(f"Scanning SAM2 dataset at {BASE_DIR}...")
    
    # Scan all sample folders
    samples = {}
    if IMG_FOLDER.exists():
        for folder in IMG_FOLDER.iterdir():
            if folder.is_dir():
                parsed = parse_folder_name(folder.name)
                if parsed:
                    key = (parsed['galaxy_id'], parsed['orientation'])
                    if key not in samples:
                        samples[key] = {'streams': {}, 'satellites': {}}
                    samples[key][parsed['feature_type']][parsed['sb_threshold']] = folder.name
    
    print(f"Found {len(samples)} galaxy-orientation pairs")
    
    # Group by galaxy (combine eo and fo)
    galaxies = defaultdict(dict)
    for (gal_id, orient), data in samples.items():
        gal_name = f"Fbox-{gal_id}-{orient}"
        galaxies[gal_name] = {
            'galaxy_id': gal_id,
            'orientation': orient,
            'data': data
        }
    
    all_galaxies = sorted(galaxies.keys())
    
    if args.galaxy:
        filter_id = args.galaxy.lstrip('0') if args.galaxy.isdigit() else args.galaxy
        all_galaxies = [g for g in all_galaxies if filter_id in g]
    
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
        info = galaxies[galaxy_name]
        gal_id = info['galaxy_id']
        orient = info['orientation']
        data = info['data']
        
        sections = []
        
        for feature_type in ['streams', 'satellites']:
            if feature_type not in data or not data[feature_type]:
                continue
            
            tiles = []
            sb_data = data[feature_type]
            
            # Find a valid folder to get the original image
            first_sb = list(sb_data.keys())[0] if sb_data else None
            if first_sb:
                first_folder = sb_data[first_sb]
                orig_img_path = IMG_FOLDER / first_folder / "0000.png"
                tiles.append(generate_tile(orig_img_path, None, "Original", thumb))
            
            # Generate tiles for each SB threshold
            for sb in SB_THRESHOLDS:
                if sb in sb_data:
                    folder_name = sb_data[sb]
                    img_path = IMG_FOLDER / folder_name / "0000.png"
                    mask_path = GT_FOLDER / folder_name / "0000.png"
                    tiles.append(generate_tile(img_path, mask_path, f"SB{sb}", thumb))
                else:
                    # Create placeholder for missing threshold
                    tile = np.zeros((thumb, thumb, 3), dtype=np.uint8)
                    tile = add_text_label(tile, f"SB{sb}", position=(5, 5), font_size=14, color=(100, 100, 100))
                    tile = add_text_label(tile, "No Data", position=(5, thumb//2), font_size=12, color=(80, 80, 80))
                    tiles.append(tile)
            
            sections.append((feature_type.capitalize(), tiles))
        
        if not sections:
            continue
        
        # --- Calculate Dimensions for 16:9 Canvas ---
        content_width = GRID_COLS * thumb + (GRID_COLS - 1) * GRID_GAP
        current_y = TITLE_HEIGHT
        
        for _, tiles in sections:
            rows = math.ceil(len(tiles) / GRID_COLS)
            section_h = SECTION_HEADER_HEIGHT + rows * thumb + (rows - 1) * GRID_GAP
            current_y += section_h + 20
        
        total_content_height = current_y
        
        target_aspect = 16 / 9
        calculated_width = int(total_content_height * target_aspect)
        final_width = max(calculated_width, content_width + 40)
        final_height = total_content_height
        
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
            canvas = add_text_label(canvas, section_title, 
                                    position=(start_x, cursor_y + 10), font_size=18, color=(100, 200, 255))
            cursor_y += SECTION_HEADER_HEIGHT
            
            for i, tile in enumerate(tiles):
                r = i // GRID_COLS
                c = i % GRID_COLS
                
                px = start_x + c * (thumb + GRID_GAP)
                py = cursor_y + r * (thumb + GRID_GAP)
                
                canvas[py:py+thumb, px:px+thumb] = tile
            
            rows = math.ceil(len(tiles) / GRID_COLS)
            cursor_y += rows * (thumb + GRID_GAP) + 20
        
        # Save
        out_path = OUTPUT_DIR / f"{galaxy_name}_16_9.jpg"
        Image.fromarray(canvas).save(str(out_path), quality=95)
    
    print(f"\nDone! Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
