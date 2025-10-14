#!/usr/bin/env python3
"""
Full Dataset999 WSI Cellularity Analysis (20 WSIs)
Using nnUNet Model (Dataset998, Epoch 52)
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import json
from datetime import datetime
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# Environment variables setup
os.environ['nnUNet_raw'] = '/home/baegjaehyeon/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/home/baegjaehyeon/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/home/baegjaehyeon/nnUNet_results'


def extract_tiles_with_overlap(image, tile_size=1280, overlap=128):
    """Extract tiles from WSI with overlap"""
    h, w = image.shape[:2]
    stride = tile_size - overlap
    tiles = []
    
    for y in range(0, h, stride):
        if y + tile_size > h:
            y = max(0, h - tile_size)
            
        for x in range(0, w, stride):
            if x + tile_size > w:
                x = max(0, w - tile_size)
                
            tile = image[y:y+tile_size, x:x+tile_size]
            
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                tiles.append({
                    'tile': tile,
                    'position': (x, y),
                    'coords': (x, y, x+tile_size, y+tile_size)
                })
            
            if x + tile_size >= w:
                break
                
        if y + tile_size >= h:
            break
            
    return tiles


def save_tiles_for_inference(tiles, output_dir):
    """Save tiles in nnUNet format for inference"""
    os.makedirs(output_dir, exist_ok=True)
    
    tile_info = []
    for i, tile_data in enumerate(tiles):
        tile = tile_data['tile']
        
        # Grayscale 변환
        if len(tile.shape) == 3:
            tile_gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        else:
            tile_gray = tile
            
        # nnUNet 형식: _0000.png
        filename = f"tile_{i:04d}_0000.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, tile_gray)
        
        tile_info.append({
            'index': i,
            'filename': filename,
            'position': tile_data['position'],
            'coords': tile_data['coords']
        })
    
    return tile_info


def run_nnunet_inference(input_dir, output_dir):
    """Run nnUNet model inference"""
    import subprocess

    # Check checkpoint
    checkpoint_path = "/home/baegjaehyeon/nnUNet_results/Dataset998_BC_1280/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_final.pth"
    if not os.path.exists(checkpoint_path):
        latest_path = "/home/baegjaehyeon/nnUNet_results/Dataset998_BC_1280/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_latest.pth"
        if os.path.exists(latest_path):
            import shutil
            shutil.copy(latest_path, checkpoint_path)
    
    # Inference command
    cmd = [
        "/home/baegjaehyeon/nnunet_env/bin/nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-d", "998",
        "-c", "2d",
        "-f", "0",
        "--disable_tta"
    ]
    
    env = os.environ.copy()
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        pred_files = glob(os.path.join(output_dir, "*.png"))
        return len(pred_files) > 0
    except Exception as e:
        print(f"❌ Inference error: {e}")
        return False


def merge_predictions(predictions_dir, tile_info, wsi_shape):
    """Merge tile predictions into full WSI"""
    h, w = wsi_shape
    
    global_cell_mask = np.zeros((h, w), dtype=np.float32)
    overlap_count = np.zeros((h, w), dtype=np.int32)
    
    for tile_data in tile_info:
        tile_idx = tile_data['index']
        pred_file = f"tile_{tile_idx:04d}.png"
        pred_path = os.path.join(predictions_dir, pred_file)
        
        if not os.path.exists(pred_path):
            continue
            
        pred_mask = np.array(Image.open(pred_path)).astype(np.float32)
        x1, y1, x2, y2 = tile_data['coords']
        
        global_cell_mask[y1:y2, x1:x2] += pred_mask
        overlap_count[y1:y2, x1:x2] += 1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        global_cell_mask = np.divide(global_cell_mask, overlap_count, 
                                    where=overlap_count > 0)
    
    global_cell_mask = (global_cell_mask > 0.5).astype(np.uint8)
    
    return global_cell_mask


def analyze_cellularity(wsi_image, cell_mask):
    """Hierarchical cellularity analysis"""

    if len(wsi_image.shape) == 3:
        gray = cv2.cvtColor(wsi_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = wsi_image

    cell_pixels = gray[cell_mask > 0]

    if len(cell_pixels) == 0:
        return None, {}

    # Adipocyte detection (top 25% brightness within cellular regions)
    threshold = np.percentile(cell_pixels, 75)
    threshold = max(180, min(240, threshold))
    
    adipocyte_mask = np.logical_and(gray > threshold, cell_mask > 0).astype(np.uint8)
    
    total_cell = np.sum(cell_mask > 0)
    total_adipocyte = np.sum(adipocyte_mask > 0)
    total_normal = total_cell - total_adipocyte
    
    cellularity = total_normal / total_cell if total_cell > 0 else 0.0
    
    stats = {
        'global_cellularity': float(cellularity),
        'total_cell_pixels': int(total_cell),
        'total_adipocyte_pixels': int(total_adipocyte),
        'total_normal_pixels': int(total_normal),
        'adipocyte_ratio': float(total_adipocyte / total_cell) if total_cell > 0 else 0,
        'adipocyte_threshold': int(threshold),
        'cell_coverage': float(total_cell / (wsi_image.shape[0] * wsi_image.shape[1]))
    }
    
    return adipocyte_mask, stats


def create_annotated_visualization(wsi_image, cell_mask, adipocyte_mask, stats, output_path):
    """Create visualization with statistical annotations"""

    if len(wsi_image.shape) == 2:
        wsi_rgb = cv2.cvtColor(wsi_image, cv2.COLOR_GRAY2RGB)
    else:
        wsi_rgb = wsi_image.copy()

    # Distinguish normal cells and adipocytes
    normal_mask = np.logical_and(cell_mask > 0, adipocyte_mask == 0)

    # Create overlay
    overlay = wsi_rgb.copy()
    overlay[normal_mask > 0] = [0, 255, 0]  # Normal cells: green
    overlay[adipocyte_mask > 0] = [255, 255, 0]  # Adipocytes: yellow

    # Blending
    result = cv2.addWeighted(wsi_rgb, 0.5, overlay, 0.5, 0)
    
    # Resize
    h_vis, w_vis = result.shape[:2]
    target_width = 2000
    if w_vis > target_width:
        scale = target_width / w_vis
        new_h, new_w = int(h_vis * scale), int(w_vis * scale)
        result = cv2.resize(result, (new_w, new_h))

    # Add statistical information
    h_final, w_final = result.shape[:2]
    info_height = 150
    canvas = np.ones((h_final + info_height, w_final, 3), dtype=np.uint8) * 255
    canvas[:h_final, :] = result
    
    # Add text with PIL
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Text information
    text_y = h_final + 20
    text_lines = [
        f"Cellularity: {stats['global_cellularity']:.1%}",
        f"Cell Coverage: {stats['cell_coverage']:.1%}",
        f"Adipocyte Ratio: {stats['adipocyte_ratio']:.1%}",
        f"Normal Cells: {stats['total_normal_pixels']:,} pixels",
        f"Adipocyte Cells: {stats['total_adipocyte_pixels']:,} pixels"
    ]
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for i, line in enumerate(text_lines):
        draw.text((20, text_y + i*25), line, fill=(0, 0, 0), font=font)
    
    # Legend
    legend_x = w_final - 200
    draw.rectangle([legend_x, text_y, legend_x+20, text_y+20], fill=(0, 255, 0))
    draw.text((legend_x+25, text_y), "Normal Cells", fill=(0, 0, 0), font=font)
    draw.rectangle([legend_x, text_y+30, legend_x+20, text_y+50], fill=(255, 255, 0))
    draw.text((legend_x+25, text_y+30), "Adipocytes", fill=(0, 0, 0), font=font)

    # Save
    final_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, final_img)
    
    return final_img


def process_single_wsi(wsi_name, wsi_path, base_output_dir):
    """Process a single WSI"""
    print(f"\n{'='*60}")
    print(f"🔬 Processing {wsi_name}")
    print(f"{'='*60}")
    
    # Output directories
    wsi_dir = os.path.join(base_output_dir, wsi_name)
    tiles_dir = os.path.join(wsi_dir, "tiles")
    predictions_dir = os.path.join(wsi_dir, "predictions")

    os.makedirs(wsi_dir, exist_ok=True)

    # Load WSI
    wsi_image = np.array(Image.open(wsi_path))
    h, w = wsi_image.shape[:2]
    print(f"📐 Dimensions: {h} × {w}")
    
    # Extract tiles
    tile_size = 1280
    overlap = 128
    tiles = extract_tiles_with_overlap(wsi_image, tile_size, overlap)
    print(f"📦 Extracted {len(tiles)} tiles")

    # Save tiles
    tile_info = save_tiles_for_inference(tiles, tiles_dir)

    # Run inference
    print(f"🧠 Running nnUNet inference...")
    success = run_nnunet_inference(tiles_dir, predictions_dir)
    
    if not success:
        print(f"❌ Inference failed for {wsi_name}")
        return None
    
    # Merge predictions
    cell_mask = merge_predictions(predictions_dir, tile_info, (h, w))

    # Cellularity analysis
    adipocyte_mask, stats = analyze_cellularity(wsi_image, cell_mask)
    
    if stats:
        stats['wsi_name'] = wsi_name
        stats['wsi_dimensions'] = [h, w]
        stats['num_tiles'] = len(tiles)
        
        # Save results
        with open(os.path.join(wsi_dir, "cellularity_analysis.json"), 'w') as f:
            json.dump(stats, f, indent=2)

        # Create visualization (with statistics)
        vis_path = os.path.join(wsi_dir, "cellularity_visualization.png")
        create_annotated_visualization(wsi_image, cell_mask, adipocyte_mask, stats, vis_path)
        
        # Save masks
        Image.fromarray(cell_mask * 255).save(os.path.join(wsi_dir, "cell_mask.png"))
        Image.fromarray(adipocyte_mask * 255).save(os.path.join(wsi_dir, "adipocyte_mask.png"))

        print(f"✅ {wsi_name}: Cellularity {stats['global_cellularity']:.1%}")

        # Remove tiles and predictions (save space)
        import shutil
        shutil.rmtree(tiles_dir)
        shutil.rmtree(predictions_dir)
        
        return stats
    
    return None


def create_summary_plot(all_results, output_path):
    """Create summary plot for all results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Prepare data
    wsi_names = [r['wsi_name'] for r in all_results]
    cellularities = [r['global_cellularity'] * 100 for r in all_results]
    coverages = [r['cell_coverage'] * 100 for r in all_results]
    adipocyte_ratios = [r['adipocyte_ratio'] * 100 for r in all_results]
    
    # 1. Cellularity distribution
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(wsi_names)), cellularities, color='steelblue')
    ax1.set_xlabel('WSI')
    ax1.set_ylabel('Cellularity (%)')
    ax1.set_title('Cellularity Distribution Across WSIs')
    ax1.set_xticks(range(len(wsi_names)))
    ax1.set_xticklabels([name.replace('BC_', '') for name in wsi_names], rotation=45)
    ax1.grid(axis='y', alpha=0.3)

    # Display values
    for bar, val in zip(bars1, cellularities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Cell Coverage
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(wsi_names)), coverages, color='green')
    ax2.set_xlabel('WSI')
    ax2.set_ylabel('Cell Coverage (%)')
    ax2.set_title('Cell Coverage Distribution')
    ax2.set_xticks(range(len(wsi_names)))
    ax2.set_xticklabels([name.replace('BC_', '') for name in wsi_names], rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Cellularity vs Adipocyte Ratio (Stacked Bar)
    ax3 = axes[1, 0]
    normal_ratios = [100 - ratio for ratio in adipocyte_ratios]
    
    bars3_1 = ax3.bar(range(len(wsi_names)), normal_ratios, color='green', label='Normal Cells')
    bars3_2 = ax3.bar(range(len(wsi_names)), adipocyte_ratios, bottom=normal_ratios, 
                     color='yellow', label='Adipocytes')
    
    ax3.set_xlabel('WSI')
    ax3.set_ylabel('Cell Composition (%)')
    ax3.set_title('Cell Type Distribution Within Detected Cells')
    ax3.set_xticks(range(len(wsi_names)))
    ax3.set_xticklabels([name.replace('BC_', '') for name in wsi_names], rotation=45)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Statistical summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""Dataset999 Analysis Summary (n=20)
    
Cellularity:
  • Mean: {np.mean(cellularities):.1f}%
  • Median: {np.median(cellularities):.1f}%
  • Std Dev: {np.std(cellularities):.1f}%
  • Range: {min(cellularities):.1f}% - {max(cellularities):.1f}%
  
Cell Coverage:
  • Mean: {np.mean(coverages):.1f}%
  • Median: {np.median(coverages):.1f}%
  • Range: {min(coverages):.1f}% - {max(coverages):.1f}%
  
Adipocyte Ratio:
  • Mean: {np.mean(adipocyte_ratios):.1f}%
  • Median: {np.median(adipocyte_ratios):.1f}%
  • Range: {min(adipocyte_ratios):.1f}% - {max(adipocyte_ratios):.1f}%
  
Model: nnUNet Dataset998 (Epoch 52)
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, va='center', 
            family='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("\n" + "="*70)
    print("🚀 Dataset999 Full Analysis (20 WSIs)")
    print("="*70)
    
    # Base directory
    base_output_dir = "/home/baegjaehyeon/Dataset999_Full_Analysis"
    os.makedirs(base_output_dir, exist_ok=True)

    # Get WSI list
    dataset_dir = "/home/baegjaehyeon/nnUNet_raw/Dataset999_BC_Segmentation/imagesTr"
    wsi_files = sorted(glob(os.path.join(dataset_dir, "*.jpg")))
    
    print(f"\n📂 Found {len(wsi_files)} WSI files")
    
    # Process all WSIs
    all_results = []

    for i, wsi_path in enumerate(wsi_files, 1):
        wsi_name = os.path.basename(wsi_path).replace("_0000.jpg", "")
        print(f"\n[{i}/{len(wsi_files)}] Processing {wsi_name}")

        stats = process_single_wsi(wsi_name, wsi_path, base_output_dir)

        if stats:
            all_results.append(stats)

    # Save all results
    with open(os.path.join(base_output_dir, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary plot
    if all_results:
        summary_plot_path = os.path.join(base_output_dir, "summary_analysis.png")
        create_summary_plot(all_results, summary_plot_path)

    # Create results summary table
    print("\n" + "="*70)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"{'WSI':<10} {'Cellularity':>12} {'Coverage':>10} {'Adipocyte':>10}")
    print("-"*70)
    
    for r in sorted(all_results, key=lambda x: x['global_cellularity'], reverse=True):
        print(f"{r['wsi_name'].replace('BC_', ''):<10} {r['global_cellularity']*100:>11.1f}% "
              f"{r['cell_coverage']*100:>9.1f}% {r['adipocyte_ratio']*100:>9.1f}%")
    
    print("="*70)
    print(f"\n✅ Analysis complete! Results saved to: {base_output_dir}")


if __name__ == "__main__":
    main()