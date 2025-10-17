#!/usr/bin/env python3
"""
Cellularity Comparison: Model Prediction vs Ground Truth
Using Binary Classification (Hematopoietic Cells vs Adipocytes)

This script:
1. Categorizes 6 samples by cellularity (High/Medium/Low)
2. Applies binary classification to BOTH prediction and GT masks
3. Compares cellularity measurements between prediction and GT
4. Calculates confidence intervals and statistical analysis
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
from scipy import stats as scipy_stats
import argparse

# Environment variables setup for GCP
os.environ['nnUNet_raw'] = '/home/baegjaehyeon/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/home/baegjaehyeon/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/home/baegjaehyeon/nnUNet_results'


# ============================================================================
# Sample Categorization
# ============================================================================

SAMPLE_CATEGORIES = {
    'high': {
        'samples': ['BP25', 'BP31'],
        'description': 'High cellularity (>30% GT coverage)'
    },
    'medium': {
        'samples': ['BP1', 'BP7'],
        'description': 'Medium cellularity (25-35% GT coverage)'
    },
    'low': {
        'samples': ['H2', 'H6'],
        'description': 'Low cellularity (<25% GT coverage)'
    }
}


# ============================================================================
# Tile Extraction and Model Inference
# ============================================================================

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
    """Run nnUNet model inference on GCP"""
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
    """Merge tile predictions into full WSI mask"""
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


def load_ground_truth_mask(gt_path):
    """Load and binarize ground truth mask from human annotation"""
    gt_img = Image.open(gt_path)
    gt_array = np.array(gt_img)

    if len(gt_array.shape) == 3:
        gt_array = cv2.cvtColor(gt_array, cv2.COLOR_RGB2GRAY)

    binary_gt = (gt_array > 0).astype(np.uint8)

    return binary_gt


# ============================================================================
# Binary Classification: Hematopoietic vs Adipocyte
# ============================================================================

def apply_binary_classification(wsi_image, cell_mask):
    """
    Apply binary classification within cell regions

    Args:
        wsi_image: Original WSI image
        cell_mask: Binary mask (1: cell region, 0: background)

    Returns:
        adipocyte_mask: Binary mask for adipocytes
        cellularity_stats: Dictionary with cellularity measurements
    """

    # Grayscale 변환
    if len(wsi_image.shape) == 3:
        gray = cv2.cvtColor(wsi_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = wsi_image

    # 세포 영역 내 픽셀만 분석
    cell_pixels = gray[cell_mask > 0]

    if len(cell_pixels) == 0:
        return None, None

    # Adipocyte detection (top 25% brightness within cellular regions)
    # 밝은 픽셀 = 지방세포, 어두운 픽셀 = 조혈세포
    threshold = np.percentile(cell_pixels, 75)
    threshold = max(180, min(240, threshold))  # 180-240 범위로 제한

    # 세포 영역 내에서만 지방세포 검출
    adipocyte_mask = np.logical_and(gray > threshold, cell_mask > 0).astype(np.uint8)

    # 통계 계산
    total_cell_pixels = np.sum(cell_mask > 0)
    total_adipocyte_pixels = np.sum(adipocyte_mask > 0)
    total_hematopoietic_pixels = total_cell_pixels - total_adipocyte_pixels

    # Cellularity = 조혈세포 비율
    cellularity = total_hematopoietic_pixels / total_cell_pixels if total_cell_pixels > 0 else 0.0

    cellularity_stats = {
        'cellularity_percentage': float(cellularity * 100),
        'total_cell_pixels': int(total_cell_pixels),
        'hematopoietic_pixels': int(total_hematopoietic_pixels),
        'adipocyte_pixels': int(total_adipocyte_pixels),
        'adipocyte_ratio': float(total_adipocyte_pixels / total_cell_pixels) if total_cell_pixels > 0 else 0,
        'adipocyte_threshold': int(threshold),
        'cell_coverage_percentage': float(total_cell_pixels / (gray.shape[0] * gray.shape[1]) * 100)
    }

    return adipocyte_mask, cellularity_stats


# ============================================================================
# Comparison Analysis
# ============================================================================

def compare_cellularity(pred_stats, gt_stats, sample_name):
    """
    Compare cellularity between prediction and ground truth

    Returns:
        comparison_metrics: Dictionary with comparison statistics
    """

    pred_cellularity = pred_stats['cellularity_percentage']
    gt_cellularity = gt_stats['cellularity_percentage']

    # 절대 차이 및 상대 차이
    absolute_diff = pred_cellularity - gt_cellularity
    relative_diff = (absolute_diff / gt_cellularity * 100) if gt_cellularity > 0 else 0

    comparison = {
        'sample_name': sample_name,
        'prediction_cellularity': pred_cellularity,
        'gt_cellularity': gt_cellularity,
        'absolute_difference': absolute_diff,
        'relative_difference_percent': relative_diff,
        'prediction_stats': pred_stats,
        'gt_stats': gt_stats
    }

    return comparison


# ============================================================================
# Visualization
# ============================================================================

def create_comparison_visualization(original_img, pred_mask, gt_mask,
                                   pred_adipocyte, gt_adipocyte,
                                   comparison, output_path):
    """
    Create 6-panel comparison visualization

    Layout:
    [Original] [GT Mask] [Pred Mask]
    [GT Cellularity] [Pred Cellularity] [Statistics]
    """

    # Ensure RGB
    if len(original_img.shape) == 2:
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original_img.copy()

    # Create GT cellularity overlay
    gt_normal = np.logical_and(gt_mask > 0, gt_adipocyte == 0)
    gt_viz = original_rgb.copy()
    gt_overlay = original_rgb.copy()
    gt_overlay[gt_normal > 0] = [0, 255, 0]  # Hematopoietic: green
    gt_overlay[gt_adipocyte > 0] = [255, 255, 0]  # Adipocyte: yellow
    gt_viz = cv2.addWeighted(original_rgb, 0.5, gt_overlay, 0.5, 0)

    # Create Prediction cellularity overlay
    pred_normal = np.logical_and(pred_mask > 0, pred_adipocyte == 0)
    pred_viz = original_rgb.copy()
    pred_overlay = original_rgb.copy()
    pred_overlay[pred_normal > 0] = [0, 255, 0]  # Hematopoietic: green
    pred_overlay[pred_adipocyte > 0] = [255, 255, 0]  # Adipocyte: yellow
    pred_viz = cv2.addWeighted(original_rgb, 0.5, pred_overlay, 0.5, 0)

    # Create GT mask visualization
    gt_mask_viz = original_rgb.copy()
    gt_mask_overlay = original_rgb.copy()
    gt_mask_overlay[gt_mask > 0] = [0, 255, 0]
    gt_mask_viz = cv2.addWeighted(original_rgb, 0.6, gt_mask_overlay, 0.4, 0)

    # Create Pred mask visualization
    pred_mask_viz = original_rgb.copy()
    pred_mask_overlay = original_rgb.copy()
    pred_mask_overlay[pred_mask > 0] = [0, 0, 255]
    pred_mask_viz = cv2.addWeighted(original_rgb, 0.6, pred_mask_overlay, 0.4, 0)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    # Row 1: Original, GT Mask, Pred Mask
    axes[0, 0].imshow(cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original WSI', fontsize=16, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(gt_mask_viz, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Ground Truth Mask', fontsize=16, fontweight='bold')
    axes[0, 1].axis('off')
    gt_patch = mpatches.Patch(color='green', label='GT Cell Region')
    axes[0, 1].legend(handles=[gt_patch], loc='upper right', fontsize=12)

    axes[0, 2].imshow(cv2.cvtColor(pred_mask_viz, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Model Prediction Mask', fontsize=16, fontweight='bold')
    axes[0, 2].axis('off')
    pred_patch = mpatches.Patch(color='blue', label='Predicted Cell Region')
    axes[0, 2].legend(handles=[pred_patch], loc='upper right', fontsize=12)

    # Row 2: GT Cellularity, Pred Cellularity, Statistics
    axes[1, 0].imshow(cv2.cvtColor(gt_viz, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'GT Cellularity: {comparison["gt_cellularity"]:.1f}%',
                         fontsize=16, fontweight='bold')
    axes[1, 0].axis('off')
    gt_h_patch = mpatches.Patch(color='green', label='Hematopoietic')
    gt_a_patch = mpatches.Patch(color='yellow', label='Adipocyte')
    axes[1, 0].legend(handles=[gt_h_patch, gt_a_patch], loc='upper right', fontsize=12)

    axes[1, 1].imshow(cv2.cvtColor(pred_viz, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Pred Cellularity: {comparison["prediction_cellularity"]:.1f}%',
                         fontsize=16, fontweight='bold')
    axes[1, 1].axis('off')
    pred_h_patch = mpatches.Patch(color='green', label='Hematopoietic')
    pred_a_patch = mpatches.Patch(color='yellow', label='Adipocyte')
    axes[1, 1].legend(handles=[pred_h_patch, pred_a_patch], loc='upper right', fontsize=12)

    # Statistics panel
    axes[1, 2].axis('off')

    stats_text = f"""Cellularity Comparison Statistics

Sample: {comparison['sample_name']}

Cellularity (Hematopoietic Cell Ratio):
  • GT:         {comparison['gt_cellularity']:.2f}%
  • Prediction: {comparison['prediction_cellularity']:.2f}%
  • Difference: {comparison['absolute_difference']:+.2f}%
  • Relative:   {comparison['relative_difference_percent']:+.2f}%

Ground Truth:
  • Total cells:       {comparison['gt_stats']['total_cell_pixels']:,}
  • Hematopoietic:     {comparison['gt_stats']['hematopoietic_pixels']:,}
  • Adipocyte:         {comparison['gt_stats']['adipocyte_pixels']:,}
  • Coverage:          {comparison['gt_stats']['cell_coverage_percentage']:.2f}%

Prediction:
  • Total cells:       {comparison['prediction_stats']['total_cell_pixels']:,}
  • Hematopoietic:     {comparison['prediction_stats']['hematopoietic_pixels']:,}
  • Adipocyte:         {comparison['prediction_stats']['adipocyte_pixels']:,}
  • Coverage:          {comparison['prediction_stats']['cell_coverage_percentage']:.2f}%

Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""

    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=11, va='center',
                   family='monospace',
                   bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Processing Pipeline
# ============================================================================

def process_single_sample(original_path, gt_path, output_dir, sample_name):
    """
    Process a single sample: run inference, apply binary classification, compare
    """

    print(f"\n{'='*70}")
    print(f"🔬 Processing Sample: {sample_name}")
    print(f"{'='*70}")

    # Create output directory
    sample_dir = os.path.join(output_dir, sample_name)
    os.makedirs(sample_dir, exist_ok=True)

    # Load original image
    print("📂 Loading original image...")
    original_img = np.array(Image.open(original_path))
    h, w = original_img.shape[:2]
    print(f"   Dimensions: {h} × {w}")

    # Load ground truth
    print("📂 Loading ground truth annotation...")
    gt_mask = load_ground_truth_mask(gt_path)

    # Extract tiles
    print("📦 Extracting tiles...")
    tile_size = 1280
    overlap = 128
    tiles = extract_tiles_with_overlap(original_img, tile_size, overlap)
    print(f"   Extracted {len(tiles)} tiles")

    # Save tiles for inference
    tiles_dir = os.path.join(sample_dir, "tiles")
    tile_info = save_tiles_for_inference(tiles, tiles_dir)

    # Run model inference
    print("🧠 Running nnUNet inference...")
    predictions_dir = os.path.join(sample_dir, "predictions")
    success = run_nnunet_inference(tiles_dir, predictions_dir)

    if not success:
        print("❌ Inference failed!")
        return None

    # Merge predictions
    print("🔗 Merging tile predictions...")
    pred_mask = merge_predictions(predictions_dir, tile_info, (h, w))

    # Apply binary classification to GT
    print("🔍 Applying binary classification to GT...")
    gt_adipocyte_mask, gt_cellularity_stats = apply_binary_classification(original_img, gt_mask)

    if gt_cellularity_stats is None:
        print("❌ GT binary classification failed!")
        return None

    # Apply binary classification to Prediction
    print("🔍 Applying binary classification to Prediction...")
    pred_adipocyte_mask, pred_cellularity_stats = apply_binary_classification(original_img, pred_mask)

    if pred_cellularity_stats is None:
        print("❌ Prediction binary classification failed!")
        return None

    # Compare cellularity
    print("📊 Comparing cellularity...")
    comparison = compare_cellularity(pred_cellularity_stats, gt_cellularity_stats, sample_name)

    # Save results
    result_path = os.path.join(sample_dir, "cellularity_comparison.json")
    with open(result_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"✅ Results saved: {result_path}")

    # Create visualization
    print("🎨 Creating visualization...")
    viz_path = os.path.join(sample_dir, "cellularity_comparison_visualization.png")
    create_comparison_visualization(original_img, pred_mask, gt_mask,
                                   pred_adipocyte_mask, gt_adipocyte_mask,
                                   comparison, viz_path)

    # Save masks
    Image.fromarray(pred_mask * 255).save(os.path.join(sample_dir, "prediction_mask.png"))
    Image.fromarray(gt_mask * 255).save(os.path.join(sample_dir, "gt_mask.png"))
    Image.fromarray(pred_adipocyte_mask * 255).save(os.path.join(sample_dir, "pred_adipocyte_mask.png"))
    Image.fromarray(gt_adipocyte_mask * 255).save(os.path.join(sample_dir, "gt_adipocyte_mask.png"))

    # Print summary
    print(f"\n{'='*70}")
    print(f"📈 {sample_name} COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"GT Cellularity:         {comparison['gt_cellularity']:.2f}%")
    print(f"Prediction Cellularity: {comparison['prediction_cellularity']:.2f}%")
    print(f"Absolute Difference:    {comparison['absolute_difference']:+.2f}%")
    print(f"Relative Difference:    {comparison['relative_difference_percent']:+.2f}%")
    print(f"{'='*70}")

    # Cleanup intermediate files
    import shutil
    shutil.rmtree(tiles_dir)
    shutil.rmtree(predictions_dir)

    return comparison


# ============================================================================
# Statistical Analysis
# ============================================================================

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for data"""
    n = len(data)
    if n < 2:
        return None, None

    mean = np.mean(data)
    std_err = scipy_stats.sem(data)
    margin = std_err * scipy_stats.t.ppf((1 + confidence) / 2, n - 1)

    return mean - margin, mean + margin


def perform_statistical_analysis(all_comparisons):
    """
    Perform comprehensive statistical analysis

    Returns:
        stats_summary: Dictionary with statistical metrics
    """

    # Extract data
    gt_cellularities = [c['gt_cellularity'] for c in all_comparisons]
    pred_cellularities = [c['prediction_cellularity'] for c in all_comparisons]
    absolute_diffs = [c['absolute_difference'] for c in all_comparisons]
    relative_diffs = [c['relative_difference_percent'] for c in all_comparisons]

    # Paired t-test
    t_stat, p_value = scipy_stats.ttest_rel(pred_cellularities, gt_cellularities)

    # Correlation
    correlation, corr_p_value = scipy_stats.pearsonr(pred_cellularities, gt_cellularities)

    # Confidence intervals
    ci_low_abs, ci_high_abs = calculate_confidence_interval(absolute_diffs)
    ci_low_rel, ci_high_rel = calculate_confidence_interval(relative_diffs)

    stats_summary = {
        'n_samples': len(all_comparisons),
        'gt_cellularity': {
            'mean': float(np.mean(gt_cellularities)),
            'std': float(np.std(gt_cellularities)),
            'median': float(np.median(gt_cellularities)),
            'range': [float(min(gt_cellularities)), float(max(gt_cellularities))]
        },
        'pred_cellularity': {
            'mean': float(np.mean(pred_cellularities)),
            'std': float(np.std(pred_cellularities)),
            'median': float(np.median(pred_cellularities)),
            'range': [float(min(pred_cellularities)), float(max(pred_cellularities))]
        },
        'absolute_difference': {
            'mean': float(np.mean(absolute_diffs)),
            'std': float(np.std(absolute_diffs)),
            'median': float(np.median(absolute_diffs)),
            'range': [float(min(absolute_diffs)), float(max(absolute_diffs))],
            'ci_95': [float(ci_low_abs), float(ci_high_abs)] if ci_low_abs else None
        },
        'relative_difference': {
            'mean': float(np.mean(relative_diffs)),
            'std': float(np.std(relative_diffs)),
            'median': float(np.median(relative_diffs)),
            'range': [float(min(relative_diffs)), float(max(relative_diffs))],
            'ci_95': [float(ci_low_rel), float(ci_high_rel)] if ci_low_rel else None
        },
        'statistical_tests': {
            'paired_t_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            },
            'pearson_correlation': {
                'r': float(correlation),
                'p_value': float(corr_p_value),
                'r_squared': float(correlation ** 2)
            }
        }
    }

    return stats_summary


def create_category_analysis(all_comparisons):
    """Analyze results by cellularity category"""

    category_results = {}

    for category, info in SAMPLE_CATEGORIES.items():
        category_samples = info['samples']
        category_comparisons = [c for c in all_comparisons if c['sample_name'] in category_samples]

        if not category_comparisons:
            continue

        gt_vals = [c['gt_cellularity'] for c in category_comparisons]
        pred_vals = [c['prediction_cellularity'] for c in category_comparisons]
        diffs = [c['absolute_difference'] for c in category_comparisons]

        category_results[category] = {
            'description': info['description'],
            'n_samples': len(category_comparisons),
            'samples': [c['sample_name'] for c in category_comparisons],
            'gt_mean': float(np.mean(gt_vals)),
            'pred_mean': float(np.mean(pred_vals)),
            'mean_difference': float(np.mean(diffs)),
            'std_difference': float(np.std(diffs))
        }

    return category_results


# ============================================================================
# Summary Visualization
# ============================================================================

def create_summary_plots(all_comparisons, category_analysis, output_path):
    """Create comprehensive summary plots"""

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Extract data
    sample_names = [c['sample_name'] for c in all_comparisons]
    gt_cellularities = [c['gt_cellularity'] for c in all_comparisons]
    pred_cellularities = [c['prediction_cellularity'] for c in all_comparisons]
    absolute_diffs = [c['absolute_difference'] for c in all_comparisons]

    # Color by category
    colors = []
    for name in sample_names:
        if name in SAMPLE_CATEGORIES['high']['samples']:
            colors.append('red')
        elif name in SAMPLE_CATEGORIES['medium']['samples']:
            colors.append('orange')
        else:
            colors.append('green')

    # Plot 1: GT vs Prediction Cellularity
    ax1 = axes[0, 0]
    x_pos = np.arange(len(sample_names))
    width = 0.35
    bars1 = ax1.bar(x_pos - width/2, gt_cellularities, width, label='GT', alpha=0.8, color='steelblue')
    bars2 = ax1.bar(x_pos + width/2, pred_cellularities, width, label='Prediction', alpha=0.8, color='orange')
    ax1.set_xlabel('Sample', fontsize=12)
    ax1.set_ylabel('Cellularity (%)', fontsize=12)
    ax1.set_title('GT vs Prediction Cellularity', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sample_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Scatter plot with identity line
    ax2 = axes[0, 1]
    ax2.scatter(gt_cellularities, pred_cellularities, c=colors, s=100, alpha=0.7)
    min_val = min(min(gt_cellularities), min(pred_cellularities))
    max_val = max(max(gt_cellularities), max(pred_cellularities))
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Agreement')
    ax2.set_xlabel('GT Cellularity (%)', fontsize=12)
    ax2.set_ylabel('Prediction Cellularity (%)', fontsize=12)
    ax2.set_title('GT vs Prediction Correlation', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Add correlation coefficient
    corr = np.corrcoef(gt_cellularities, pred_cellularities)[0, 1]
    ax2.text(0.05, 0.95, f'r = {corr:.3f}\nR² = {corr**2:.3f}',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Absolute Difference
    ax3 = axes[0, 2]
    bars3 = ax3.bar(x_pos, absolute_diffs, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    ax3.set_xlabel('Sample', fontsize=12)
    ax3.set_ylabel('Difference (%)', fontsize=12)
    ax3.set_title('Cellularity Difference (Pred - GT)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(sample_names, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)

    # Add values on bars
    for bar, val in zip(bars3, absolute_diffs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height,
                f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

    # Plot 4: Category Comparison
    ax4 = axes[1, 0]
    categories = list(category_analysis.keys())
    cat_gt = [category_analysis[c]['gt_mean'] for c in categories]
    cat_pred = [category_analysis[c]['pred_mean'] for c in categories]
    cat_diff = [category_analysis[c]['mean_difference'] for c in categories]

    x_cat = np.arange(len(categories))
    width = 0.25
    ax4.bar(x_cat - width, cat_gt, width, label='GT', alpha=0.8, color='steelblue')
    ax4.bar(x_cat, cat_pred, width, label='Prediction', alpha=0.8, color='orange')
    ax4.bar(x_cat + width, cat_diff, width, label='Difference', alpha=0.8, color='red')
    ax4.set_xlabel('Cellularity Category', fontsize=12)
    ax4.set_ylabel('Cellularity (%)', fontsize=12)
    ax4.set_title('Category-wise Analysis', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_cat)
    ax4.set_xticklabels([c.capitalize() for c in categories])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    # Plot 5: Bland-Altman plot
    ax5 = axes[1, 1]
    means = [(gt + pred) / 2 for gt, pred in zip(gt_cellularities, pred_cellularities)]
    diffs = absolute_diffs

    ax5.scatter(means, diffs, c=colors, s=100, alpha=0.7)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    ax5.axhline(y=mean_diff, color='blue', linestyle='-', label=f'Mean: {mean_diff:.2f}%')
    ax5.axhline(y=mean_diff + 1.96*std_diff, color='red', linestyle='--',
                label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.2f}%')
    ax5.axhline(y=mean_diff - 1.96*std_diff, color='red', linestyle='--',
                label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.2f}%')
    ax5.set_xlabel('Mean Cellularity (%)', fontsize=12)
    ax5.set_ylabel('Difference (Pred - GT) (%)', fontsize=12)
    ax5.set_title('Bland-Altman Plot', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)

    # Plot 6: Statistical Summary
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Calculate statistics
    stats_text = f"""Statistical Summary (n={len(all_comparisons)})

Cellularity Difference (Pred - GT):
  • Mean:   {np.mean(absolute_diffs):.2f}%
  • Median: {np.median(absolute_diffs):.2f}%
  • Std:    {np.std(absolute_diffs):.2f}%
  • Range:  [{min(absolute_diffs):.2f}%, {max(absolute_diffs):.2f}%]

Correlation:
  • Pearson r: {corr:.3f}
  • R²:         {corr**2:.3f}

Paired t-test:
  • t-stat:     {scipy_stats.ttest_rel(pred_cellularities, gt_cellularities)[0]:.3f}
  • p-value:    {scipy_stats.ttest_rel(pred_cellularities, gt_cellularities)[1]:.4f}

Category Legend:
  🔴 High cellularity (BP25, BP31)
  🟠 Medium cellularity (BP1, BP7)
  🟢 Low cellularity (H2, H6)

Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""

    ax6.text(0.1, 0.5, stats_text, fontsize=10, va='center',
            family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare cellularity between model prediction and ground truth using binary classification"
    )

    parser.add_argument("--performance-dir",
                       default="/Users/baegjaehyeon/Desktop/Desktop/nnUNet/performance",
                       help="Performance directory containing raw/ and labeled/ subdirectories")

    parser.add_argument("--output", default="./cellularity_comparison_results",
                       help="Output directory for results")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🚀 Cellularity Comparison: Prediction vs Ground Truth")
    print("="*70)

    raw_dir = os.path.join(args.performance_dir, "raw")
    labeled_dir = os.path.join(args.performance_dir, "labeled")

    # Get all samples
    all_samples = []
    for category_info in SAMPLE_CATEGORIES.values():
        all_samples.extend(category_info['samples'])

    print(f"\n📊 Samples to process ({len(all_samples)}):")
    for category, info in SAMPLE_CATEGORIES.items():
        print(f"  {category.capitalize()}: {', '.join(info['samples'])}")

    # Process all samples
    all_comparisons = []

    for sample_name in all_samples:
        raw_path = os.path.join(raw_dir, f"{sample_name}.jpg")
        labeled_path = os.path.join(labeled_dir, f"{sample_name}-labeled-0.png")

        if not os.path.exists(raw_path):
            print(f"⚠️ Raw image not found: {raw_path}")
            continue

        if not os.path.exists(labeled_path):
            print(f"⚠️ Labeled image not found: {labeled_path}")
            continue

        try:
            comparison = process_single_sample(raw_path, labeled_path, args.output, sample_name)
            if comparison:
                all_comparisons.append(comparison)
        except Exception as e:
            print(f"❌ Error processing {sample_name}: {e}")
            continue

    if not all_comparisons:
        print("\n❌ No samples were successfully processed!")
        return

    # Perform statistical analysis
    print(f"\n📊 Performing statistical analysis...")
    stats_summary = perform_statistical_analysis(all_comparisons)

    # Category analysis
    category_analysis = create_category_analysis(all_comparisons)

    # Save all results
    summary_data = {
        'comparisons': all_comparisons,
        'statistical_summary': stats_summary,
        'category_analysis': category_analysis
    }

    summary_path = os.path.join(args.output, "cellularity_comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"✅ Summary saved: {summary_path}")

    # Create summary plots
    print(f"\n🎨 Creating summary visualization...")
    summary_plot_path = os.path.join(args.output, "cellularity_comparison_summary.png")
    create_summary_plots(all_comparisons, category_analysis, summary_plot_path)
    print(f"✅ Summary plot saved: {summary_plot_path}")

    # Print final summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    print(f"{'Sample':<10} {'GT (%)':<10} {'Pred (%)':<10} {'Diff (%)':<10} {'Category':<15}")
    print("-"*70)

    for comp in all_comparisons:
        # Find category
        cat = None
        for category, info in SAMPLE_CATEGORIES.items():
            if comp['sample_name'] in info['samples']:
                cat = category.capitalize()
                break

        print(f"{comp['sample_name']:<10} {comp['gt_cellularity']:<10.2f} "
              f"{comp['prediction_cellularity']:<10.2f} {comp['absolute_difference']:<+10.2f} {cat:<15}")

    print("-"*70)
    print(f"{'MEAN':<10} {stats_summary['gt_cellularity']['mean']:<10.2f} "
          f"{stats_summary['pred_cellularity']['mean']:<10.2f} "
          f"{stats_summary['absolute_difference']['mean']:<+10.2f}")
    print(f"{'STD':<10} {stats_summary['gt_cellularity']['std']:<10.2f} "
          f"{stats_summary['pred_cellularity']['std']:<10.2f} "
          f"{stats_summary['absolute_difference']['std']:<10.2f}")
    print("="*70)

    print(f"\n✅ Analysis complete! Results saved to: {args.output}")


if __name__ == "__main__":
    main()
