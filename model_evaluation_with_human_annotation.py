#!/usr/bin/env python3
"""
Model Performance Evaluation with Human Annotation Comparison
Compares nnUNet model predictions with human-labeled ground truth
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import argparse

# Environment variables setup for GCP
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
        print(f"Inference stdout: {result.stdout}")
        if result.stderr:
            print(f"Inference stderr: {result.stderr}")

        # Check if prediction files exist
        from glob import glob
        pred_files = glob(os.path.join(output_dir, "*.png"))
        print(f"✅ Generated {len(pred_files)} prediction files")
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
            print(f"⚠️ Prediction file not found: {pred_file}")
            continue

        pred_mask = np.array(Image.open(pred_path)).astype(np.float32)
        x1, y1, x2, y2 = tile_data['coords']

        global_cell_mask[y1:y2, x1:x2] += pred_mask
        overlap_count[y1:y2, x1:x2] += 1

    # Average overlapping regions
    with np.errstate(divide='ignore', invalid='ignore'):
        global_cell_mask = np.divide(global_cell_mask, overlap_count,
                                    where=overlap_count > 0)

    # Binarize
    global_cell_mask = (global_cell_mask > 0.5).astype(np.uint8)

    return global_cell_mask


def load_ground_truth_mask(gt_path):
    """Load and binarize ground truth mask from human annotation"""
    gt_img = Image.open(gt_path)
    gt_array = np.array(gt_img)

    # Convert to binary: any non-zero value becomes 1
    if len(gt_array.shape) == 3:
        # RGB 이미지인 경우 grayscale로 변환
        gt_array = cv2.cvtColor(gt_array, cv2.COLOR_RGB2GRAY)

    binary_gt = (gt_array > 0).astype(np.uint8)

    return binary_gt


def calculate_metrics(pred_mask, gt_mask):
    """
    Calculate evaluation metrics between prediction and ground truth

    Metrics:
    - IoU (Intersection over Union / Jaccard Index)
    - Dice Coefficient (F1 Score)
    - Precision
    - Recall
    - Pixel Accuracy
    """
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()

    # True Positives, False Positives, False Negatives, True Negatives
    tp = np.sum(np.logical_and(pred_flat == 1, gt_flat == 1))
    fp = np.sum(np.logical_and(pred_flat == 1, gt_flat == 0))
    fn = np.sum(np.logical_and(pred_flat == 0, gt_flat == 1))
    tn = np.sum(np.logical_and(pred_flat == 0, gt_flat == 0))

    # IoU (Intersection over Union)
    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union > 0 else 0.0

    # Dice Coefficient (F1 Score)
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall (Sensitivity)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Pixel Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'specificity': float(specificity),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
        'overlap_percentage': float(iou * 100)
    }

    return metrics


def create_comparison_visualization(original_img, pred_mask, gt_mask, metrics, output_path):
    """
    Create comprehensive comparison visualization
    Shows: Original | Ground Truth | Prediction | Overlay Comparison
    """
    # Ensure original is RGB
    if len(original_img.shape) == 2:
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original_img.copy()

    # Create Ground Truth overlay (Green)
    gt_overlay = original_rgb.copy()
    gt_overlay[gt_mask > 0] = [0, 255, 0]
    gt_viz = cv2.addWeighted(original_rgb, 0.6, gt_overlay, 0.4, 0)

    # Create Prediction overlay (Blue)
    pred_overlay = original_rgb.copy()
    pred_overlay[pred_mask > 0] = [0, 0, 255]
    pred_viz = cv2.addWeighted(original_rgb, 0.6, pred_overlay, 0.4, 0)

    # Create comparison overlay (TP: Yellow, FP: Red, FN: Cyan)
    comparison = original_rgb.copy()
    tp_mask = np.logical_and(pred_mask > 0, gt_mask > 0)
    fp_mask = np.logical_and(pred_mask > 0, gt_mask == 0)
    fn_mask = np.logical_and(pred_mask == 0, gt_mask > 0)

    comparison[tp_mask] = [255, 255, 0]  # True Positive: Yellow
    comparison[fp_mask] = [255, 0, 0]    # False Positive: Red
    comparison[fn_mask] = [0, 255, 255]  # False Negative: Cyan
    comparison_viz = cv2.addWeighted(original_rgb, 0.5, comparison, 0.5, 0)

    # Create figure with 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    # Panel 1: Original Image
    axes[0, 0].imshow(cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=16, fontweight='bold')
    axes[0, 0].axis('off')

    # Panel 2: Ground Truth
    axes[0, 1].imshow(cv2.cvtColor(gt_viz, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Ground Truth (Human Annotation)', fontsize=16, fontweight='bold')
    axes[0, 1].axis('off')
    gt_patch = mpatches.Patch(color='green', label='GT Cell Region')
    axes[0, 1].legend(handles=[gt_patch], loc='upper right', fontsize=12)

    # Panel 3: Model Prediction
    axes[1, 0].imshow(cv2.cvtColor(pred_viz, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Model Prediction', fontsize=16, fontweight='bold')
    axes[1, 0].axis('off')
    pred_patch = mpatches.Patch(color='blue', label='Predicted Cell Region')
    axes[1, 0].legend(handles=[pred_patch], loc='upper right', fontsize=12)

    # Panel 4: Comparison with Metrics
    axes[1, 1].imshow(cv2.cvtColor(comparison_viz, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Comparison: GT vs Prediction', fontsize=16, fontweight='bold')
    axes[1, 1].axis('off')

    # Legend for comparison
    tp_patch = mpatches.Patch(color='yellow', label=f'True Positive (TP): {metrics["true_positives"]:,}')
    fp_patch = mpatches.Patch(color='red', label=f'False Positive (FP): {metrics["false_positives"]:,}')
    fn_patch = mpatches.Patch(color='cyan', label=f'False Negative (FN): {metrics["false_negatives"]:,}')
    axes[1, 1].legend(handles=[tp_patch, fp_patch, fn_patch], loc='upper right', fontsize=11)

    # Add metrics text
    metrics_text = f"""Performance Metrics:

IoU (Overlap):        {metrics['iou']:.1%}
Dice Coefficient:     {metrics['dice']:.1%}
Precision:            {metrics['precision']:.1%}
Recall (Sensitivity): {metrics['recall']:.1%}
Pixel Accuracy:       {metrics['accuracy']:.1%}
Specificity:          {metrics['specificity']:.1%}

Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Model: nnUNet Dataset998"""

    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12,
             family='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Visualization saved: {output_path}")


def evaluate_single_image(original_path, gt_path, output_dir, image_name=None):
    """
    Evaluate model performance on a single image pair

    Args:
        original_path: Path to original image
        gt_path: Path to ground truth annotation
        output_dir: Directory to save results
        image_name: Optional name for the image
    """
    print(f"\n{'='*70}")
    print(f"🔬 Evaluating Image: {image_name or Path(original_path).stem}")
    print(f"{'='*70}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if image_name is None:
        image_name = Path(original_path).stem

    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # Load original image
    print("📂 Loading original image...")
    original_img = np.array(Image.open(original_path))
    h, w = original_img.shape[:2]
    print(f"   Dimensions: {h} × {w}")

    # Load ground truth
    print("📂 Loading ground truth annotation...")
    gt_mask = load_ground_truth_mask(gt_path)
    print(f"   GT pixels: {np.sum(gt_mask > 0):,} / {h*w:,} ({np.sum(gt_mask > 0)/(h*w)*100:.2f}%)")

    # Extract tiles
    print("📦 Extracting tiles...")
    tile_size = 1280
    overlap = 128
    tiles = extract_tiles_with_overlap(original_img, tile_size, overlap)
    print(f"   Extracted {len(tiles)} tiles")

    # Save tiles for inference
    tiles_dir = os.path.join(image_output_dir, "tiles")
    tile_info = save_tiles_for_inference(tiles, tiles_dir)

    # Run model inference
    print("🧠 Running nnUNet inference...")
    predictions_dir = os.path.join(image_output_dir, "predictions")
    success = run_nnunet_inference(tiles_dir, predictions_dir)

    if not success:
        print("❌ Inference failed!")
        return None

    # Merge predictions
    print("🔗 Merging tile predictions...")
    pred_mask = merge_predictions(predictions_dir, tile_info, (h, w))
    print(f"   Predicted pixels: {np.sum(pred_mask > 0):,} / {h*w:,} ({np.sum(pred_mask > 0)/(h*w)*100:.2f}%)")

    # Calculate metrics
    print("📊 Calculating evaluation metrics...")
    metrics = calculate_metrics(pred_mask, gt_mask)

    # Add metadata
    metrics['image_name'] = image_name
    metrics['image_dimensions'] = [h, w]
    metrics['num_tiles'] = len(tiles)
    metrics['gt_coverage'] = float(np.sum(gt_mask > 0) / (h * w))
    metrics['pred_coverage'] = float(np.sum(pred_mask > 0) / (h * w))

    # Save metrics
    metrics_path = os.path.join(image_output_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved: {metrics_path}")

    # Create visualization
    print("🎨 Creating comparison visualization...")
    viz_path = os.path.join(image_output_dir, "comparison_visualization.png")
    create_comparison_visualization(original_img, pred_mask, gt_mask, metrics, viz_path)

    # Save masks
    Image.fromarray(pred_mask * 255).save(os.path.join(image_output_dir, "prediction_mask.png"))
    Image.fromarray(gt_mask * 255).save(os.path.join(image_output_dir, "ground_truth_mask.png"))

    # Print summary
    print(f"\n{'='*70}")
    print("📈 EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"IoU (Intersection over Union): {metrics['iou']:.1%}")
    print(f"Dice Coefficient (F1 Score):   {metrics['dice']:.1%}")
    print(f"Precision:                      {metrics['precision']:.1%}")
    print(f"Recall (Sensitivity):           {metrics['recall']:.1%}")
    print(f"Pixel Accuracy:                 {metrics['accuracy']:.1%}")
    print(f"Specificity:                    {metrics['specificity']:.1%}")
    print(f"{'='*70}")

    # Cleanup intermediate files
    import shutil
    shutil.rmtree(tiles_dir)
    shutil.rmtree(predictions_dir)

    return metrics


def find_image_pairs(raw_dir, labeled_dir):
    """
    Automatically find matching image pairs between raw and labeled directories

    Returns:
        list of tuples: [(raw_path, labeled_path, image_name), ...]
    """
    from glob import glob

    raw_images = sorted(glob(os.path.join(raw_dir, "*.jpg")))
    pairs = []

    for raw_path in raw_images:
        basename = Path(raw_path).stem  # e.g., "BP31"

        # Find matching labeled file (e.g., "BP31-labeled-0.png")
        labeled_pattern = os.path.join(labeled_dir, f"{basename}-labeled-*.png")
        labeled_matches = glob(labeled_pattern)

        if labeled_matches:
            labeled_path = labeled_matches[0]  # Take first match
            pairs.append((raw_path, labeled_path, basename))
            print(f"✅ Found pair: {basename}")
        else:
            print(f"⚠️ No labeled image found for: {basename}")

    return pairs


def evaluate_batch(raw_dir, labeled_dir, output_dir):
    """
    Evaluate all image pairs in the performance directory
    """
    print("\n" + "="*70)
    print("🔍 Searching for image pairs...")
    print("="*70)
    print(f"Raw directory:     {raw_dir}")
    print(f"Labeled directory: {labeled_dir}")
    print("="*70)

    # Find all pairs
    pairs = find_image_pairs(raw_dir, labeled_dir)

    if not pairs:
        print("❌ No image pairs found!")
        return []

    print(f"\n📊 Found {len(pairs)} image pair(s) to evaluate")

    # Evaluate each pair
    all_metrics = []

    for i, (raw_path, labeled_path, image_name) in enumerate(pairs, 1):
        print(f"\n{'='*70}")
        print(f"Processing [{i}/{len(pairs)}]: {image_name}")
        print(f"{'='*70}")

        try:
            metrics = evaluate_single_image(
                original_path=raw_path,
                gt_path=labeled_path,
                output_dir=output_dir,
                image_name=image_name
            )

            if metrics:
                all_metrics.append(metrics)
        except Exception as e:
            print(f"❌ Error processing {image_name}: {e}")
            continue

    return all_metrics


def create_summary_report(all_metrics, output_dir):
    """
    Create summary report and visualization for all evaluated images
    """
    if not all_metrics:
        print("⚠️ No metrics to summarize")
        return

    # Save summary JSON
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✅ Summary JSON saved: {summary_path}")

    # Calculate statistics
    iou_scores = [m['iou'] for m in all_metrics]
    dice_scores = [m['dice'] for m in all_metrics]
    precision_scores = [m['precision'] for m in all_metrics]
    recall_scores = [m['recall'] for m in all_metrics]

    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    image_names = [m['image_name'] for m in all_metrics]
    x_pos = np.arange(len(image_names))

    # Plot 1: IoU Scores
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x_pos, [s*100 for s in iou_scores], color='steelblue', alpha=0.8)
    ax1.axhline(y=np.mean(iou_scores)*100, color='red', linestyle='--',
                label=f'Mean: {np.mean(iou_scores)*100:.1f}%')
    ax1.set_xlabel('Images', fontsize=12)
    ax1.set_ylabel('IoU (%)', fontsize=12)
    ax1.set_title('IoU (Intersection over Union) Scores', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(image_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, iou_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10)

    # Plot 2: Dice Coefficient
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, [s*100 for s in dice_scores], color='green', alpha=0.8)
    ax2.axhline(y=np.mean(dice_scores)*100, color='red', linestyle='--',
                label=f'Mean: {np.mean(dice_scores)*100:.1f}%')
    ax2.set_xlabel('Images', fontsize=12)
    ax2.set_ylabel('Dice Coefficient (%)', fontsize=12)
    ax2.set_title('Dice Coefficient (F1 Score)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(image_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, dice_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10)

    # Plot 3: Precision vs Recall
    ax3 = axes[1, 0]
    width = 0.35
    bars3_1 = ax3.bar(x_pos - width/2, [s*100 for s in precision_scores],
                      width, label='Precision', color='orange', alpha=0.8)
    bars3_2 = ax3.bar(x_pos + width/2, [s*100 for s in recall_scores],
                      width, label='Recall', color='purple', alpha=0.8)
    ax3.set_xlabel('Images', fontsize=12)
    ax3.set_ylabel('Score (%)', fontsize=12)
    ax3.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(image_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Summary Statistics Table
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_stats = f"""Evaluation Summary (n={len(all_metrics)})

IoU (Overlap):
  • Mean:   {np.mean(iou_scores)*100:.2f}%
  • Median: {np.median(iou_scores)*100:.2f}%
  • Std:    {np.std(iou_scores)*100:.2f}%
  • Range:  {min(iou_scores)*100:.1f}% - {max(iou_scores)*100:.1f}%

Dice Coefficient:
  • Mean:   {np.mean(dice_scores)*100:.2f}%
  • Median: {np.median(dice_scores)*100:.2f}%
  • Std:    {np.std(dice_scores)*100:.2f}%

Precision:
  • Mean:   {np.mean(precision_scores)*100:.2f}%

Recall:
  • Mean:   {np.mean(recall_scores)*100:.2f}%

Model: nnUNet Dataset998
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""

    ax4.text(0.1, 0.5, summary_stats, fontsize=11, va='center',
            family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    summary_plot_path = os.path.join(output_dir, "evaluation_summary_plot.png")
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Summary plot saved: {summary_plot_path}")

    # Print text summary
    print("\n" + "="*70)
    print("📊 EVALUATION SUMMARY")
    print("="*70)
    print(f"{'Image':<15} {'IoU':>10} {'Dice':>10} {'Precision':>12} {'Recall':>10}")
    print("-"*70)
    for m in all_metrics:
        print(f"{m['image_name']:<15} {m['iou']*100:>9.1f}% {m['dice']*100:>9.1f}% "
              f"{m['precision']*100:>11.1f}% {m['recall']*100:>9.1f}%")
    print("-"*70)
    print(f"{'MEAN':<15} {np.mean(iou_scores)*100:>9.1f}% {np.mean(dice_scores)*100:>9.1f}% "
          f"{np.mean(precision_scores)*100:>11.1f}% {np.mean(recall_scores)*100:>9.1f}%")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate nnUNet model performance against human annotations"
    )

    # Mode selection
    parser.add_argument("--mode", choices=['single', 'batch'], default='batch',
                       help="Evaluation mode: single image or batch processing")

    # Batch mode arguments
    parser.add_argument("--performance-dir",
                       default="/Users/baegjaehyeon/Desktop/Desktop/nnUNet/performance",
                       help="Performance directory containing raw/ and labeled/ subdirectories")

    # Single mode arguments (optional)
    parser.add_argument("--original", help="Path to original image (single mode)")
    parser.add_argument("--gt", help="Path to ground truth annotation (single mode)")
    parser.add_argument("--name", help="Optional name for this evaluation (single mode)")

    # Common arguments
    parser.add_argument("--output", default="./model_evaluation_results",
                       help="Output directory for results")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🚀 nnUNet Model Evaluation with Human Annotation")
    print("="*70)

    if args.mode == 'batch':
        # Batch mode: evaluate all pairs in performance directory
        raw_dir = os.path.join(args.performance_dir, "raw")
        labeled_dir = os.path.join(args.performance_dir, "labeled")

        if not os.path.exists(raw_dir):
            print(f"❌ Raw directory not found: {raw_dir}")
            sys.exit(1)

        if not os.path.exists(labeled_dir):
            print(f"❌ Labeled directory not found: {labeled_dir}")
            sys.exit(1)

        all_metrics = evaluate_batch(raw_dir, labeled_dir, args.output)

        if all_metrics:
            create_summary_report(all_metrics, args.output)
            print(f"\n✅ Batch evaluation complete! Results saved to: {args.output}")
        else:
            print("\n❌ No images were successfully evaluated!")
            sys.exit(1)

    else:
        # Single mode: evaluate one image pair
        if not args.original or not args.gt:
            print("❌ For single mode, --original and --gt are required!")
            sys.exit(1)

        print(f"Original Image: {args.original}")
        print(f"Ground Truth:   {args.gt}")
        print(f"Output Dir:     {args.output}")
        print("="*70)

        metrics = evaluate_single_image(
            original_path=args.original,
            gt_path=args.gt,
            output_dir=args.output,
            image_name=args.name
        )

        if metrics:
            print(f"\n✅ Evaluation complete! Results saved to: {args.output}")
        else:
            print("\n❌ Evaluation failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
