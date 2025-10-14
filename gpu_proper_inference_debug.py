#!/usr/bin/env python3
"""
WSI Inference using nnUNet Model on GPU Server (Debug Version)
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2
import json
from datetime import datetime
from glob import glob

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

        # Convert to grayscale
        if len(tile.shape) == 3:
            tile_gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        else:
            tile_gray = tile
            
        # nnUNet format: _0000.png
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
    print("🧠 Running nnUNet inference...")

    # Use nnUNet v2
    import subprocess

    # Check checkpoint
    checkpoint_path = "/home/baegjaehyeon/nnUNet_results/Dataset998_BC_1280/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_final.pth"
    if not os.path.exists(checkpoint_path):
        # Copy checkpoint_latest to final
        latest_path = "/home/baegjaehyeon/nnUNet_results/Dataset998_BC_1280/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_latest.pth"
        if os.path.exists(latest_path):
            import shutil
            shutil.copy(latest_path, checkpoint_path)
            print(f"✅ Copied checkpoint_latest to checkpoint_final")
        else:
            print(f"❌ No checkpoint found!")
            return False
    
    # 추론 명령
    cmd = [
        "/home/baegjaehyeon/nnunet_env/bin/nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-d", "998",
        "-c", "2d",
        "-f", "0",
        "--disable_tta"
    ]
    
    print(f"📋 Command: {' '.join(cmd)}")
    
    # 환경변수와 함께 실행
    env = os.environ.copy()
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        print(f"stdout: {result.stdout[:500]}")
        if result.stderr:
            print(f"stderr: {result.stderr[:500]}")
        
        # 예측 파일 확인
        pred_files = glob(os.path.join(output_dir, "*.png"))
        print(f"✅ Generated {len(pred_files)} prediction files")
        
        if len(pred_files) == 0:
            print("⚠️ No predictions generated, using fallback...")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def fallback_inference(input_dir, output_dir):
    """Fallback: Simple threshold-based prediction"""
    print("⚠️ Using fallback threshold-based prediction...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    from scipy import ndimage
    
    tile_files = sorted(glob(os.path.join(input_dir, "tile_*_0000.png")))
    
    for tile_path in tile_files:
        img = np.array(Image.open(tile_path))
        
        if len(img.shape) == 3:
            img = img[:,:,0]
        
        # Threshold-based cell detection
        cell_mask = (img < 180).astype(np.uint8)
        
        # 모폴로지 연산
        cell_mask = ndimage.binary_opening(cell_mask, iterations=2)
        cell_mask = ndimage.binary_closing(cell_mask, iterations=2)
        
        # 저장
        basename = os.path.basename(tile_path).replace("_0000.png", ".png")
        output_path = os.path.join(output_dir, basename)
        Image.fromarray(cell_mask.astype(np.uint8) * 255).save(output_path)
    
    print(f"✅ Fallback completed: {len(tile_files)} tiles")
    return True


def merge_predictions(predictions_dir, tile_info, wsi_shape):
    """Merge tile predictions into full WSI"""
    h, w = wsi_shape

    # Initialize masks for full WSI size
    global_cell_mask = np.zeros((h, w), dtype=np.float32)
    overlap_count = np.zeros((h, w), dtype=np.int32)
    
    # 예측 파일 확인
    pred_files = glob(os.path.join(predictions_dir, "tile_*.png"))
    print(f"📊 Found {len(pred_files)} prediction files to merge")
    
    # 각 타일 예측 결과 병합
    for tile_data in tile_info:
        tile_idx = tile_data['index']
        pred_file = f"tile_{tile_idx:04d}.png"
        pred_path = os.path.join(predictions_dir, pred_file)
        
        if not os.path.exists(pred_path):
            continue
            
        # 예측 마스크 로드
        pred_mask = np.array(Image.open(pred_path)) / 255.0
        
        # 타일 위치
        x1, y1, x2, y2 = tile_data['coords']
        
        # 전체 마스크에 누적
        global_cell_mask[y1:y2, x1:x2] += pred_mask
        overlap_count[y1:y2, x1:x2] += 1
    
    # 오버랩 영역 평균 처리
    with np.errstate(divide='ignore', invalid='ignore'):
        global_cell_mask = np.divide(global_cell_mask, overlap_count, 
                                    where=overlap_count > 0)
    
    # 이진화
    global_cell_mask = (global_cell_mask > 0.5).astype(np.uint8)
    
    print(f"✅ Merged mask: {np.sum(global_cell_mask > 0)} cell pixels detected")
    
    return global_cell_mask


def analyze_cellularity(wsi_image, cell_mask):
    """계층적 세포충실도 분석"""
    
    # Grayscale 변환
    if len(wsi_image.shape) == 3:
        gray = cv2.cvtColor(wsi_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = wsi_image
    
    # 세포 영역 내 픽셀만 분석
    cell_pixels = gray[cell_mask > 0]
    
    print(f"📊 Analyzing {len(cell_pixels)} cell pixels...")
    
    if len(cell_pixels) == 0:
        return None, {}
    
    # Adipocyte detection (top 25% brightness within cellular regions)
    threshold = np.percentile(cell_pixels, 75)
    threshold = max(180, min(240, threshold))
    
    print(f"🔍 Adipocyte threshold: {threshold}")
    
    # 세포 영역 내에서만 지방세포 검출
    adipocyte_mask = np.logical_and(gray > threshold, cell_mask > 0).astype(np.uint8)
    
    # 전체 통계 계산
    total_cell_pixels = np.sum(cell_mask > 0)
    total_adipocyte_pixels = np.sum(adipocyte_mask > 0)
    total_normal_pixels = total_cell_pixels - total_adipocyte_pixels
    
    cellularity = total_normal_pixels / total_cell_pixels if total_cell_pixels > 0 else 0.0
    
    stats = {
        'global_cellularity': float(cellularity),
        'total_cell_pixels': int(total_cell_pixels),
        'total_adipocyte_pixels': int(total_adipocyte_pixels),
        'total_normal_pixels': int(total_normal_pixels),
        'adipocyte_ratio': float(total_adipocyte_pixels / total_cell_pixels) if total_cell_pixels > 0 else 0,
        'adipocyte_threshold': int(threshold),
        'cell_coverage': float(total_cell_pixels / (wsi_image.shape[0] * wsi_image.shape[1]))
    }
    
    print(f"✅ Cellularity: {cellularity:.1%}, Coverage: {stats['cell_coverage']:.1%}")
    
    return adipocyte_mask, stats


def process_wsi(wsi_name, wsi_path):
    """Process a single WSI"""
    print(f"\n{'='*60}")
    print(f"🔬 Processing {wsi_name}")
    print(f"{'='*60}")
    
    # 출력 디렉토리
    base_dir = f"/home/baegjaehyeon/{wsi_name}_nnunet"
    tiles_dir = os.path.join(base_dir, "tiles")
    predictions_dir = os.path.join(base_dir, "predictions")
    analysis_dir = os.path.join(base_dir, "analysis")
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Load WSI
    print(f"📂 Loading WSI: {wsi_path}")
    wsi_image = np.array(Image.open(wsi_path))
    h, w = wsi_image.shape[:2]
    print(f"📐 WSI dimensions: {h} × {w}")
    
    # 타일 추출
    tile_size = 1280
    overlap = 128
    print(f"\n🔍 Extracting tiles (size={tile_size}, overlap={overlap})")
    tiles = extract_tiles_with_overlap(wsi_image, tile_size, overlap)
    print(f"✅ Extracted {len(tiles)} tiles")
    
    # 타일 저장
    print(f"\n💾 Saving tiles...")
    tile_info = save_tiles_for_inference(tiles, tiles_dir)
    
    # 타일 정보 저장
    info_data = {
        'wsi_name': wsi_name,
        'wsi_path': wsi_path,
        'wsi_dimensions': [h, w],
        'tile_size': tile_size,
        'overlap': overlap,
        'num_tiles': len(tiles),
        'tiles': tile_info
    }
    
    info_path = os.path.join(base_dir, "tile_info.json")
    with open(info_path, 'w') as f:
        json.dump(info_data, f, indent=2)
    
    # Run nnUNet inference
    print(f"\n🚀 Running inference...")
    success = run_nnunet_inference(tiles_dir, predictions_dir)
    
    if not success:
        success = fallback_inference(tiles_dir, predictions_dir)
    
    if not success:
        print("❌ Inference failed completely")
        return None
    
    # 예측 결과 병합
    print(f"\n📦 Merging predictions...")
    cell_mask = merge_predictions(predictions_dir, tile_info, (h, w))
    
    # 세포충실도 분석
    print(f"\n📊 Analyzing cellularity...")
    adipocyte_mask, stats = analyze_cellularity(wsi_image, cell_mask)
    
    if stats:
        stats['wsi_name'] = wsi_name
        stats['wsi_dimensions'] = [h, w]
        stats['wsi_total_pixels'] = h * w
        stats['num_tiles'] = len(tiles)
        stats['tile_size'] = tile_size
        stats['overlap'] = overlap
        
        # 결과 저장
        result_path = os.path.join(analysis_dir, "cellularity_analysis.json")
        with open(result_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"📊 Saved results to {result_path}")
        
        # 시각화 생성
        print(f"\n🎨 Creating visualization...")
        try:
            if len(wsi_image.shape) == 2:
                wsi_rgb = cv2.cvtColor(wsi_image, cv2.COLOR_GRAY2RGB)
            else:
                wsi_rgb = wsi_image.copy()
            
            # 정상 세포와 지방세포 구분
            normal_mask = np.logical_and(cell_mask > 0, adipocyte_mask == 0)

            # Create overlay
            overlay = wsi_rgb.copy()
            overlay[normal_mask > 0] = [0, 255, 0]  # Normal cells: green
            overlay[adipocyte_mask > 0] = [255, 255, 0]  # Adipocytes: yellow
            
            # 블렌딩
            result_vis = cv2.addWeighted(wsi_rgb, 0.5, overlay, 0.5, 0)
            
            # 리사이즈
            h_vis, w_vis = result_vis.shape[:2]
            max_dim = 3000
            if max(h_vis, w_vis) > max_dim:
                scale = max_dim / max(h_vis, w_vis)
                new_h, new_w = int(h_vis * scale), int(w_vis * scale)
                result_vis = cv2.resize(result_vis, (new_w, new_h))
            
            vis_path = os.path.join(analysis_dir, "cellularity_visualization.png")
            cv2.imwrite(vis_path, cv2.cvtColor(result_vis, cv2.COLOR_RGB2BGR))
            print(f"🎨 Saved visualization to {vis_path}")
            
            # 마스크 저장
            Image.fromarray(cell_mask * 255).save(os.path.join(analysis_dir, "cell_mask.png"))
            Image.fromarray(adipocyte_mask * 255).save(os.path.join(analysis_dir, "adipocyte_mask.png"))
            
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
        
        # 결과 출력
        print(f"\n{'='*60}")
        print(f"📊 {wsi_name} RESULTS")
        print(f"{'='*60}")
        print(f"🎯 Cellularity: {stats['global_cellularity']:.1%}")
        print(f"📍 Coverage: {stats['cell_coverage']:.1%}")
        print(f"📍 Adipocyte: {stats['adipocyte_ratio']:.1%}")
        print(f"{'='*60}")
        
        return stats
    
    return None


def main():
    print("\n" + "="*70)
    print("🚀 WSI Cellularity Analysis (Debug Version)")
    print("="*70)
    
    # Process BC_002 and BC_012
    wsi_configs = [
        ("BC_002", "/home/baegjaehyeon/nnUNet_raw/Dataset999_BC_Segmentation/imagesTr/BC_002_0000.jpg"),
        ("BC_012", "/home/baegjaehyeon/nnUNet_raw/Dataset999_BC_Segmentation/imagesTr/BC_012_0000.jpg")
    ]
    
    results = {}
    for wsi_name, wsi_path in wsi_configs:
        stats = process_wsi(wsi_name, wsi_path)
        if stats:
            results[wsi_name] = stats
    
    # 비교 결과 출력
    if len(results) == 2:
        print("\n" + "="*70)
        print("📊 COMPARISON")
        print("="*70)
        bc002 = results.get('BC_002', {})
        bc012 = results.get('BC_012', {})
        
        print(f"BC_002: Cellularity {bc002.get('global_cellularity', 0):.1%}, Coverage {bc002.get('cell_coverage', 0):.1%}")
        print(f"BC_012: Cellularity {bc012.get('global_cellularity', 0):.1%}, Coverage {bc012.get('cell_coverage', 0):.1%}")
        print("="*70)
    
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()