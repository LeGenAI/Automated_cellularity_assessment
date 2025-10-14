#!/usr/bin/env python3
"""
Complete WSI-Aware Tiling System for All Resolutions
Enables multi-resolution ensemble inference with full WSI reconstruction
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import shutil
from datetime import datetime
import time

class CompleteWSITilingSystem:
    def __init__(self, base_path="/Users/baegjaehyeon/Desktop/nnUNet"):
        self.base_path = Path(base_path)
        self.source_path = self.base_path / "nnUNet_raw" / "Dataset999_BC_Segmentation"
        
        # Create metadata directory
        self.metadata_path = self.base_path / "wsi_metadata"
        self.metadata_path.mkdir(exist_ok=True)
        
        # Complete configuration for all resolutions
        self.configs = {
            "640": {
                "dataset_id": "Dataset997_BC_640",
                "tile_size": 640,
                "overlap": 64,
                "min_foreground": 100,
                "augmentation_enabled": True,
                "max_augmentations": 2
            },
            "960": {
                "dataset_id": "Dataset996_BC_Transfer",
                "tile_size": 960,
                "overlap": 96,
                "min_foreground": 150,
                "augmentation_enabled": True,
                "max_augmentations": 1
            },
            "1280": {
                "dataset_id": "Dataset998_BC_1280",
                "tile_size": 1280,
                "overlap": 128,
                "min_foreground": 200,
                "augmentation_enabled": False,  # No augmentation for high-res
                "max_augmentations": 0
            }
        }
        
        # Global statistics
        self.global_stats = {
            "total_wsi": 0,
            "tiles_per_resolution": {},
            "processing_time": {},
            "wsi_dimensions": {}
        }
    
    def clean_existing_datasets(self):
        """Remove existing tiled datasets for fresh start"""
        print("🧹 Cleaning existing datasets...")
        for config in self.configs.values():
            dataset_path = self.base_path / "nnUNet_raw" / config["dataset_id"]
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
                print(f"   Removed {dataset_path.name}")
        print("✅ Cleanup complete")
    
    def get_wsi_id(self, image_path):
        """Generate unique WSI identifier"""
        return image_path.stem.replace("_0000", "")
    
    def extract_tiles_with_full_metadata(self, image_path, label_path, config, resolution):
        """Extract tiles with comprehensive metadata for reconstruction"""
        wsi_id = self.get_wsi_id(image_path)
        
        # Read images
        image = cv2.imread(str(image_path))
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None or label is None:
            return []
        
        h, w = image.shape[:2]
        tile_size = config["tile_size"]
        overlap = config["overlap"]
        step = tile_size - overlap
        
        # Calculate grid dimensions
        grid_x = (w - tile_size) // step + 1
        grid_y = (h - tile_size) // step + 1
        
        tiles_with_metadata = []
        
        # Systematic grid extraction
        for y_idx in range(grid_y):
            for x_idx in range(grid_x):
                x = x_idx * step
                y = y_idx * step
                
                # Ensure we don't exceed image boundaries
                x = min(x, w - tile_size)
                y = min(y, h - tile_size)
                
                img_tile = image[y:y+tile_size, x:x+tile_size]
                lbl_tile = label[y:y+tile_size, x:x+tile_size]
                
                foreground_pixels = np.sum(lbl_tile > 0)
                
                if foreground_pixels >= config["min_foreground"]:
                    # Comprehensive metadata
                    tile_metadata = {
                        "wsi_id": wsi_id,
                        "resolution": resolution,
                        "wsi_dimensions": {"width": w, "height": h},
                        "grid_position": {"x_idx": x_idx, "y_idx": y_idx},
                        "pixel_coordinates": {"x": x, "y": y, "w": tile_size, "h": tile_size},
                        "grid_dimensions": {"grid_x": grid_x, "grid_y": grid_y},
                        "overlap": overlap,
                        "step": step,
                        "foreground_pixels": int(foreground_pixels),
                        "foreground_ratio": float(foreground_pixels / (tile_size * tile_size)),
                        "augmentation": "original"
                    }
                    
                    tiles_with_metadata.append((img_tile, lbl_tile, tile_metadata))
        
        # Edge tiles handling
        # Right edge
        if w % step != 0 and w > tile_size:
            x = w - tile_size
            for y_idx in range(grid_y):
                y = y_idx * step
                y = min(y, h - tile_size)
                
                img_tile = image[y:y+tile_size, x:x+tile_size]
                lbl_tile = label[y:y+tile_size, x:x+tile_size]
                
                foreground_pixels = np.sum(lbl_tile > 0)
                if foreground_pixels >= config["min_foreground"]:
                    tile_metadata = {
                        "wsi_id": wsi_id,
                        "resolution": resolution,
                        "wsi_dimensions": {"width": w, "height": h},
                        "grid_position": {"x_idx": -1, "y_idx": y_idx},  # -1 indicates edge
                        "pixel_coordinates": {"x": x, "y": y, "w": tile_size, "h": tile_size},
                        "grid_dimensions": {"grid_x": grid_x, "grid_y": grid_y},
                        "overlap": overlap,
                        "step": step,
                        "foreground_pixels": int(foreground_pixels),
                        "foreground_ratio": float(foreground_pixels / (tile_size * tile_size)),
                        "edge_type": "right",
                        "augmentation": "original"
                    }
                    tiles_with_metadata.append((img_tile, lbl_tile, tile_metadata))
        
        # Bottom edge
        if h % step != 0 and h > tile_size:
            y = h - tile_size
            for x_idx in range(grid_x):
                x = x_idx * step
                x = min(x, w - tile_size)
                
                img_tile = image[y:y+tile_size, x:x+tile_size]
                lbl_tile = label[y:y+tile_size, x:x+tile_size]
                
                foreground_pixels = np.sum(lbl_tile > 0)
                if foreground_pixels >= config["min_foreground"]:
                    tile_metadata = {
                        "wsi_id": wsi_id,
                        "resolution": resolution,
                        "wsi_dimensions": {"width": w, "height": h},
                        "grid_position": {"x_idx": x_idx, "y_idx": -1},  # -1 indicates edge
                        "pixel_coordinates": {"x": x, "y": y, "w": tile_size, "h": tile_size},
                        "grid_dimensions": {"grid_x": grid_x, "grid_y": grid_y},
                        "overlap": overlap,
                        "step": step,
                        "foreground_pixels": int(foreground_pixels),
                        "foreground_ratio": float(foreground_pixels / (tile_size * tile_size)),
                        "edge_type": "bottom",
                        "augmentation": "original"
                    }
                    tiles_with_metadata.append((img_tile, lbl_tile, tile_metadata))
        
        # Corner tile
        if w % step != 0 and h % step != 0 and w > tile_size and h > tile_size:
            x = w - tile_size
            y = h - tile_size
            
            img_tile = image[y:y+tile_size, x:x+tile_size]
            lbl_tile = label[y:y+tile_size, x:x+tile_size]
            
            foreground_pixels = np.sum(lbl_tile > 0)
            if foreground_pixels >= config["min_foreground"]:
                tile_metadata = {
                    "wsi_id": wsi_id,
                    "resolution": resolution,
                    "wsi_dimensions": {"width": w, "height": h},
                    "grid_position": {"x_idx": -1, "y_idx": -1},
                    "pixel_coordinates": {"x": x, "y": y, "w": tile_size, "h": tile_size},
                    "grid_dimensions": {"grid_x": grid_x, "grid_y": grid_y},
                    "overlap": overlap,
                    "step": step,
                    "foreground_pixels": int(foreground_pixels),
                    "foreground_ratio": float(foreground_pixels / (tile_size * tile_size)),
                    "edge_type": "corner",
                    "augmentation": "original"
                }
                tiles_with_metadata.append((img_tile, lbl_tile, tile_metadata))
        
        return tiles_with_metadata
    
    def apply_augmentation(self, img_tile, lbl_tile, aug_type):
        """Apply specific augmentation type"""
        if aug_type == "rot90":
            return cv2.rotate(img_tile, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(lbl_tile, cv2.ROTATE_90_CLOCKWISE)
        elif aug_type == "rot180":
            return cv2.rotate(img_tile, cv2.ROTATE_180), cv2.rotate(lbl_tile, cv2.ROTATE_180)
        elif aug_type == "flip_h":
            return cv2.flip(img_tile, 1), cv2.flip(lbl_tile, 1)
        elif aug_type == "flip_v":
            return cv2.flip(img_tile, 0), cv2.flip(lbl_tile, 0)
        else:
            return img_tile, lbl_tile
    
    def process_resolution(self, resolution):
        """Process a single resolution with full WSI tracking"""
        config = self.configs[resolution]
        target_path = self.base_path / "nnUNet_raw" / config["dataset_id"]
        
        # Create directories
        (target_path / "imagesTr").mkdir(parents=True, exist_ok=True)
        (target_path / "labelsTr").mkdir(parents=True, exist_ok=True)
        
        print(f"\n📐 Processing {resolution}x{resolution} tiles with WSI reconstruction mapping")
        print("="*70)
        
        # Get all image-label pairs
        images = sorted((self.source_path / "imagesTr").glob("*.jpg"))
        labels = sorted((self.source_path / "labelsTr").glob("*.png"))
        
        tile_id = 0
        all_tile_metadata = []
        wsi_reconstruction_map = {}
        tiles_per_wsi = []
        
        start_time = time.time()
        
        # Process all WSIs
        for img_path, lbl_path in tqdm(zip(images, labels), 
                                       total=len(images), 
                                       desc=f"Processing {resolution}"):
            wsi_id = self.get_wsi_id(img_path)
            wsi_reconstruction_map[wsi_id] = {
                "source_image": str(img_path.relative_to(self.base_path)),
                "source_label": str(lbl_path.relative_to(self.base_path)),
                "resolution": resolution,
                "tiles": []
            }
            
            # Extract tiles
            tiles = self.extract_tiles_with_full_metadata(img_path, lbl_path, config, resolution)
            tiles_per_wsi.append(len(tiles))
            
            # Save tiles
            for img_tile, lbl_tile, metadata in tiles:
                case_id = f"BC_{tile_id:05d}"
                
                # Save images
                img_save = target_path / "imagesTr" / f"{case_id}_0000.png"
                lbl_save = target_path / "labelsTr" / f"{case_id}.png"
                
                cv2.imwrite(str(img_save), img_tile)
                cv2.imwrite(str(lbl_save), lbl_tile)
                
                # Add file paths to metadata
                metadata["case_id"] = case_id
                metadata["file_paths"] = {
                    "image": str(img_save.relative_to(self.base_path)),
                    "label": str(lbl_save.relative_to(self.base_path))
                }
                
                all_tile_metadata.append(metadata)
                wsi_reconstruction_map[wsi_id]["tiles"].append(case_id)
                
                tile_id += 1
                
                # Apply augmentations if enabled
                if config["augmentation_enabled"] and metadata["foreground_ratio"] > 0.1:
                    augmentation_types = ["rot90", "flip_h"][:config["max_augmentations"]]
                    
                    for aug_type in augmentation_types:
                        aug_img, aug_lbl = self.apply_augmentation(img_tile, lbl_tile, aug_type)
                        
                        case_id = f"BC_{tile_id:05d}"
                        img_save = target_path / "imagesTr" / f"{case_id}_0000.png"
                        lbl_save = target_path / "labelsTr" / f"{case_id}.png"
                        
                        cv2.imwrite(str(img_save), aug_img)
                        cv2.imwrite(str(lbl_save), aug_lbl)
                        
                        # Create augmented metadata
                        aug_metadata = metadata.copy()
                        aug_metadata["case_id"] = case_id
                        aug_metadata["augmentation"] = aug_type
                        aug_metadata["file_paths"] = {
                            "image": str(img_save.relative_to(self.base_path)),
                            "label": str(lbl_save.relative_to(self.base_path))
                        }
                        
                        all_tile_metadata.append(aug_metadata)
                        wsi_reconstruction_map[wsi_id]["tiles"].append(case_id)
                        
                        tile_id += 1
        
        processing_time = time.time() - start_time
        
        # Create dataset.json for nnUNet
        dataset_json = {
            "channel_names": {"0": "microscopy"},
            "labels": {
                "background": 0,
                "cell": 1
            },
            "numTraining": tile_id,
            "file_ending": ".png",
            "description": f"BC WSI {resolution}x{resolution} with reconstruction mapping",
            "wsi_aware": True,
            "tile_config": config,
            "statistics": {
                "total_tiles": tile_id,
                "source_wsi_count": len(images),
                "processing_time": processing_time,
                "avg_tiles_per_wsi": np.mean(tiles_per_wsi) if tiles_per_wsi else 0
            }
        }
        
        with open(target_path / "dataset.json", 'w') as f:
            json.dump(dataset_json, f, indent=2)
        
        # Save detailed tile metadata
        with open(target_path / "tile_metadata.json", 'w') as f:
            json.dump(all_tile_metadata, f, indent=2)
        
        # Save WSI reconstruction map
        recon_file = self.metadata_path / f"wsi_reconstruction_{resolution}.json"
        with open(recon_file, 'w') as f:
            json.dump(wsi_reconstruction_map, f, indent=2)
        
        # Update global stats
        self.global_stats["tiles_per_resolution"][resolution] = tile_id
        self.global_stats["processing_time"][resolution] = processing_time
        
        print(f"✅ Created {tile_id} tiles for {resolution}x{resolution}")
        print(f"   Average tiles per WSI: {np.mean(tiles_per_wsi):.1f}")
        print(f"   Processing time: {processing_time:.1f}s")
        
        return all_tile_metadata
    
    def create_master_ensemble_config(self):
        """Create master configuration for multi-resolution ensemble"""
        ensemble_config = {
            "created": datetime.now().isoformat(),
            "pipeline_name": "Multi-Resolution WSI Ensemble with Reconstruction",
            "resolutions": {},
            "wsi_master_index": {},
            "inference_strategy": {
                "method": "weighted_ensemble",
                "weights": {
                    "640": 0.25,
                    "960": 0.35,
                    "1280": 0.40
                },
                "reconstruction": {
                    "method": "grid_based",
                    "overlap_handling": "weighted_average",
                    "edge_blending": True
                }
            },
            "post_processing": {
                "remove_small_objects": True,
                "min_object_size": 50,
                "morphological_operations": ["closing", "opening"],
                "smoothing": "bilateral"
            }
        }
        
        # Build resolution information
        for resolution in ["640", "960", "1280"]:
            config = self.configs[resolution]
            dataset_path = self.base_path / "nnUNet_raw" / config["dataset_id"]
            
            if dataset_path.exists():
                # Load reconstruction map
                recon_file = self.metadata_path / f"wsi_reconstruction_{resolution}.json"
                if recon_file.exists():
                    with open(recon_file, 'r') as f:
                        recon_map = json.load(f)
                    
                    ensemble_config["resolutions"][resolution] = {
                        "dataset_path": str(dataset_path.relative_to(self.base_path)),
                        "tile_size": config["tile_size"],
                        "overlap": config["overlap"],
                        "total_tiles": self.global_stats["tiles_per_resolution"].get(resolution, 0),
                        "wsi_count": len(recon_map)
                    }
                    
                    # Build master WSI index
                    for wsi_id in recon_map:
                        if wsi_id not in ensemble_config["wsi_master_index"]:
                            ensemble_config["wsi_master_index"][wsi_id] = {}
                        
                        ensemble_config["wsi_master_index"][wsi_id][resolution] = {
                            "tile_count": len(recon_map[wsi_id]["tiles"]),
                            "reconstruction_map": f"wsi_reconstruction_{resolution}.json"
                        }
        
        # Save master ensemble configuration
        ensemble_path = self.metadata_path / "master_ensemble_config.json"
        with open(ensemble_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        print(f"✅ Master ensemble configuration saved to {ensemble_path}")
        
        return ensemble_config
    
    def run_complete_pipeline(self):
        """Execute complete WSI-aware tiling pipeline"""
        print("\n" + "="*70)
        print("🚀 COMPLETE WSI-AWARE TILING PIPELINE")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Clean existing datasets
        self.clean_existing_datasets()
        
        # Get dataset info
        images = list((self.source_path / "imagesTr").glob("*.jpg"))
        self.global_stats["total_wsi"] = len(images)
        print(f"\n📊 Found {len(images)} WSI images to process")
        
        # Process each resolution
        for resolution in ["640", "960", "1280"]:
            self.process_resolution(resolution)
        
        # Create master ensemble configuration
        self.create_master_ensemble_config()
        
        # Print final summary
        self.print_summary()
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*70)
        print("📊 COMPLETE TILING SUMMARY")
        print("="*70)
        
        total_tiles = sum(self.global_stats["tiles_per_resolution"].values())
        total_time = sum(self.global_stats["processing_time"].values())
        
        print(f"\n⏱️  Total processing time: {total_time:.1f} seconds")
        print(f"📷 WSI images processed: {self.global_stats['total_wsi']}")
        
        print("\n🎯 Tiles created per resolution:")
        for res, count in self.global_stats["tiles_per_resolution"].items():
            print(f"   {res:>4}x{res:<4}: {count:,} tiles")
        print(f"   {'Total':>8}: {total_tiles:,} tiles")
        
        print("\n💾 Metadata files created:")
        for res in ["640", "960", "1280"]:
            print(f"   wsi_reconstruction_{res}.json ✅")
        print("   master_ensemble_config.json ✅")
        
        print("\n✅ WSI-aware tiling complete!")
        print("🎯 Ready for multi-resolution training and ensemble inference with WSI reconstruction!")
        print("="*70)

if __name__ == "__main__":
    system = CompleteWSITilingSystem()
    system.run_complete_pipeline()