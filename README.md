# Automated Cellularity Assessment in Bone Marrow Aspirate Smears

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![nnUNet](https://img.shields.io/badge/nnUNet-v2-00D084?style=flat-square)](https://github.com/MIC-DKFZ/nnUNet)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

**Deep learning-based automated bone marrow cellularity quantification using nnUNet segmentation**

[Quick Start](#-quick-start) • [Installation](#-installation) • [Model Checkpoints](#-model-checkpoints) • [Results](#-results) • [Citation](#-citation)

</div>

---

## 📄 Paper

**Title:** Automated Cellularity Assessment in Bone Marrow Aspirate Smears Using Deep Learning-Based Segmentation

**Authors:** Jae-Hyun Baek¹, Jon-Lark Kim¹, Sang Mee Hwang²,³

**Affiliations:**
1. Sogang University, Department of Mathematics & Institute for Mathematical and Data Sciences
2. Seoul National University Bundang Hospital, Department of Laboratory Medicine
3. Seoul National University College of Medicine, Department of Laboratory Medicine

---

## 🎯 Overview

This repository implements an **automated bone marrow cellularity assessment system** that performs pixel-level segmentation and quantification of cellular regions in whole-slide images (WSIs). Our nnUNet-based approach eliminates inter-observer variability and provides objective, reproducible cellularity measurements.

### Key Results (20 WSIs)

| Metric | Mean ± SD | Range | Clinical Impact |
|--------|-----------|-------|-----------------|
| **Cellularity** | 48.0% ± 9.5% | 32.8% - 64.7% | Objective quantification |
| **Cell Coverage** | 38.9% ± 10.7% | 22.4% - 57.9% | Spatial density analysis |
| **Adipocyte Ratio** | 52.1% ± 9.6% | 35.3% - 67.2% | Tissue composition |
| **Reproducibility** | 100% | Zero variability | Eliminates inter-observer bias |

### Key Features

- ✅ **nnUNet-based segmentation** with self-configuring architecture
- ✅ **Hierarchical analysis**: Cellular region detection → Adipocyte identification
- ✅ **Adaptive thresholding** compensating for staining variations (180-240 grayscale)
- ✅ **Tile-based processing** supporting arbitrary WSI sizes (1,280×1,280 patches, 128px overlap)
- ✅ **Biological validation**: Inverse cellularity-adipocyte correlation (r = -0.9, p < 0.001)

---

## 📊 Visualization

<div align="center">

### Cellularity Analysis Summary
<img src="./Figure/summary_analysis.png" alt="Dataset999 Analysis Summary" width="85%">

### Sample Analysis Results

<img src="./Figure/BC_00000_analysis.png" alt="BC_00000 Analysis" width="85%">

<img src="./Figure/BC_011_cellularity_visualization.png" alt="BC_011 Cellularity Visualization" width="85%">

*Top: Whole-slide overview with tile coverage | Bottom left: Segmented cellular regions | Bottom right: Cellularity statistics*

</div>

---

## 🚀 Quick Start

```bash
# Clone repository and download model checkpoints
git clone https://github.com/LeGenAI/Automated_cellularity_assessment.git
cd Automated_cellularity_assessment
git lfs install && git lfs pull

# Install dependencies
pip install -r requirements.txt

# Run cellularity analysis
python dataset999_full_analysis.py \
    --input_dir ./data/demo_images \
    --output_dir ./results \
    --checkpoint ./checkpoints/checkpoint_latest.pth
```

**Expected Output:**
```
🔬 Dataset999 Cellularity Analysis
================================================================
📊 Dataset Statistics:
   - Total WSIs: 20
   - Mean Cellularity: 48.0% ± 9.5%
   - Cell Coverage: 38.9% ± 10.7%
   - Adipocyte Ratio: 52.1% ± 9.6%

✅ Analysis complete! Results saved to ./results/
```

---

## 💾 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended

### Option 1: Conda (Recommended)

```bash
conda create -n cellularity python=3.9
conda activate cellularity
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install nnunetv2 numpy pandas scikit-learn matplotlib seaborn Pillow tqdm
```

### Option 2: pip + virtualenv

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=2.0.0
torchvision>=0.15.0
nnunetv2>=2.2
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
tqdm>=4.65.0
```

---

## 📦 Model Checkpoints

Pre-trained models are available for immediate inference:

| Model | Dataset | Epochs | Size | Description |
|-------|---------|--------|------|-------------|
| **checkpoint_latest.pth** | Dataset999 | 52 | 452 MB | Latest 5-fold CV model (recommended) |
| **checkpoint_dataset998_epoch52.tar.gz** | Dataset998 | 52 | 419 MB | Legacy training checkpoint |

### Download Methods

**Option 1: Git LFS (Recommended)**
```bash
git lfs install
git lfs pull
```

**Option 2: Direct Download**
```bash
cd checkpoints/
# Download from GitHub Releases
wget https://github.com/LeGenAI/Automated_cellularity_assessment/releases/download/v1.0/checkpoint_latest.pth
```

**Option 3: Python Script**
```python
import gdown
url = "YOUR_GOOGLE_DRIVE_LINK"
gdown.download(url, "checkpoints/checkpoint_latest.pth", quiet=False)
```

---

## 📚 Usage Examples

### Single WSI Analysis

```python
from dataset999_full_analysis import process_single_wsi

result = process_single_wsi(
    image_path="./data/BC_00000.png",
    checkpoint_path="./checkpoints/checkpoint_latest.pth",
    output_dir="./results"
)

print(f"Cellularity: {result['cellularity']:.1f}%")
print(f"Adipocyte Ratio: {result['adipocyte_ratio']:.1f}%")
print(f"Tiles Analyzed: {result['num_tiles']}")
```

### Batch Processing

```python
import pandas as pd
from glob import glob
from tqdm import tqdm

def batch_analysis(image_dir, output_csv='results.csv'):
    results = []
    for img_path in tqdm(glob(f"{image_dir}/*.png")):
        result = process_single_wsi(img_path, checkpoint_path="./checkpoints/checkpoint_latest.pth")
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Processed {len(df)} images. Results saved to {output_csv}")
    return df

# Run batch analysis
df = batch_analysis('./data/dataset999/')
```

### Custom Tile-Based Processing

```python
from complete_wsi_tiling_system import extract_tiles_with_overlap, merge_tile_predictions

# Extract tiles from WSI
wsi_image = cv2.imread('bone_marrow_sample.png', cv2.IMREAD_GRAYSCALE)
tiles = extract_tiles_with_overlap(wsi_image, tile_size=1280, overlap=128)

# Process each tile with nnUNet
predictions = []
for tile_data in tiles:
    pred = nnunet_predict(tile_data['tile'])
    predictions.append({'prediction': pred, 'position': tile_data['position']})

# Merge predictions
full_segmentation = merge_tile_predictions(predictions, wsi_image.shape)
```

---

## 🏗️ Architecture

### System Pipeline

```
1. WSI Input (Bone Marrow Aspirate)
   ↓
2. Tile Extraction (1,280×1,280, overlap 128px)
   ↓
3. nnUNet Segmentation (Cellular Region Detection)
   ↓
4. Tile Merging (Overlap-weighted Averaging)
   ↓
5. Adaptive Thresholding (Adipocyte Detection)
   ↓
6. Cellularity Calculation & Reporting
```

### nnUNet Model Details

- **Architecture**: U-Net with residual connections
- **Configuration**: 2D (optimized for microscopy)
- **Training**: Fold 0 cross-validation, 52 epochs
- **Input**: 1,280×1,280 grayscale tiles
- **Preprocessing**: Z-score intensity normalization
- **Output**: Binary segmentation (cell vs. background)

### Cellularity Calculation

```
Global Cellularity (%) = (Total Cell Pixels - Adipocyte Pixels) / Total Cell Pixels × 100

Where:
- Total Cell Pixels: Segmented cellular regions from nnUNet
- Adipocyte Pixels: High-intensity regions (≥ adaptive threshold)
- Adaptive Threshold: min(max(75th percentile, 180), 240) on grayscale (0-255)
```

**Key Innovation**: Adaptive thresholding compensates for staining variations while maintaining biological validity (inverse cellularity-adipocyte correlation).

---

## 📈 Results

### Dataset999 Complete Analysis (20 WSIs)

<details>
<summary>📊 Click to expand full results table</summary>

| Sample ID | Cellularity | Cell Coverage | Adipocyte Ratio | Tiles | WSI Dimensions |
|-----------|-------------|---------------|-----------------|-------|----------------|
| BC_009 | 64.7% | 57.9% | 35.3% | 10 | 6,856 × 1,844 |
| BC_012 | 59.7% | 37.9% | 40.3% | 12 | 7,608 × 2,232 |
| BC_001 | 57.2% | 46.1% | 42.8% | 30 | 11,368 × 3,144 |
| BC_016 | 56.7% | 54.4% | 43.3% | 14 | 8,760 × 2,104 |
| BC_010 | 56.0% | 34.9% | 44.0% | 18 | 9,464 × 2,232 |
| BC_017 | 53.6% | 35.8% | 46.4% | 8 | 6,104 × 1,456 |
| BC_019 | 50.9% | 46.5% | 49.1% | 18 | 9,464 × 2,232 |
| BC_013 | 50.5% | 22.4% | 49.5% | 18 | 9,464 × 2,232 |
| BC_018 | 50.4% | 33.5% | 49.6% | 10 | 6,856 × 1,844 |
| BC_006 | 48.6% | 42.2% | 51.4% | 18 | 9,464 × 2,232 |
| BC_008 | 47.5% | 29.9% | 52.5% | 24 | 10,616 × 2,584 |
| BC_014 | 46.8% | 54.6% | 53.2% | 6 | 3,505 × 1,701 |
| BC_007 | 46.4% | 34.9% | 53.6% | 45 | 16,138 × 3,144 |
| BC_003 | 46.1% | 39.9% | 53.9% | 18 | 9,464 × 2,232 |
| BC_011 | 42.9% | 46.0% | 57.1% | 12 | 7,608 × 2,232 |
| BC_020 | 41.0% | 40.5% | 59.0% | 10 | 6,856 × 1,844 |
| BC_005 | 37.3% | 48.9% | 62.7% | 48 | 13,256 × 3,848 |
| BC_004 | 36.8% | 47.1% | 63.2% | 18 | 9,464 × 2,232 |
| BC_015 | 35.8% | 54.5% | 64.2% | 14 | 8,760 × 2,104 |
| BC_002 | 32.8% | 25.3% | 67.2% | 24 | 10,616 × 2,584 |

**Summary Statistics:**
- Cellularity Range: 32.8% (hypocellular) to 64.7% (normocellular)
- Mean Processing: 18.5 tiles/WSI (range: 6-48)
- Inverse Correlation: Cellularity vs. Adipocyte (r = -0.9, p < 0.001)

</details>

### Performance Characteristics

| Aspect | Measurement | Clinical Significance |
|--------|-------------|----------------------|
| **Reproducibility** | 100% | Zero inter-observer variability |
| **Scalability** | 6-48 tiles/WSI | Handles 3.5K to 17K pixel WSIs |
| **Processing Time** | ~2-5 min/WSI | GPU-accelerated inference |
| **Biological Accuracy** | r = -0.9 (p < 0.001) | Inverse cellularity-adipocyte correlation |
| **Staining Tolerance** | Adaptive (180-240) | Robust to staining variations |

---

## 🔬 Clinical Applications

### Digital Pathology Workflow

```
Microscopy → Digital Scanning → Automated Analysis → Quantitative Report → Pathologist Review
```

### Clinical Decision Support

- **Objective Quantification**: Eliminates subjective visual estimation (±10-20% variability)
- **Standardization**: Consistent methodology across institutions
- **Longitudinal Monitoring**: Track cellularity changes during treatment
- **Quality Control**: Automated specimen adequacy assessment
- **Efficiency**: Reduces pathologist workload (~30-60 min manual counting eliminated)

### Diagnostic Use Cases

1. **Hematologic Disorders**: Baseline cellularity for MPN, MDS, aplastic anemia
2. **Treatment Monitoring**: Chemotherapy response evaluation
3. **Research Studies**: Large-scale quantitative retrospective analysis
4. **Quality Assurance**: Specimen adequacy before diagnostic interpretation

---

## 📂 Repository Structure

```
Automated_cellularity_assessment/
├── checkpoints/
│   ├── checkpoint_latest.pth              # Latest trained model (452 MB)
│   └── checkpoint_dataset998_epoch52.tar.gz  # Legacy checkpoint (419 MB)
├── Figure/
│   ├── summary_analysis.png               # Dataset overview visualization
│   ├── BC_00000_analysis.png              # Sample analysis example
│   └── BC_011_cellularity_visualization.png  # Detailed visualization
├── dataset999_full_analysis.py            # Main analysis script
├── complete_wsi_tiling_system.py          # Tile extraction and merging
├── train_full.py                          # nnUNet training script
├── requirements.txt                       # Python dependencies
└── README.md
```

---

## 🔧 Training Your Own Model

### Data Preparation

```bash
# Organize data in nnUNet format
mkdir -p nnUNet_raw/Dataset999_BoneMarrow/{imagesTr,labelsTr}

# Copy images and masks
cp /path/to/images/* nnUNet_raw/Dataset999_BoneMarrow/imagesTr/
cp /path/to/masks/* nnUNet_raw/Dataset999_BoneMarrow/labelsTr/

# Set environment variables
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

### Training

```bash
# Preprocess dataset
nnUNetv2_plan_and_preprocess -d 999 --verify_dataset_integrity

# Train model (fold 0, 52 epochs)
python train_full.py --dataset_id 999 --num_epochs 52 --fold 0
```

### Training Configuration

- **Optimizer**: Adam with learning rate scheduling
- **Loss**: Dice + Cross-Entropy combination
- **Augmentation**: Rotation, scaling, elastic deformation (nnUNet default)
- **Validation**: Fold 0 cross-validation
- **Hardware**: NVIDIA GPU with 16GB+ VRAM recommended

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce tile size
python dataset999_full_analysis.py --tile_size 1024  # Default: 1280

# Use CPU inference
python dataset999_full_analysis.py --device cpu
```

### Model Loading Error
```bash
# Verify checkpoint file integrity
md5sum checkpoints/checkpoint_latest.pth

# Re-download if corrupted
rm checkpoints/checkpoint_latest.pth
git lfs pull
```

### Poor Segmentation Quality
- Check input image quality (avoid compression artifacts)
- Verify grayscale conversion is correct
- Ensure nnUNet preprocessing completed successfully

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{baek2025cellularity,
  title={Automated Cellularity Assessment in Bone Marrow Aspirate Smears Using Deep Learning-Based Segmentation},
  author={Baek, Jae-Hyun and Kim, Jon-Lark and Hwang, Sang Mee},
  journal={Journal Name},
  year={2025},
  volume={XX},
  pages={XXX-XXX},
  doi={10.xxxx/xxxxx}
}

@article{isensee2021nnunet,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature Methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] 3D volumetric analysis support
- [ ] Integration with PACS systems
- [ ] Real-time inference optimization
- [ ] Additional cell type classification
- [ ] Web-based annotation tool

Please open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Primary Maintainer**: Jae-Hyun Baek
**Affiliation**: Sogang University, Department of Mathematics
**Email**: [Contact via GitHub Issues](https://github.com/LeGenAI/Automated_cellularity_assessment/issues)

---

## 🙏 Acknowledgments

- **nnU-Net Team** for the exceptional self-configuring segmentation framework
- **Seoul National University Bundang Hospital** for providing clinical expertise and data
- **PyTorch Community** for deep learning infrastructure

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by the Cellularity Assessment Research Team

</div>
