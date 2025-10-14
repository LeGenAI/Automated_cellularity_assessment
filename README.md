# Automated Cellularity Assessment in Bone Marrow Aspirate Smears

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

Automated bone marrow cellularity quantification system using nnUNet deep learning architecture for whole-slide image analysis of aspirate smears.

## 📄 Paper

**Title:** Automated Cellularity Assessment in Bone Marrow Aspirate Smears Using Deep Learning-Based Segmentation

**Authors:** Jae-Hyun Baek, Jon-Lark Kim, Sang Mee Hwang

**Affiliations:**
- Sogang University, Department of Mathematics & Institute for Mathematical and Data Sciences
- Seoul National University Bundang Hospital, Department of Laboratory Medicine
- Seoul National University College of Medicine, Department of Laboratory Medicine

## 🎯 Overview

This repository contains the implementation of an automated bone marrow cellularity assessment system that:
- Performs pixel-level segmentation of cellular regions in whole-slide images (WSIs)
- Distinguishes between normal hematopoietic cells and adipocytes
- Provides objective, reproducible cellularity quantification
- Eliminates inter-observer variability in manual estimation

## 🔬 Key Results

- **Dataset:** 20 whole-slide images of bone marrow aspirate smears
- **Mean Cellularity:** 48.0% ± 9.5% (range: 32.8-64.7%)
- **Cell Coverage:** 38.9% ± 10.7% (range: 22.4-57.9%)
- **Adipocyte Ratio:** 52.1% ± 9.6% (range: 35.3-67.2%)

## 🏗️ Architecture

- **Base Model:** nnUNet v2 (self-configuring deep learning framework)
- **Training:** Dataset998, fold 0, 52 epochs
- **Processing:** Tile-based approach with 1,280×1,280 patches and 128-pixel overlap
- **Segmentation:** Hierarchical analysis (cellular regions → adipocyte detection)

## 📁 Repository Structure

```
.
├── checkpoints/                    # Pre-trained model checkpoints
│   ├── checkpoint_dataset998_epoch52.tar.gz
│   └── checkpoint_latest.pth
├── custom_scripts/                 # Custom processing scripts
│   └── complete_wsi_tiling_system.py  # WSI tiling and merging system
├── Figure/                         # Figures used in the paper
│   ├── BC_00000_analysis.pdf      # Representative sample analysis
│   ├── BC_011_cellularity_visualization.pdf
│   └── summary_analysis.pdf        # Dataset summary statistics
├── dataset999_full_analysis.py    # Full WSI analysis pipeline
├── gpu_proper_inference_debug.py  # Inference with debugging
└── README.md                       # This file
```

**Note:** nnUNet framework should be installed separately following the installation instructions below.

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/LeGenAI/Automated_cellularity_assessment.git
cd Automated_cellularity_assessment

# Install nnUNet
pip install nnunetv2

# Install additional dependencies
pip install torch torchvision
pip install numpy pillow matplotlib
pip install opencv-python scikit-image
```

## 💻 Usage

### 1. Download Pre-trained Model

Extract the checkpoint:
```bash
cd checkpoints
tar -xzf checkpoint_dataset998_epoch52.tar.gz
```

### 2. Prepare Your Data

Organize your whole-slide images following nnUNet conventions:
```
input_data/
└── imagesTs/
    ├── sample_0000.png
    ├── sample_0001.png
    └── ...
```

### 3. Run Inference

```bash
python dataset999_full_analysis.py \
    --input_dir ./input_data/imagesTs \
    --output_dir ./output_results \
    --checkpoint ./checkpoints/checkpoint_latest.pth
```

### 4. Analyze Results

The output directory will contain:
- Segmentation masks (PNG format)
- Cellularity metrics (JSON format)
- Visualization overlays (PNG format)
- Summary statistics (CSV format)

## 📊 Output Metrics

For each WSI, the system provides:

1. **Global Cellularity (%)**: Ratio of normal hematopoietic cells to total cellular area
2. **Cell Coverage (%)**: Proportion of image occupied by cellular material
3. **Adipocyte Ratio (%)**: Proportion of adipocytes within cellular regions
4. **Quantitative Measurements**: Pixel counts, area measurements, confidence scores

## 🔬 Methodology

### Tile-based Processing
- **Tile Size:** 1,280 × 1,280 pixels
- **Overlap:** 128 pixels (10%)
- **Merging:** Overlap-weighted averaging with 0.5 threshold

### Cellularity Calculation

```
Cellularity = (Total Cell Pixels - Adipocyte Pixels) / Total Cell Pixels
```

### Adipocyte Detection
- Adaptive intensity-based thresholding
- Threshold: 75th percentile of cellular region intensities
- Bounds: 180-240 on grayscale (0-255)

## 📈 Performance

- Successfully analyzed 20 diverse clinical specimens
- Robust across varying:
  - Image dimensions (3,505×1,701 to 16,994×3,535 pixels)
  - Staining intensities
  - Cell densities
  - Morphological characteristics

## 🔐 Data Privacy

**Important:** This repository does NOT include any patient data or original whole-slide images due to privacy regulations and IRB restrictions.

- Raw clinical data is protected and not shared
- Figure visualizations use anonymized, de-identified samples
- Demo dataset available upon reasonable request

## 📝 Citation

If you use this code or method in your research, please cite:

```bibtex
@article{baek2025automated,
  title={Automated Cellularity Assessment in Bone Marrow Aspirate Smears Using Deep Learning-Based Segmentation},
  author={Baek, Jae-Hyun and Kim, Jon-Lark and Hwang, Sang Mee},
  journal={[Journal Name]},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

- **Jae-Hyun Baek:** jhbaek@sogang.ac.kr
- **Jon-Lark Kim:** jlkim@sogang.ac.kr
- **Sang Mee Hwang:** sangmee1@snu.ac.kr

## 🙏 Acknowledgments

This study was supported by:
- Seoul National University Bundang Hospital (SNUBH) research fund (grant number 02-2021-0051)
- BK21 FOUR (Fostering Outstanding Universities for Research) funded by the Ministry of Education (MOE, Korea) and National Research Foundation of Korea (NRF) under Grant No. 4120240415042

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚖️ Ethics

All experimental protocols were approved by the Seoul National University Bundang Hospital Institutional Review Board (IRB) (approval number: B-2401-876-104). All patient data were anonymized to protect patient privacy.

## 🔗 Related Resources

- [nnUNet](https://github.com/MIC-DKFZ/nnUNet) - Self-configuring method for deep learning-based biomedical image segmentation
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

**Note:** For access to demo datasets or additional information, please contact the corresponding author.
