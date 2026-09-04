# Collaborative Identity-Guided Generative Total Correlation Learning for Multi-Modal Recommendation
This repository contains the official implementation of the model described in the paper "Collaborative Identity-Guided Generative Total Correlation Learning for Multi-Modal Recommendation". 
The framework is designed with a modular architecture to facilitate reproducible experiments in multimodal recommendation.


## 🗂️ Repository structure

```
.
├── preprocessing              # code for dataset preprocessing
├── src/
│   ├── common/                # Core utilities and base abstractions
│   │   ├── abstract_recommender.py  # Base class defining the recommender interface
│   │   ├── encoders.py        # Implementation of neural encoding layers
│   │   ├── loss.py            # Custom loss functions and optimization criteria
│   │   └── trainer.py         # Standardized training loops and evaluation logic
│   ├── configs/               # Hyperparameter and environment configurations
│   │   ├── dataset/           # Dataset-specific settings (e.g., baby, cell, sports)
│   │   └── model/             # Model-specific hyperparameters (e.g., GTC, mg)
│   ├── models/                # Architecture implementations
│   │   └── GTC.py             # Main implementation of the proposed model
│   ├── utils/                 # Auxiliary helper functions and data loaders
│   ├── main.py                # Primary execution entry point
│   └── run_gtc.sh             # Shell script for automated testing and inference
└── requirements.txt           # Environment dependencies
```


## ⚙️ Installation

To ensure a reproducible environment, we recommend using a virtual environment (e.g., Conda or `venv`).

### Prerequisites

* Python >= 3.8
* CUDA-enabled GPU (recommended for training)
* numpy>=1.21.5
* pandas>=1.3.5
* python>=3.8
* scipy>=1.7.3
* torch>=2.0.5
* pyyaml>=6.0
* matplotlib>=3.5.2
* torchvision>=0.15.2
* torchaudio>=2.0.2
* torch_geometric>=2.0.4
* scikit-learn>=1.0.0

### Setup Steps

1. **Clone the repository**:
```bash
git clone jingdu-cs/GTC.git
cd GTC

```


2. **Install dependencies**:
Install the required packages as specified in the `requirements.txt` file:
```bash
pip install -r requirements.txt

```

3. **Data**:

   * Download the rating and metadata files from the [Amazon Review Data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) repository.
   * Then preprocess the data step by step using the scripts in the `preprocessing` folder in the following order:

     `0rating.py` → `1split.py` → `2reindex.py` → `3feature.py` → `dualgnn-gen-u-u-matrix.py`

---

## 📖 Usage (Quick Start)

To verify the installation and run the model on the "Sports" dataset using the provided shell script `src/run_gtc.sh`.

---
