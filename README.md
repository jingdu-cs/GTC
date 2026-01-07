# Conditional Generative Total Correlation Learning for Multi-Modal Recommendation
This repository contains the official implementation of the model described in the paper "Conditional Generative Total Correlation Learning for Multi-Modal Recommendation". 
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
│   │   └── mmr_dragon.py      # Main implementation of the proposed model
│   ├── utils/                 # Auxiliary helper functions and data loaders
│   ├── main.py                # Primary execution entry point
│   └── test-sports.sh         # Shell script for automated testing and inference
└── requirements.txt           # Environment dependencies
```


## ⚙️ Installation

To ensure a reproducible environment, we recommend using a virtual environment (e.g., Conda or `venv`).

### Prerequisites

* Python >= 3.8
* CUDA-enabled GPU (recommended for training)

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


---

## 3. Usage (Quick Start)

To verify the installation and run the model on the "Sports" dataset using the provided shell script `run_gtc.sh`.

---
