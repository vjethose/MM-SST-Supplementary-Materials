# MM-SST: Multi-Modal Self-Supervised Transformer for Energy Forecasting

Complete supplementary materials for reproducible research.

**Paper**: MM-SST: A Multi-Modal Self-Supervised Transformer for Energy Consumption Forecasting with Graph Attention Networks

**Authors**: Jethose VIJAYAKUMAR, Manoharan SUBRAMANIAN

---

## Contents

### Core Implementation
- **`mm_sst.py`** – Complete PyTorch implementation of the MM-SST model with all components:
  - Multi-modal embedding layer
  - Temporal patching module
  - Positional encoding
  - Graph Attention Networks (GAT)
  - Transformer encoder with integrated GAT
  - Self-supervised pre-training with masked reconstruction
  - Supervised forecasting head

- **`sim_data.py`** – Synthetic dataset generation:
  - SIM-D: Urban district (50 buildings, daily, 5 years)
  - SIM-B: Single building zones (8 zones, hourly, 100 days)
  - Realistic demand patterns with autocorrelation, seasonality, weather coupling
  - Spatial structure and relational dependencies

- **`train.py`** – Two-stage training pipeline:
  - Stage 1: Self-supervised pre-training with masked patch reconstruction
  - Stage 2: Supervised fine-tuning for forecasting
  - Early stopping and learning rate scheduling
  - Comprehensive logging and checkpointing

- **`evaluate.py`** – Evaluation and analysis tools:
  - Performance metrics (MAE, RMSE, MAPE)
  - Ablation studies
  - Sensitivity analysis
  - Reproducibility documentation

### Configuration Files
- **`config_sim_d.json`** – Model and training configuration for SIM-D
- **`config_sim_b.json`** – Model and training configuration for SIM-B
- **`requirements.txt`** – Python package dependencies

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Datasets
```bash
python sim_data.py --output-dir ./data
```

This generates:
- `sim_d_dataset.csv` (50 buildings × 1,825 days)
- `sim_d_graph.json` (k-NN adjacency for GAT)
- `sim_b_dataset.csv` (8 zones × 2,400 hours)
- `sim_b_graph.json` (grid adjacency for GAT)

### 3. Train MM-SST on SIM-D
```bash
python train.py \
  --dataset sim_d \
  --data-path ./data/sim_d_dataset.csv \
  --output-dir ./results_sim_d \
  --batch-size 32 \
  --pretrain-epochs 8 \
  --finetune-epochs 15
```

### 4. Train MM-SST on SIM-B
```bash
python train.py \
  --dataset sim_b \
  --data-path ./data/sim_b_dataset.csv \
  --output-dir ./results_sim_b \
  --batch-size 64 \
  --pretrain-epochs 4 \
  --finetune-epochs 10
```

### 5. Evaluate and Generate Reports
```bash
python evaluate.py \
  --model-path ./results_sim_d/best_model.pt \
  --config-path ./results_sim_d/config.json \
  --dataset sim_d \
  --output-dir ./results_sim_d
```

---

## Model Architecture

### Components

1. **Multi-Modal Embedding**
   - Separate embeddings for consumption, weather, contextual data
   - Fusion in shared latent space via non-linear transformation
   - Preserves domain-specific structure before temporal modeling

2. **Temporal Patching**
   - Groups consecutive time steps into patches
   - Reduces sequence length from L to L/S tokens
   - Preserves local temporal continuity

3. **Transformer Encoder**
   - Multi-head self-attention for long-range dependencies
   - 2 layers × 4 heads (configurable)
   - Residual connections + layer normalization

4. **Graph Attention Networks (GAT)**
   - Integrated within Transformer encoder
   - Models relational dependencies among nodes
   - Learned attention coefficients per attention head
   - Supports flexible graph definitions (k-NN, grid, spatial, etc.)

5. **Self-Supervised Pre-Training**
   - Masked patch reconstruction objective
   - Masking ratio: 0.3 (configurable)
   - Reduces dependence on labeled data

6. **Forecasting Head**
   - Linear projection from embedding dim to 1
   - MSE + MAE combined loss for robustness

---

## Dataset Specifications

### SIM-D: Urban District Energy Demand (UDED)
```
Buildings:  50
Days:       1,825 (5 years)
Resolution: Daily
Horizon:    7-day ahead
Lookback:   30 days
Features:
  - Consumption: 50 (one per building)
  - Weather: 3 (temperature, humidity, solar irradiance)
  - Temporal: 2 (month, day-of-week)
Graph: k-NN with k=5 nearest neighbors
Splits: 70% train / 15% validation / 15% test
```

### SIM-B: Single Building Zone Energy (SBEC)
```
Zones:      8
Hours:      2,400 (100 days)
Resolution: Hourly
Horizon:    24-hour ahead
Lookback:   168 hours (1 week)
Features:
  - Consumption: 8 (one per zone)
  - Weather: 3 (temperature, humidity, solar irradiance)
  - Temporal: 2 (hour-of-day, day-of-week)
  - Occupancy: 8 (WiFi signal strength, ~15% missing)
Graph: 2×4 grid adjacency (physically adjacent zones)
Splits: 70% train / 15% validation / 15% test
```

---

## Hyperparameters

### Default Model Configuration
```json
{
  "embedding_dim": 32,
  "fusion_dim": 64,
  "patch_size": 7,
  "num_transformer_layers": 2,
  "n_heads": 4,
  "ff_dim": 64,
  "gat_heads": 2,
  "dropout": 0.1
}
```

### Training Configuration (SIM-D)
```json
{
  "batch_size": 32,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "pretrain_epochs": 8,
  "finetune_epochs": 15,
  "masking_ratio": 0.3,
  "patience": 5
}
```

### Training Configuration (SIM-B)
```json
{
  "batch_size": 64,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "pretrain_epochs": 4,
  "finetune_epochs": 10,
  "masking_ratio": 0.3,
  "patience": 5
}
```

---

## Expected Results

### SIM-D (Urban District, 7-day forecast)
| Model | MAE (MWh) | RMSE (MWh) | MAPE (%) |
|-------|-----------|-----------|----------|
| MM-SST | 0.330 ± 0.008 | 0.455 ± 0.011 | 13.1 ± 0.6 |
| SVR (baseline) | 0.256 ± 0.004 | 0.353 ± 0.006 | 10.2 ± 0.3 |
| LSTM (baseline) | 0.294 ± 0.010 | 0.406 ± 0.014 | 11.7 ± 0.5 |

### SIM-B (Building Zones, 24-hour forecast)
| Model | MAE (kWh) | RMSE (kWh) | MAPE (%) |
|-------|-----------|-----------|----------|
| MM-SST | 0.327 ± 0.014 | 0.451 ± 0.019 | 13.0 ± 0.7 |
| SVR (baseline) | 0.293 ± 0.006 | 0.404 ± 0.008 | 11.6 ± 0.3 |
| LSTM (baseline) | 0.319 ± 0.013 | 0.440 ± 0.017 | 12.7 ± 0.6 |

---

## Ablation Study Results

### SIM-D Component Contributions
| Ablation | MAE Change | % Change |
|----------|-----------|----------|
| Full Model | 0.330 | — |
| No GAT | +0.004 | +1.2% |
| No Patching | +0.007 | +2.1% |
| No Weather | +0.014 | +4.3% |
| No Context | +0.011 | +3.4% |

**Key Finding**: Weather data and contextual features drive most of the improvement; graph attention contributes minimally on weakly-coupled synthetic data.

---

## Reproducibility

### Random Seeds
All experiments use fixed seeds for reproducibility:
- NumPy: 42
- PyTorch: 42
- Python: 42

### Hardware
- Tested on: CPU (2 cores) and GPU (NVIDIA A100 recommended for faster training)
- Memory: ~2-4 GB for SIM-D, ~1-2 GB for SIM-B
- Training time: ~5-10 minutes per dataset (on GPU)

### Verification Checklist
- ✅ Dataset generation produces consistent files with fixed seeds
- ✅ Model instantiation and forward pass work correctly
- ✅ Pre-training converges within 8-10 epochs
- ✅ Fine-tuning achieves expected performance within 15 epochs
- ✅ Evaluation metrics match reported values ±0.5%

---

## File Structure

```
.
├── mm_sst.py                 # Model implementation
├── sim_data.py               # Dataset generation
├── train.py                  # Training pipeline
├── evaluate.py               # Evaluation tools
├── config_sim_d.json         # SIM-D configuration
├── config_sim_b.json         # SIM-B configuration
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── data/                     # Generated datasets (after running sim_data.py)
    ├── sim_d_dataset.csv
    ├── sim_d_graph.json
    ├── sim_d_spatial.csv
    ├── sim_d_metadata.json
    ├── sim_b_dataset.csv
    ├── sim_b_graph.json
    └── sim_b_metadata.json
```

---

## Advanced Usage

### Custom Hyperparameter Tuning
```bash
python train.py \
  --dataset sim_d \
  --data-path ./data/sim_d_dataset.csv \
  --embedding-dim 64 \
  --patch-size 14 \
  --num-layers 3 \
  --n-heads 8 \
  --learning-rate 5e-4 \
  --batch-size 16
```

### Run with Different Graph Structures
Modify `load_graph()` in `train.py` to load alternative adjacency matrices:
- **Spatial proximity**: Distance-based edges
- **Demand correlation**: Pearson correlation edges
- **Learned similarity**: Learn edge weights from data

### Extend to Real Datasets
Replace `sim_data.py` dataset with real building energy data:
1. Prepare CSV with columns: [Date, Consumption_*nodes*, Temperature, Humidity, SolarIrradiance, Month, DayOfWeek]
2. Define graph structure (spatial locations or feeder connectivity)
3. Run training pipeline as above

---

## References

### Key Papers
- Vaswani et al. (2017) – Attention Is All You Need
- Wu et al. (2021) – Autoformer: Decomposition Transformers with Auto-Correlation
- Nie et al. (2023) – A Time Series is Worth 64 Words (PatchTST)
- Veličković et al. (2018) – Graph Attention Networks

### Energy Forecasting Literature
- Hong & Fan (2016) – Probabilistic Electric Load Forecasting
- Deb & Schlueter (2021) – Data-Driven Energy Modelling Techniques
- Ahmad & Chen (2019) – Nonlinear Autoregressive and Random Forest Approaches

---

## Contact & Support

**Corresponding Author**: Jethose VIJAYAKUMAR  
**Email**: vjithosh@gmail.com  
**Institution**: Department of AI & Data Science, JCT College of Engineering

For questions about implementation, hyperparameters, or dataset generation, please refer to the relevant source file or the paper's methodology section.

---

## License

This code is provided for research and educational purposes. Please cite the paper if you use this implementation in your work.

```bibtex
@article{vijayakumar2024mmsst,
  title={MM-SST: A Multi-Modal Self-Supervised Transformer for Energy Consumption Forecasting with Graph Attention Networks},
  author={VIJAYAKUMAR, Jethose and SUBRAMANIAN, Manoharan},
  year={2024}
}
```

---

**Last Updated**: October 2026  
**Status**: Ready for Submission to Applied Energy
