# Regime-Aware Flux Estimation over Heterogeneous Surfaces

Code repository for the paper: **"Regime-aware and Physically Guided Learning for Surface Turbulent Flux Estimation over Heterogeneous Land Surface"**

## Overview

This repository provides **a minimal reproducible implementation** of a physics-informed machine learning framework for estimating surface turbulent fluxes (e.g., friction velocity u*) over heterogeneous land surfaces.

The proposed method integrates:

1. **Gaussian Mixture Models (GMM)** for regime-aware decomposition of atmospheric conditions
2. **Self-attention neural networks** for modeling nonlinear interactions among meteorological variables
3. **MOST-informed loss functions** incorporating Monin-Obukhov Similarity (MOST) as a physical constraint

## Repository Structure

```
code/
├── main.m                          % Main entry point (proposed method)
├── custom_loss/                    % Loss functions (Huber + MOST-informed)
│   ├── get_lambda_PI.m
│   └── myloss.m
├── data_prepare/                   % Observation calculation and cleaning
│   ├── data_clean.m
│   └── obs_calculation.m
├── experiments/
│   ├── reproduce_minimal_comparison.m
│   └── run_regime_model.m
├── gmm/                            % GMM-based regime identification
│   └── gmm_identification.m
├── models/                         % MLP_SA_PI (PI-GSAM) model
│   └── get_prediction_MLP_SA_PI.m
├── physics/                        % MOST-related calculations
│   ├── mo_calculation.m
│   ├── derived.py
│   └── mo.py
├── plots/                  
│   ├── plot_example_results.m      % Main result visualization
│   └── plot_flux_distributions.m   % Optional dataset distribution plot
├── utils/
│   ├── py_init_mo.m                % Python environment initialization
│   ├── struct2table_if_needed.m
│   └── validate_target.m
data/
├── sample/
│   ├── example_data_train.mat
│   └── example_data_test.mat
result/
├── example_result_ustar.png        % Example output
requirements.txt
environment.yml
README.md
```

## Data

### Example Dataset

This repository provides a reduced and anonymized example dataset:

- `data/sample/example_data_train.mat`
- `data/sample/example_data_test.mat`

The dataset:

- preserves key statistical characteristics of the original observations
- is sufficient to reproduce the modeling workflow
- is randomly shuffled and partially sampled to ensure data privacy

### Data Variables

The data files should contain a table with the following variables:

| Variable | Description | Unit |
|----------|-------------|------|
| `Ta_10M_Avg`, `Ta_35M_Avg` | Air temperature at 10m and 35m | °C |
| `WS_10M`, `WS_35M` | Wind speed at 10m and 35m | m/s |
| `U_10M`, `U_35M` | U component of wind speed at 10m and 35m | m/s |
| `V_10M`, `V_35M` | V component of wind speed at 10m and 35m | m/s |
| `RH_10M_Avg`, `RH_35M_Avg` | Relative humidity at 10m and 35m | % |
| `P` | Pressure | hPa |
| `DR_Avg`, `UR_Avg` | Downward/upward shortwave radiation | W/m² |
| `DLR_Avg`, `ULR_Avg` | Downward/upward longwave radiation | W/m² |
| `Tau_30` | Momentum flux (surface stress) | kg/(m·s²) |
| `Hs_30` | Sensible heat flux | W/m² |
| `LE_30` | Latent heat flux | W/m² |

**Note:** The tower data used in this study is from the Huainan (HCEO) flux tower site.

### Reproducibility Note

The full dataset used in the manuscript is not publicly available due to data-sharing restrictions.
The provided dataset is a representative subset rather than the original dataset. Therefore, numerical results will not exactly match those reported in the paper. 

The repository is designed for:
- **methodological reproducibility** (workflow, model behavior, relative performance)

It is not intended for exact reproduction of all figures in the manuscript which depend on the full dataset. 
In particular, regime identification via GMM depends on the data distribution, and therefore the identified regimes should be interpreted as statistical representations rather than exact physical counterparts.

## Requirements

### MATLAB Version
- MATLAB R2022b or later (tested on R2025a)

### Required Toolboxes
- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox

### Python (auxiliary)
A lightweight Python module is used for MOST-related computations.

Two setup options are provided:

- **Option 1 (recommended)**

```
conda env create -f environment.yml
```

- **Option 2:**

```
pip install -r requirements.txt
```

Python is invoked from MATLAB via 

```
pyenv("ExecutionMode", "OutOfProcess")
```

## Input Features (10 dimensions)

The model uses 10 physically motivated input features derived from multi-level meteorological observations:

1. Air temperature at 35m (Ta)
2. Temperature gradient (∂T/∂z)
3. Wind speed at 35m (WS)
4. Wind speed gradient (∂WS/∂z)
5. Relative humidity at 35m (RH)
6. RH gradient (∂RH/∂z)
7. Pressure (P)
8. Net radiation (Rn)
9. Solar altitude angle (SAA)
10. Bulk Richardson number (Rib)

These features are designed to capture both local atmospheric conditions and stability-related processes, which are essential for flux estimation.

## Usage

### Quick Start

1. Navigate to the `code/` directory
2. Ensure the dataset exists in `data/sample/`
3. Run:

```matlab
main
```

### Main Script (main.m):

The main script implements the **proposed method**:

1. Load example dataset
2. Compute observation-derived variables
3. Apply MOST-based calculation
4. Construct input features
5. Perform GMM-based regime identification
6. Train Regime-aware PI-GSAM model
7. Evaluate model performance
8. Generate result plots

Default target variable:

```
target = "ustar"
```

Other targets can be specified manually:

```
main(struct('target',"tstar"))
main(struct('target',"qstar"))
```

### Minimal Comparison

To reproduce the core comparison:

```
reproduce_minimal_comparison
```

This script evaluates four configurations:

1. Single model (no GMM, no PI)
2. Single model + PI
3. Regime-aware model (GMM only)
4. Regime-aware + PI (proposed)

## Model Summary

### Architecture

```
Input (10 features)
    ↓
Self-Attention Layer (feature interaction modeling)
    ↓
Fully Connected (nonlinear regression)
    ↓
Fully Connected (nonlinear regression)
    ↓
Output (flux prediction)
```

### Physics-Informed Loss Function

```
L = L_Huber + λ × L_MOST
```

where:
- `L_Huber`: data-driven loss
- `L_MOST`: physical constraint from MOST
- `λ`: regularization weight

**Implementation note:**
The regularization weight λ is target- and data-dependent. Conservative small values are preferred for robustness.

## Visualization

- `plot_example_results.m`: used in the main workflow.
- `plot_flux_distributions.m`: optional function for inspecting dataset distributions and not required for reproducing the main results.

## Example Output

The repository includes:

- `result/example_result_ustar.png`

This figure illustrates a representative result generated using the example dataset. 
Running `main.m` will reproduce a similar output.

## Data Availability

The example dataset used in this repository is provided in the `data/sample/` directory.

The full observational dataset used in the manuscript is not publicly available due to data-sharing restrictions, but may be available from the corresponding author upon reasonable request.

## Code Availability

The code used in this study will be made publicly available upon publication.

## Citation

If you use this code, please cite:

```bibtex
@article{author2026regime,
  title={Regime-aware and Physically Guided Learning for Surface Turbulent Flux Estimation over Heterogeneous Land Surface},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2026}
}
```

## License

[Specify your license here]

## Contact

For questions, please contact the corresponding author.
