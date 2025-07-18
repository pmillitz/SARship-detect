# SARFish Environment Configuration

 ***environment.yml* file for HPC deployment:**

```yaml
# name: SARFish
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
 - python=3.9
  - jupyterlab
  - pandas
  - numpy=1.26.4 for compatibility with ultralytics
  - matplotlib
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=12.4 # latest version on Kaya
  - scipy
  - scikit-learn
  - seaborn
  - gdal
  - pyyaml
  - ipython
  - ipywidgets
  - ipykernel
  - typing_extensions  # for enhanced typing support
  - qt
  - pyqt
  - nodejs  # optional: to suppress jupyter error messages 
  - vispy
  - "libstdcxx-ng=12"  # for strict Kaya compatibility
```

## Setup Instructions

### Prerequisites

1. Access to conda/anaconda on the HPC system
2. Write permissions in your user directory

### Installation Steps

*As a rule, use conda for environment activation and use mamba for package management (see section on Recommended general approach below.)*

1. **Install mamba (optional but recommended for faster dependency resolution):**
   
   ```bash
   conda create -n tools mamba -c conda-forge
   conda activate tools
   ```

2. **Create the SARFish environment:**
   
   ```bash
   # Create environment at specific path
   mamba env create -p /path/to/your/envs/SARFish -f environment.yml
   
   # Or with conda:
   conda env create -p /path/to/your/envs/SARFish -f environment.yml
   ```

3. **Fix execute permissions (HPC systems only):**
   
   ```bash
   conda activate SARFish
   chmod +x ~/path/to/env/SARFish/bin/*
   ```

4. Install `ultralytics` via pip
   
   ```bash
   pip install ultralytics  # Pip pulls fewer conflicting dependencies
   ```

5. **Register Jupyter kernel:**
   
   ```bash
   conda activate SARFish
   python -m ipykernel install --user --name SARFish --display-name "Python (SARFish)"
   ```

6. **Test the installation:**
   
   ```bash
   conda activate SARFish
   python -c 'import pandas, numpy, matplotlib, torch, vispy, ultralytics; print("All packages imported successfully!")'
   jupyter lab
   ```

### Key Packages Included

- **Data Science:** pandas, numpy, scipy, scikit-learn, matplotlib, seaborn
- **Deep Learning:** pytorch, torchvision, torchaudio, ultralytics
- **Geospatial:** gdal (for osgeo)
- **Visualisation:** matplotlib, seaborn, vispy
- **Development:** jupyterlab, ipython, ipywidgets
- **System:** qt, pyqt (for GUI backends), nodejs (for JupyterLab extensions)

### Troubleshooting Notes

- The `libstdcxx-ng<12` constraint fixes GLIBCXX compatibility issues on older HPC systems
- If you get "Permission denied" errors, run the chmod command in step 3
- Make sure to select the "Python (SARFish)" kernel in JupyterLab
- Node.js warnings about worker_threads are cosmetic and don't affect functionality

### Updating the Environment

```bash
conda activate SARFish
conda update --all
# or
mamba update --all
```

### Recommended general approach (avoids shell problem) :

```bash
# Use conda for environment activation
conda activate tools     # Get access to mamba
conda activate SARFish   # Switch to your work environment

# Use mamba for package management
mamba install package_name
mamba update --all
mamba env create -f environment.yml
```
