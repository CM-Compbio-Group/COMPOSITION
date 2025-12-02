# COMPOSITION

COMPOSITION is a hierarchical probabilistic model for single-cell spatial transcriptomics that represents tissues at multiple scales, from individual cell types to local niches and higher-order domains. We, as Cong Ma Research Group, implement this hierarchy using two coupled variational autoencoder (VAE) networks whose encoder–decoder pairs link molecular profiles to latent variables at each layer, and we show that their training objective is equivalent to variational inference in our generative model with explicit conditional dependencies between adjacent layers. This formulation makes the multi-layer spatial structure of the tissue explicit and numerically stable, and GPU-accelerated inference scales to ~0.5M cells in about 18 minutes (e.g., a Visium HD mouse brain sample), while achieving domain inference accuracy comparable to state-of-the-art methods. By combining the hierarchical model with computer-vision–inspired postprocessing, COMPOSITION recovers biologically meaningful co-localized cell-type groups and continuous spatial gradients of cell-type composition that are often missed by standard clustering-based spatial domain identification approaches.

<br>

<img width="969" height="545" src="https://github.com/CM-Compbio-Group/COMPOSITION/blob/main/overview.png" />

<br>

_**Planned Publication**_

* _Park, J., Zhang, T., Ma, C. (2025) COMPOSITION: Cell type and spatial Organization Modeling using scalable Probabilistic Optimization of Spatially Informed Topic Inference Of Niches. Submitted to RECOMB 2026_ 


## Installation

```bash
conda env create -f environment.yml
conda activate minibatch
```

Alternatively, you can follow these steps:

```bash
conda create -n minibatch python=3.10
conda activate minibatch
pip install ipykernel
python -m ipykernel install --user --name minibatch --display-name minibatch

pip install torch==2.1.0+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

pip install pandas scanpy squidpy cellcharter

# (Option) you may want to reinstall numpy
pip uninstall numpy
pip install numpy==1.26.4
```

## How to Use

* Please refer to `tutorial/COMPOSITION_mouse_hypothalamus.ipynb`

## Fun Fact

* Interestingly, the name COMPOSITION is inspired by Piet Mondrian’s “Composition with Red, Blue and Yellow”, and our simulated data visually resembles that painting.
