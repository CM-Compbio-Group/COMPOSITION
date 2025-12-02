# COMPOSITION

COMPOSITION is a hierarchical model for single-cell spatial transcriptomics (SRT) data that represents these datasets as multi-scale features. Our group, Cong Ma Research Group, designed and implemented a VAE-based architecture with separate encoder–decoder modules that link molecular data to cell types, niches, and their co-enrichments, and we derived the ELBO for the multi-layer latent spaces to ground the probabilistic model in variational inference. This formulation makes hierarchical spatial structure explicit and stable, and GPU-accelerated inference allows processing of ~500k cells within minutes. We also incorporated computer-vision–inspired postprocessing to capture continuous spatial gradients, enabling the model to detect subtle, biologically meaningful patterns that are often missed by standard clustering.

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
