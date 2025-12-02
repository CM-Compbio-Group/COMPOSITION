# COMPOSITION

We developed `COMPOSITION` to model hierarchical spatial structure across multiple scales of a tissue profiled by the latest single-cell spatial transcriptomics technologies and a fast inference using GPUs. We explicitly model the conditional probability relationships between adjacent layers of the tissue hierarchy in `COMPOSITION` (i.e., gene expression, cell type, niche abundance, and spatial domain) and construct the equivalence from the posterior of the probabilistic model and the optimization objective of two combined variational autoencoder neural networks. `COMPOSITION` achieved a comparable domain inference accuracy to state-of-the-art methods and learned biologically meaningful co-localized cell type groups. `COMPOSITION` captured spatial gradients of cell type compositions, a phenomenon observed in many tissue types recently, while previous spatial domain identification methods missed. Running `COMPOSITION` took 18 mins on a Visium HD sample with 0.45 million cells using GPUs and recapitulated known mouse brain domains.

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

* The name `COMPOSITION` is inspired by Piet Mondrian’s “Composition with Red, Blue and Yellow”, and our simulated data visually resembles that painting.
