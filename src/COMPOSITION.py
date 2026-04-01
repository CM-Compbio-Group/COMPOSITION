import os
import math
from datetime import datetime
import anndata
import anndata as ad
import cellcharter as cc
import copy
from itertools import chain
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from numba.core.errors import NumbaDeprecationWarning
import numpy as np
import pandas as pd
from pathlib import Path
import scanpy as sc
import scipy.sparse as sp
import scipy
from scipy.stats import norm, multivariate_normal, wishart, Covariance
from scipy.special import logsumexp
import seaborn
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
import squidpy as sq
import sys
import itertools
from collections import Counter
import torch
from torch import nn
from torch.autograd import Variable
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.nn import VGAE, GCNConv, InnerProductDecoder, Sequential, SAGEConv
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
from tqdm import trange
from scipy import sparse
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

def step1_preprocess(adata_orig, X_pca=None, n_comps=20, standardization=None, target_sum=1e4):
    """
    Args:
        adata_orig: Raw AnnData. If no adata_orig.layers['counts'], adata_orig.X should be raw counts

    Returns:
        data: for running w/o minibatch
        dataloader: for running w/ minibatch
    """

    def ensure_xy_from_obs(adata, inplace=True, verbose=True):
        """
        In adata.obs, the x and y coordinate columns are unified and matched with 'x' and 'y'.
        """
    
        obs = adata.obs
    
        # 1. if there are x and y already, no action
        if {"x", "y"}.issubset(obs.columns):
            if verbose:
                print("Found existing 'x', 'y' in adata.obs. Nothing to do.")
            return adata
    
        # 2. define candidate pair in the order of priority
        candidates = [
            ("array_row", "array_col"),
            ("x_centroid", "y_centroid"),
            ("center_x", "center_y"),
            ("x_location", "y_location"),
            ("global_x", "global_y"),
            ("X", "Y"),
            ("pxl_row_in_fullres", "pxl_col_in_fullres"),
            ("Coordinates.X", "Coordinates.Y"),
            ("xc", "yc"),
            ("bin_row", "bin_col"),
            ("grid_row", "grid_col"),
            ("vertex_x", "vertex_y"),
            ("x_int", "y_int"),
            ("x_um", "y_um"),
        ]
    
        src_pair = None
        for cx, cy in candidates:
            if {cx, cy}.issubset(obs.columns):
                src_pair = (cx, cy)
                break
    
        if src_pair is None:
            raise ValueError(
                "Cannot infer spatial coordinates: none of the expected column pairs "
                "are present in adata.obs."
            )
    
        src_x, src_y = src_pair
    
        if not inplace:
            adata = adata.copy()
            obs = adata.obs
    
        # generate x, y from selected columns
        obs["x"] = obs[src_x].values
        obs["y"] = obs[src_y].values
    
        if verbose:
            print(f"Created 'x', 'y' from '{src_x}', '{src_y}' in adata.obs.")
    
        return adata
    
    
    def load_data_deprecated(X, adjacency):
        """
        Converts the node features and adjacency matrix into a PyTorch Geometric `Data` object.
    
        Args:
            X (np.ndarray): Node features, shape (num_nodes, num_features).[real number]
            adjacency (scipy.sparse.csr_matrix): Adjacency matrix of the graph.[{0,1}]
    
        Returns:
            Data: PyTorch Geometric `Data` object containing:
                  - x (torch.Tensor): Node features, shape (num_nodes, num_features).
                  - edge_index (torch.Tensor): Edge indices in COO format, shape (2, num_edges).
        """
        
        edge_index = np.vstack((adjacency.row, adjacency.col))
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    
        # Load node feature
        x = torch.tensor(X, dtype=torch.float)
    
        return Data(x=x, edge_index=edge_index)
    
    
    def load_data(X, adjacency):
        """
        Converts the node features and adjacency matrix into a PyTorch Geometric `Data` object.
    
        Args:
            X (np.ndarray): Node features, shape (num_nodes, num_features).[real number]
            adjacency (scipy.sparse.csr_matrix): Adjacency matrix of the graph.[{0,1}]
    
        Returns:
            Data: PyTorch Geometric `Data` object containing:
                  - x (torch.Tensor): Node features, shape (num_nodes, num_features).
                  - edge_index (torch.Tensor): Edge indices in COO format, shape (2, num_edges).
        """
        
        # Ensure adjacency is in COO format
        adjacency_coo = adjacency.tocoo()
        edge_index = np.vstack((adjacency_coo.row, adjacency_coo.col))
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    
        # Convert the DataFrame to a NumPy array and then to a PyTorch tensor
        x = torch.tensor(X.values, dtype=torch.float)
    
        return Data(x=x, edge_index=edge_index)
    
    adata = adata_orig.copy()
    if 'spatial' in adata.obsm and adata.obsm['spatial'].shape[1] == 2:
        adata.obs[['x', 'y']] = adata.obsm['spatial']
    else:
        ensure_xy_from_obs(adata)
        adata.obsm['spatial'] = adata.obs[['x', 'y']].values
    adata_orig.obs[['x', 'y']] = adata.obs[['x', 'y']] # for use outside this function
    
    if hasattr(adata, 'obsp') and 'spatial_connectivities' in adata.obsp:
        adjacency = adata.obsp['spatial_connectivities']
    else:
        sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, spatial_key='spatial')
        cc.gr.remove_long_links(adata)
        adjacency = adata.obsp['spatial_connectivities']

    if X_pca is not None:
        X = X_pca
    elif 'X_pca' in adata.obsm:
        X = adata.obsm['X_pca']
    else:
        if standardization is None:
            standardization = adata.shape[1] < 1000
        if not standardization:
            if 'counts' in adata.layers:
                adata.X = adata.layers['counts'].copy()

            ''' The following could not reveal the layered structure of the cerebral cortex
            sc.pp.normalize_total(adata, target_sum=target_sum)
            sc.pp.log1p(adata)
        
            # HVG selection
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')
            adata = adata[:, adata.var['highly_variable']].copy()

            # scaling
            sc.pp.scale(adata, zero_center=True)

            # PCA
            sc.tl.pca(adata, n_comps=n_comps)
            '''

            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.scale(adata, zero_center=True, max_value=10)
            sc.tl.pca(adata, n_comps=20)
            
            X = adata.obsm['X_pca']

            # optional Harmony if multiple slices
            # sce.pp.harmony_integrate(adata, key='sample_names')
            # X = adata.obsm['X_pca_harmony']
        else:            
            if 'counts' in adata.layers:
                raw = adata.layers['counts'].copy()
            else:
                raw = adata.X.copy()

            if sparse.issparse(raw):
                raw = raw.toarray()

            raw = raw.T
            
            sf = raw.sum(axis=0)            
            #sf = sf / sf.mean()
            sf_safe = sf.copy()
            sf_safe[sf_safe == 0] = 1
            sf_safe = sf_safe / sf_safe.mean()
            #x = np.log2(raw / sf + 1.0) 
            x = np.log2(raw / sf_safe + 1.0)
            x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)
            x = x.T
            X = PCA(n_components=n_comps, svd_solver="full").fit_transform(x)
    
    X = pd.DataFrame(X, index=adata.obs_names, columns=[f"PC{i+1}" for i in range(X.shape[1])])
    
    try:
        data = load_data_deprecated(X, adjacency)
    except:
        data = load_data(X, adjacency)
    
    dataloader = NeighborLoader( 
        data,
        input_nodes=torch.arange(data.num_nodes), # [0, 1, 2, ..., n_obs-1]
        num_neighbors=[10,5],                     # Node sampling for each GNN layer
        batch_size=2048,                          # Number of center nodes for each batch
        shuffle=True
    ) 
    return data, dataloader


def step2_run(adata, data, dataloader, seed=1, hid_dim=128, num_topics=32, n_celltypes=20, minibatch=False, temperature=0.1, early_stopping=False, alpha=1, wloss_spatial=1.2, wloss_KLD=0.005, wloss_recon=1, wloss_clf=1, wloss_entropy=1.2, tanh_thr=0.005, grad_clip=100, l1_ratio=0, coupling_weight=0.05, optim='adam', lr=9e-3, weight_decay=5e-3, momentum=0, epochs=3000, extra_epochs=600):
    def get_clf_class_weights(coords, n_celltypes, vae_z, p=4):
        # partition the range of x and y axes
        EPS = 1e-8
        px = np.ones(p) * 1.0 / p
        px[-1] -= EPS
        xboundary = np.percentile(coords[:, 0], 100 * np.cumsum(px))
        xboundary[-1] = np.max(coords[:, 0]) + 1
        xdigit = np.digitize(coords[:, 0], xboundary, right=True)
        ydigit = np.zeros(coords.shape[0], dtype=int)
        for x in range(p):
            idx_xbin = np.where(xdigit == x)[0]
            py = np.ones(p) * 1.0 / p
            py[-1] -= EPS
            yboundary = np.percentile(coords[idx_xbin, 1], 100 * np.cumsum(py))
            yboundary[-1] = np.max(coords[:, 1]) + 1
            ydigit[idx_xbin] = np.digitize(coords[idx_xbin, 1], yboundary, right=True)
        block_id = xdigit * p + ydigit
        background = 1.0 * np.bincount(block_id) / len(block_id)
        vae_z_np = vae_z.detach().cpu().numpy()
        vae_argmax = np.argmax(vae_z_np, axis=1)
        unique_info_cell_type = []
        for i in range(n_celltypes):
            joint_prob = pd.concat([
                pd.Series(block_id[vae_argmax == i]).value_counts(),
                pd.Series(block_id[vae_argmax != i]).value_counts()
            ], axis=1)
        
            joint_prob.fillna(0, inplace=True)
            joint_prob = joint_prob.values / joint_prob.values.sum()    
            cond_entropy = scipy.stats.entropy(joint_prob.flatten()) - scipy.stats.entropy(joint_prob.sum(axis=1))
            unique_info_cell_type.append(cond_entropy / scipy.stats.entropy(joint_prob.sum(axis=0)))    
        weights = np.where( (np.array(unique_info_cell_type) > 0.95) | (np.mean(vae_z_np, axis=0) <= 0.01), 0.1, 1.0)
        clf_class_weights = torch.from_numpy(weights.reshape(1,-1)).to(torch.float32)
    
        return clf_class_weights / clf_class_weights.mean()
    
    pyg.seed_everything(seed)
    model = VGAE(ProdLDAEncoder(data.num_features, hid_dim, num_topics))
    model_ct = VAE(data.num_features, hid_dim, 1, num_categories=n_celltypes)
    model_ff = FFPredict(num_topics, n_celltypes)

    [model, model_ct, model_ff], device, loss_values = train_vae(data, model, model_ct, model_ff, temperature=temperature, early_stopping=early_stopping, alpha=alpha, wloss_spatial=wloss_spatial, wloss_KLD=wloss_KLD, wloss_recon=wloss_recon, wloss_clf=wloss_clf, wloss_entropy=wloss_entropy, grad_clip=grad_clip, l1_ratio=l1_ratio, lr=lr, weight_decay=weight_decay, epochs=epochs)

    recon_x, logits, logits_re = model_ct(data.x.to(device), temperature=temperature)
    vae_z = logits_re
    vae_z = vae_z.squeeze(1)
    spatial_coords = adata.obs[['x', 'y']]
    coords = spatial_coords[['x', 'y']].values

    clf_class_weights = get_clf_class_weights(coords, model_ct.num_categories, vae_z)
    
    pyg.seed_everything(seed)
    model = VGAE(ProdLDAEncoder(data.num_features, hid_dim, num_topics))
    #model_ct = VAE(data.num_features, hid_dim, 1, num_categories=n_celltypes) # muted to avoid re-ordering cell types
    model_ff = FFPredict(num_topics, n_celltypes)
    if not minibatch:
        [model, model_ct, model_ff], device, loss_values = train(data, model, model_ct, model_ff, clf_class_weights=clf_class_weights, temperature=temperature, early_stopping=early_stopping, alpha=alpha, wloss_spatial=wloss_spatial, wloss_KLD=wloss_KLD, wloss_recon=wloss_recon, wloss_clf=wloss_clf, wloss_entropy=wloss_entropy, tanh_thr=tanh_thr, grad_clip=grad_clip, l1_ratio=l1_ratio, coupling_weight=coupling_weight, optim=optim, lr=lr, weight_decay=weight_decay, momentum=momentum, epochs=epochs, extra_epochs=extra_epochs)

    else:
        [model, model_ct, model_ff], device, loss_values = train_batch(dataloader, model, model_ct, model_ff, clf_class_weights=clf_class_weights, temperature=temperature, early_stopping=early_stopping, alpha=alpha, wloss_spatial=wloss_spatial, wloss_KLD=wloss_KLD, wloss_recon=wloss_recon, wloss_clf=wloss_clf, wloss_entropy=wloss_entropy, tanh_thr=tanh_thr, grad_clip=grad_clip, l1_ratio=l1_ratio, coupling_weight=coupling_weight, optim=optim, lr=lr, weight_decay=weight_decay, momentum=momentum, epochs=epochs, extra_epochs=extra_epochs)

    plt.figure()
    plt.plot(loss_values)
    plt.title("All losses")
    plt.show()
    
    plt.figure()
    plt.plot(loss_values[-200:])
    plt.title("Last 200 losses")
    plt.show()

    return model, model_ct, model_ff


def step3_postprocess(data, model, model_ct, model_ff, temperature=0.1, n_clusters=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model_ff = model_ff.to(device)
    model_ct = model_ct.to(device)
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    
    model.eval()
    with torch.no_grad():
        n_samples = 50
        p_mat = []
        with torch.no_grad():
            for r in range(n_samples):
                z, p, posterior_mean, posterior_logvar, posterior_var = model.encoder(data.x, data.edge_index)
                p_mat.append(p)
    
        p = torch.stack(p_mat, dim=0).mean(dim=0)
        #z, p, posterior_mean, posterior_logvar, posterior_var = model.encoder(data.x, data.edge_index)
        recon = F.softmax(model_ff(p))             # reconstructed distribution over vocabulary
    
    recon_celltype = model_ff(p)                                              # logits: (N, n_classes)
    cell_types_niche = recon_celltype.argmax(dim=1).cpu().numpy().astype(str)  # sc.pl.dotplot handles string label better
    
    recon_x, logits, _ = model_ct(data.x.to(device), temperature=temperature)
    logits = logits.squeeze(1)
    cell_types_vae = F.softmax(logits, dim=1).argmax(dim=1).cpu().numpy().astype(str)

    niche_composition = F.softmax(model_ff.fc1.weight.detach().cpu(), dim=0)

    pred_domains = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(p.detach().cpu())
                                  
    return p, cell_types_niche, cell_types_vae, niche_composition, pred_domains, recon_celltype, logits


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) with Gumbel-softmax reparameterization for learning categorical latent representations.

    Args:
        input_dim (int): Dimension of the input features.[positive int]
        hidden_dim (int): Number of hidden units in the encoder.[positive int]
        latent_dim (int): Dimension of the latent representation.[positive int]
        num_categories (int): Number of categories [positive int]

    Inputs:
        x (torch.Tensor): Input node features, shape (num_nodes, input_dim).[real number]
        temperature (float): Temperature parameter for Gumbel-softmax reparameterization.[positive real number]

    Outputs:
        reconstruction (torch.Tensor): Reconstructed node features, shape (num_nodes, input_dim).[real number]
        latent_logits (torch.Tensor): Logits before sampling, shape (num_nodes, latent_dim, num_categories).[real number]
        z (torch.Tensor): Sampled latent variables, shape (num_nodes, latent_dim, num_categories).[[0,1]]
        
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim, num_categories):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_categories = num_categories

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * num_categories)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * num_categories, input_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, input_dim),
        )

        self.log_sigma2 = nn.Parameter(torch.zeros(self.input_dim))

    def reparameterize_gumbel_softmax(self, logits, temperature):
        # Gumbel-softmax reparameterization
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        y = logits + gumbel_noise
        return F.softmax(y / temperature, dim=-1)

    def encode(self, x):
        logits = self.encoder(x)
        logits = logits.view(-1, self.latent_dim, self.num_categories)
        return logits

    def decode(self, z):
        z_flat = z.view(-1, self.latent_dim * self.num_categories)
        return self.decoder(z_flat)

    def forward(self, x, temperature):
        # Encoder
        logits = self.encode(x)

        # Reparameterization trick
        z = self.reparameterize_gumbel_softmax(logits, temperature)

        # Decoder
        reconstruction = self.decode(z)
        return reconstruction, logits, z


def vae_loss(recon_x, x, logits, log_sigma2, beta=1.0):
    """
    recon_x: (B, G)
    x      : (B, G)
    logits : (B, latent_dim, num_categories)
    log_sigma2: (G,)   # genewise log-variance
    """

    # genewise variance (shape: [G])
    sigma2 = torch.exp(log_sigma2) + 1e-8   # (G,)

    # broadcasting: (B, G) - (B, G), sigma2 (G,) → (B, G)
    nll_elem = 0.5 * (log_sigma2 + (x - recon_x)**2 / sigma2)  # (B, G)

    # batch/gene total sum
    recon_loss = nll_elem.sum()

    # KL as original
    q = F.softmax(logits, dim=-1)
    log_q = F.log_softmax(logits, dim=-1)
    kl_div = (q * (log_q - torch.log(torch.tensor(1.0 / logits.size(-1), device=logits.device)))).sum(dim=-1).sum()

    return recon_loss + beta * kl_div


class ProdLDAEncoder(torch.nn.Module):
    """
    # Define Encoder
    # Code adapted from https://github.com/hyqneuron/pytorch-avitm/tree/master
    # Remove batchnorm layers
    # Lower the weight of KL divergence
    """
    def __init__(self, in_channels, hid_channels, num_topics):
        super(ProdLDAEncoder, self).__init__()
        self.num_topics = num_topics

        self.base_conv = GCNConv(in_channels, hid_channels)
        self.conv1 = GCNConv(hid_channels, hid_channels)
        self.conv2 = GCNConv(hid_channels, hid_channels)
        self.conv_dropout   = nn.Dropout(0.2)
        self.conv_mu = GCNConv(hid_channels, num_topics)
        self.conv_logstd = GCNConv(hid_channels, num_topics)
        self.p_drop     = nn.Dropout(0.2)
        self.alpha = 1

        def prodlda_laplace_prior(num_topics: int, alpha):
        	"""
        	Laplace approximation of Dirichlet in softmax basis (Hennig+2012; used by ProdLDA).
        	Returns (prior_mean, prior_var, prior_logvar) for diagonal case by default.
        	If full_cov=True, also returns full covariance matrix 'Sigma' (KxK).
        	"""
        	K = int(num_topics)
        	a = torch.as_tensor(alpha, dtype=torch.float32)
        	if a.dim() == 0: a = a.repeat(K)                 # (K,)
        	a = a.clamp_min(1e-8)                            # numeric safety
        	a = a.unsqueeze(0)                               # (1,K)
        
        	loga = a.log()                                   # (1,K)
        	prior_mean = loga - loga.mean(dim=1, keepdim=True)  # μ_k = log α_k − mean(log α)
        
        	inva = 1.0 / a                                   # (1,K)
        	sum_inva = inva.sum(dim=1, keepdim=True)         # (1,1)
        	prior_var = inva * (1.0 - 2.0/K) + sum_inva / (K**2)   # Σ_kk
        
        	prior_var = prior_var.clamp_min(1e-8)
        	prior_logvar = prior_var.log()
        
        	return prior_mean, prior_var, prior_logvar
        	
        prior_mean, prior_var, prior_logvar = prodlda_laplace_prior(num_topics, self.alpha)
        
        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar)

    def forward(self, x, edge_index):
        x = F.softplus(self.base_conv(x, edge_index))
        x = F.softplus(self.conv1(x, edge_index))
        x = F.softplus(self.conv2(x, edge_index))
        if self.train:
            x = self.conv_dropout(x)
        posterior_mean = self.conv_mu(x, edge_index)  # posterior mean
        posterior_logvar = self.conv_logstd(x, edge_index)  # posterior log variance
        posterior_var = posterior_logvar.exp()
        
        # take sample
        eps = Variable(x.data.new().resize_as_(posterior_mean.data).normal_()) # noise
        z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization
        p = F.softmax(z)                                                # mixture probability
        if self.train:
            p = self.p_drop(p)

        return z, p, posterior_mean, posterior_logvar, posterior_var

    def KLD(self, posterior_mean, posterior_logvar, posterior_var):
        prior_mean   = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var    = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.num_topics )
        return KLD.sum()


class FFPredict(nn.Module):
  def __init__(self, latent_dim, output_dim):
      super().__init__()
      self.fc1 = nn.Linear(latent_dim, output_dim, bias=False)
      nn.init.xavier_uniform_(self.fc1.weight) 

  def forward(self, p): 
      p = p.clamp(min=0)
      p = p / (p.sum(dim=1, keepdim=True) + 1e-12)

      W = F.softplus(self.fc1.weight)
      W = W / (W.sum(dim=0, keepdim=True) + 1e-12) 

      probs = p @ W.t() 
      return probs


# train data
def train_batch(dataloader, model, model_ct, model_ff, clf_class_weights=None, temperature=0.1, optim='adam', lr=5e-3, weight_decay=0, momentum=0, alpha=None, betas=(0.9, 0.999), wloss_spatial=1.2, wloss_KLD=0.005, wloss_recon=1, wloss_clf=1, wloss_entropy=1.2, wtanh = None, tanh_thr = 0.005, l1_ratio=0, grad_clip=200, early_stopping=True, spotwise_celltype_probability=None, adjacency=None, coupling_weight=0.05, epochs=600, extra_epochs=120):
    """
    Simultaneous model training for VGAE(model), VAE(model_ct), and FFPredict(model_ff)
    dataloader : mini-batch loader, e.g. NeighborLoader
    epochs     : number of epochs
    lr         : learning rate
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if clf_class_weights is None:
        clf_class_weights = torch.ones(model_ct.num_categories).to(device)
        clf_class_weights0 = torch.ones(model_ct.num_categories).to(device)
    else:
        clf_class_weights = clf_class_weights.to(device)
        clf_class_weights0 = torch.ones(model_ct.num_categories).to(device)
    
    model = model.to(device)       # move model to GPU 
    model_ct = model_ct.to(device) # move model_ct to GPU
    model_ff = model_ff.to(device) # move model_ff to GPU 
    if alpha is not None:
        model.alpha=alpha
    if wtanh is None:
        wtanh = 0 #dataloader.data.x.shape[0] / 60
                    
    if spotwise_celltype_probability is None:    
        if optim=='sgd':
            optimizer = torch.optim.SGD( chain(model.parameters(), model_ct.parameters(), model_ff.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay )
        elif optim=='adam':
            optimizer = torch.optim.Adam( chain(model.parameters(), model_ct.parameters(), model_ff.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        elif optim=='adamw':
            optimizer = torch.optim.AdamW( chain(model.parameters(), model_ct.parameters(), model_ff.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        elif optim=='adagrad':
            optimizer = torch.optim.Adagrad( chain(model.parameters(), model_ct.parameters(), model_ff.parameters()), weight_decay=weight_decay)  
    else:
        if optim=='sgd':
            optimizer = torch.optim.SGD( chain(model.parameters(), model_ff.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optim=='adam':
            optimizer = torch.optim.Adam( chain(model.parameters(), model_ff.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        elif optim=='adamw':
            optimizer = torch.optim.AdamW( chain(model.parameters(), model_ff.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        elif optim=='adagrad':
            optimizer = torch.optim.Adagrad( chain(model.parameters(), model_ff.parameters()), weight_decay=weight_decay )  
    
    model.train()                  # switch to training mode
    model_ct.train()               # switch to training mode
    model_ff.train()               # switch to training mode
                    
    # loss fun
    loss_connection = nn.CrossEntropyLoss(reduction='sum')
    loss_mse = nn.MSELoss(reduction='sum')
    loss_values = []
    log_sigma2 = 0
    log_sigma2_fixed = 0
    
    for epoch in range(epochs+extra_epochs):
        epoch_loss = 0
        for batch in dataloader:
            batch = batch.to(device)    # move data to GPU
            optimizer.zero_grad()       # clear previous gradients

            # VGAE
            z, p, posterior_mean, posterior_logvar, posterior_var = model.encoder(batch.x, batch.edge_index)
                        
            loss_spatial = wloss_spatial * loss_mse(p[batch.edge_index[0]], p[batch.edge_index[1]])
            loss_KLD = wloss_KLD * model.encoder.KLD(posterior_mean, posterior_logvar, posterior_var) 
            
            # favor a low entropy of p
            EPS = 1e-20
            if epoch < int(epochs/6):
                loss_entropy = 0.0 * wloss_entropy * -(p * torch.log(p + EPS)).sum()
                t_anneal = 1
                log_sigma2 = model_ct.log_sigma2
            elif epoch < int(2*epochs/6):
                loss_entropy = 0.5 * wloss_entropy * -(p * torch.log(p + EPS)).sum()
                t_anneal = 1
                if epoch == int(epochs/6):
                    log_sigma2_fixed = model_ct.log_sigma2.detach()
                log_sigma2 = log_sigma2_fixed
            elif epoch < int(3*epochs/6):
                loss_entropy = 1.0 * wloss_entropy * -(p * torch.log(p + EPS)).sum()
                t_anneal = 1
                log_sigma2 = log_sigma2_fixed
            elif epoch > epochs:
                loss_entropy = 0.8 * wloss_entropy * -(p * torch.log(p + EPS)).sum()
                t_anneal = temperature
                #coupling_weight = 0
            else:
                loss_entropy = 1.5 * wloss_entropy * -(p * torch.log(p + EPS)).sum()
                t_anneal = max(temperature, 1.0 - (1.0-temperature)/500 * (epoch - int(epochs/2)) )
                log_sigma2 = log_sigma2_fixed
    
            if spotwise_celltype_probability is None:
                # VAE
                recon_x, logits, logits_re = model_ct(batch.x, temperature=t_anneal)
                loss_recon = wloss_recon * vae_loss(recon_x, batch.x, logits, log_sigma2)
    
                tensor_target = logits_re.squeeze(1).detach() + coupling_weight * (logits_re.squeeze(1) - logits_re.squeeze(1).detach())
                if adjacency is not None:
                    tensor_target = adjacency @ tensor_target
                recon_celltype = model_ff(p)
                eps = 1e-12
                log_recon_celltype = (recon_celltype.clamp_min(eps)).log()

                #loss_clf = -(tensor_target * log_recon_celltype).sum() * wloss_clf
                loss_clf = -(tensor_target * log_recon_celltype * clf_class_weights).sum() * wloss_clf
                loss_clf0 = -(tensor_target * log_recon_celltype * clf_class_weights0).sum() * wloss_clf
                    
            else:
                loss_recon = 0
                tensor_target = torch.from_numpy(spotwise_celltype_probability).to(torch.float32)
                tensor_target = tensor_target.to(device)
                if adjacency is not None:
                    tensor_target = adjacency @ tensor_target
                recon_celltype = model_ff(p)
                eps = 1e-12
                log_recon_celltype = (recon_celltype.clamp_min(eps)).log()
                
                #loss_clf = -(tensor_target * log_recon_celltype).sum() * wloss_clf
                loss_clf = -(tensor_target * log_recon_celltype * clf_class_weights).sum() * wloss_clf
                loss_clf0 = -(tensor_target * log_recon_celltype * clf_class_weights0).sum() * wloss_clf
            
            loss = loss_spatial + loss_KLD + loss_entropy + loss_recon + loss_clf + l1_ratio * z.abs().sum(axis=0).sum() -wtanh * torch.tanh(tanh_thr * p.abs().sum(dim=0)).sum()
                                                             # l1_ratio*model_ff(p).abs().sum(axis=0).sum()
            epoch_loss += loss.item()
            loss.backward()             # backprop
            
            if grad_clip is not None:              # Gradient Clipping for Gradient Explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(model_ct.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(model_ff.parameters(), grad_clip)
            
            optimizer.step()            # update parameters
            
        loss_values.append( epoch_loss ) 
        #loss_values.append( epoch_loss / len(dataloader) ) 
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch} Loss: {epoch_loss}")

        if early_stopping and len(loss_values) >= 20 and epoch > int(2*epochs/3) + 20:
            import numpy as np
            from sklearn.linear_model import LinearRegression

            window = loss_values[-20:]
            smoothed = np.convolve(window, np.ones(3)/3, mode='valid')
            
            # [1] count the number of increasing segments
            count_up = sum(1 for prev, cur in zip(smoothed, smoothed[1:]) if cur > prev)
        
            # [2] calculate R^2 (lower R^2 when high vibration and low trend)
            X = np.arange(len(window)).reshape(-1, 1)
            y = np.array(window).reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            r2 = reg.score(X, y)
        
            # [3] little change
            val_range = max(window) - min(window)
            tolerance = 1e-4 * min(window)
        
            # [4] slight increase in recent loss
            delta = window[-1] - window[0]
            small_delta = 1e-3 * window[0]
        
            # Early termination when the condition is met
            if (
                count_up >= 13 or       # [1] frequent rise
                r2 < 0.05 or            # [2] no trend
                val_range < tolerance or# [3] little change
                delta > small_delta     # [4] slight increase
            ):
                print(f"Early stopping at epoch {epoch+1}")
                break        

    print(f"loss: {loss}")
    print(f"loss-loss_entropy: {loss-loss_entropy}")
    print(f"loss_recon: {loss_recon}")
    print(f"loss_clf: {loss_clf}")
    print(f"loss0: {loss - loss_clf + loss_clf0}")
    return [model, model_ct, model_ff], device, loss_values


# train data
def train(data, model, model_ct, model_ff, clf_class_weights=None, temperature=0.1, optim='adam', lr=5e-3, weight_decay=0, momentum=0, alpha=None, betas=(0.9, 0.999), wloss_spatial=1.2, wloss_KLD=0.005, wloss_recon=1, wloss_clf=1, wloss_entropy=1.2, wtanh = None, tanh_thr = 0.005, l1_ratio=0, grad_clip=200, early_stopping=True, spotwise_celltype_probability=None, adjacency=None, coupling_weight=0.05, epochs=3000, extra_epochs=600):
    """
    Train the VGAE, VAE, and feed-forward predictor jointly.

    Args:
        data (Data): PyTorch Geometric `Data` object containing node features and edge indices.
        model (VGAE): VGAE model for learning spatial domains.
        model_ct (VAE): VAE model for learning cell type representations.
        model_ff (FFPredict): Feed-forward model for predicting cell types.
        epochs (int): Number of training epochs.

    Outputs:
        model, model_ct, model_ff (torch.nn.Module): Trained models.
        device (torch.device): The device (CPU/GPU) used for training.
        loss_values (list): List of training loss values at each epoch.[positive real number]
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    if clf_class_weights is None:
        clf_class_weights = torch.ones(model_ct.num_categories).to(device)
        clf_class_weights0 = torch.ones(model_ct.num_categories).to(device)
    else:
        clf_class_weights = clf_class_weights.to(device)
        clf_class_weights0 = torch.ones(model_ct.num_categories).to(device)

    model = model.to(device)       # move model to GPU 
    model_ct = model_ct.to(device)    # move model_ct to GPU 
    model_ff = model_ff.to(device) # move model_ff to GPU
    if alpha is not None:
        model.alpha=alpha
    if wtanh is None:
        wtanh = 0 #data.x.shape[0] / 60

    if spotwise_celltype_probability is None:
        if optim=='sgd':
            optimizer = torch.optim.SGD( chain(model.parameters(), model_ct.parameters(), model_ff.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay )
        elif optim=='adam':
            optimizer = torch.optim.Adam( chain(model.parameters(), model_ct.parameters(), model_ff.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        elif optim=='adamw':
            optimizer = torch.optim.AdamW( chain(model.parameters(), model_ct.parameters(), model_ff.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        elif optim=='adagrad':
            optimizer = torch.optim.Adagrad( chain(model.parameters(), model_ct.parameters(), model_ff.parameters()), weight_decay=weight_decay )
    else:
        if optim=='sgd':
            optimizer = torch.optim.SGD( chain(model.parameters(), model_ff.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay )
        elif optim=='adam':
            optimizer = torch.optim.Adam( chain(model.parameters(), model_ff.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        elif optim=='adamw':
            optimizer = torch.optim.AdamW( chain(model.parameters(), model_ff.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        elif optim=='adagrad':
            optimizer = torch.optim.Adagrad( chain(model.parameters(), model_ff.parameters()), weight_decay=weight_decay )  

    model.train()                   # switch to training mode
    model_ct.train()                # switch to training mode
    model_ff.train()                # switch to training mode
    
    # loss fun
    loss_connection = nn.CrossEntropyLoss(reduction='sum')
    loss_mse = nn.MSELoss(reduction='sum')
    loss_values = []
    log_sigma2 = 0
    log_sigma2_fixed = 0
        
    for epoch in range(epochs+extra_epochs):
        optimizer.zero_grad()       # clear previous gradients
        
        # GVAE
        z, p, posterior_mean, posterior_logvar, posterior_var = model.encoder(data.x, data.edge_index)
        
        loss_spatial = wloss_spatial * loss_mse(p[data.edge_index[0]], p[data.edge_index[1]])       
        loss_KLD = wloss_KLD * model.encoder.KLD(posterior_mean, posterior_logvar, posterior_var)

        # favor a low entropy of p
        EPS = 1e-20
        if epoch < int(epochs/6):
            loss_entropy = 0.0 * wloss_entropy * -(p * torch.log(p + EPS)).sum()
            t_anneal = 1
            log_sigma2 = model_ct.log_sigma2
        elif epoch < int(2*epochs/6):
            loss_entropy = 0.5 * wloss_entropy * -(p * torch.log(p + EPS)).sum()
            t_anneal = 1
            if epoch == int(epochs/6):
                log_sigma2_fixed = model_ct.log_sigma2.detach()
            log_sigma2 = log_sigma2_fixed
        elif epoch < int(3*epochs/6):
            loss_entropy = 1.0 * wloss_entropy * -(p * torch.log(p + EPS)).sum()
            t_anneal = 1
            log_sigma2 = log_sigma2_fixed
        elif epoch > epochs:
            loss_entropy = 0.8 * wloss_entropy * -(p * torch.log(p + EPS)).sum()
            t_anneal = temperature
            #coupling_weight = 0
        else:
            loss_entropy = 1.5 * wloss_entropy * -(p * torch.log(p + EPS)).sum()
            t_anneal = max(temperature, 1.0 - (1.0-temperature)/500 * (epoch - int(epochs/2)) )
            log_sigma2 = log_sigma2_fixed
        
        if spotwise_celltype_probability is None:
            # VAE
            recon_x, logits, logits_re = model_ct(data.x, temperature=t_anneal)
            loss_recon = wloss_recon * vae_loss(recon_x, data.x, logits, log_sigma2)

            tensor_target = logits_re.squeeze(1).detach() + coupling_weight * (logits_re.squeeze(1) - logits_re.squeeze(1).detach())
            if adjacency is not None:
                tensor_target = adjacency @ tensor_target
            recon_celltype = model_ff(p)
            eps = 1e-12
            log_recon_celltype = (recon_celltype.clamp_min(eps)).log()

            #loss_clf = -(tensor_target * log_recon_celltype).sum() * wloss_clf
            loss_clf = -(tensor_target * log_recon_celltype * clf_class_weights).sum() * wloss_clf
            loss_clf0 = -(tensor_target * log_recon_celltype * clf_class_weights0).sum() * wloss_clf
            
        else:
            loss_recon = 0
            tensor_target = torch.from_numpy(spotwise_celltype_probability).to(torch.float32)
            tensor_target = tensor_target.to(device)
            if adjacency is not None:
                tensor_target = adjacency @ tensor_target
            recon_celltype = model_ff(p)
            eps = 1e-12
            log_recon_celltype = (recon_celltype.clamp_min(eps)).log()

            #loss_clf = -(tensor_target * log_recon_celltype).sum() * wloss_clf
            loss_clf = -(tensor_target * log_recon_celltype * clf_class_weights).sum() * wloss_clf
            loss_clf0 = -(tensor_target * log_recon_celltype * clf_class_weights0).sum() * wloss_clf

        loss = loss_spatial + loss_KLD + loss_entropy + loss_recon + loss_clf + l1_ratio *  z.abs().sum(axis=0).sum()  -wtanh * torch.tanh(tanh_thr * p.abs().sum(dim=0)).sum()
                                        #p_cell = F.softmax(model_ff.fc1.weight, dim=0)
                                        #l1_ratio * -(p_cell * torch.log(p_cell + EPS)).sum()
                                        #l1_ratio*model_ff(p).abs().sum(axis=0).sum()
        
        loss_values.append( loss.item() )
        loss.backward()             # backprop
        
        if grad_clip is not None:              # Gradient Clipping for Gradient Explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(model_ct.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(model_ff.parameters(), grad_clip)
        
        optimizer.step()            # update parameters
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch} Loss: {loss.item()}")
        
        if early_stopping and len(loss_values) >= 20 and epoch > int(2*epochs/3) + 20:
            import numpy as np
            from sklearn.linear_model import LinearRegression

            window = loss_values[-20:]
            smoothed = np.convolve(window, np.ones(3)/3, mode='valid')
            
            # [1] count the number of increasing segments
            count_up = sum(1 for prev, cur in zip(smoothed, smoothed[1:]) if cur > prev)
        
            # [2] calculate R^2 (lower R^2 when high vibration and low trend)
            X = np.arange(len(window)).reshape(-1, 1)
            y = np.array(window).reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            r2 = reg.score(X, y)
        
            # [3] little change
            val_range = max(window) - min(window)
            tolerance = 1e-4 * min(window)
        
            # [4] slight increase in recent loss
            delta = window[-1] - window[0]
            small_delta = 1e-3 * window[0]
        
            # Early termination when the condition is met
            if (
                count_up >= 13 or       # [1] frequent rise
                r2 < 0.05 or            # [2] no trend
                val_range < tolerance or# [3] little change
                delta > small_delta     # [4] slight increase 
            ):
                print(f"Early stopping at epoch {epoch+1}")
                break        

    print(f"loss: {loss}")
    print(f"loss-loss_entropy: {loss-loss_entropy}")
    print(f"loss_recon: {loss_recon}")
    print(f"loss_clf: {loss_clf}")
    print(f"loss0: {loss - loss_clf + loss_clf0}")
    return [model, model_ct, model_ff], device, loss_values


def train_vae(data, model, model_ct, model_ff, epochs=1500, temperature=1.0, lr=5e-3, alpha=None, betas=(0.9, 0.999), weight_decay=0,
          wloss_spatial=0.8, wloss_KLD=0.005, wloss_recon=1, wloss_clf=1, wloss_entropy=2.0, wtanh = None, tanh_thr = 0.005,
          l1_ratio=0, grad_clip=200, early_stopping=True, spotwise_celltype_probability=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model_ct = model_ct.to(device)    # move model_ct to GPU 
    if alpha is not None:
        model.alpha=alpha

    optimizer = torch.optim.Adam( chain(model_ct.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
    model_ct.train()                # switch to training mode
    
    # loss fun
    loss_connection = nn.CrossEntropyLoss(reduction='sum')
    loss_mse = nn.MSELoss(reduction='sum')
    loss_values = []
    log_sigma2 = 0
    log_sigma2_fixed = 0
        
    for epoch in range(epochs):
        optimizer.zero_grad()       # clear previous gradients
        
        # favor a low entropy of p
        EPS = 1e-20
        if epoch < int(epochs/6):
            t_anneal = 1
            log_sigma2 = model_ct.log_sigma2
        elif epoch < int(2*epochs/6):
            t_anneal = 1
            if epoch == int(epochs/6):
                log_sigma2_fixed = model_ct.log_sigma2.detach()
            log_sigma2 = log_sigma2_fixed
        elif epoch < int(3*epochs/6):
            t_anneal = 1
            log_sigma2 = log_sigma2_fixed
        else:
            t_anneal = max(temperature, 1.0 - (1.0-temperature)/500 * (epoch - int(epochs/2)) )
            log_sigma2 = log_sigma2_fixed
        
        # VAE
        recon_x, logits, logits_re = model_ct(data.x, temperature=t_anneal)
        loss_recon = wloss_recon * vae_loss(recon_x, data.x, logits, log_sigma2)

        tensor_target = logits_re.squeeze(1)
        
        loss = loss_recon 
        loss_values.append( loss.item() )
        loss.backward()             # backprop
        
        if grad_clip is not None:              # Gradient Clipping for Gradient Explosion
            torch.nn.utils.clip_grad_norm_(model_ct.parameters(), grad_clip)
            
        optimizer.step()            # update parameters
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch} Loss: {loss.item()}")
        
    print(f"loss: {loss}")
    return [model, model_ct, model_ff], device, loss_values


def step1_prev_simulation():
    # ============================================================
    # 1) Global parameters
    #    ── (If you want to change it, you can change it here) ───────────────────────────
    # ============================================================
    # --- composition ----------------------------------------------------
    K1 = 9        # domain-marker cell-types
    K2 = 9        # paired  ✕2  → 18
    K3 = 9        # rare
    C  = K1 + 2*K2 + K3        # 36 cell-types total
    
    # --- spot / space ------------------------------------------------
    spots_per_domain  = 5_000                   # 50×100 grid
    H_dom, W_dom      = 50, 100                 # domain height, width
    domains_per_side  = 3                       # 3×3 = 9 domains
    n_domains         = K1                      # (=9)
    
    # --- statistical distribution --------------------------------------------------
    mu_dom,  sd_dom  = 0.50, 0.01
    mu_pair, sd_pair = 0.20, 0.04
    EPS = 1e-6
    rng = np.random.default_rng(123)
    
    # --- z-embedding / expression -------------------------------------------
    DIM_Z   = K1 + K2 + 1       # 19
    d_expr  = 20                # top-PC dimension
    # ============================================================
    
    # ------------------------------------------------------------
    # 2) spatial coordinate & domain index
    # ------------------------------------------------------------
    coords_list, domain_list = [], []
    for dy in range(domains_per_side):
        for dx in range(domains_per_side):
            dom_idx = dy * domains_per_side + dx
            ys, xs = np.mgrid[0:H_dom, 0:W_dom]                      # 50×100
            # Putting (row, col) as (x, y) is intuitive when visualizing
            coords_dom = np.stack([xs.ravel() + dx*W_dom,
                                   ys.ravel() + dy*H_dom], axis=1)
            coords_list.append(coords_dom)
            domain_list.append(np.full(spots_per_domain, dom_idx, int))
    
    coords      = np.vstack(coords_list)         # (N, 2)
    domain_ids  = np.concatenate(domain_list)    # (N,)
    N           = coords.shape[0]                # Total spot num (= 45 000)
    
    # ------------------------------------------------------------
    # 3) Celltype label & z-embedding simulation
    # ------------------------------------------------------------
    def tnorm(m, s, lo=0., hi=1.):
        """truncated Normal one-liner"""
        while True:
            x = rng.normal(m, s)
            if lo < x < hi:
                return x
    
    # helper for pair index
    def idx_pairA(d): return K1 + 2*d
    def idx_pairB(d): return K1 + 2*d + 1
    rare_idx = np.arange(K1 + 2*K2, C)
    
    cell_types_obs = np.empty(N,  dtype=int)
    z_raw_matrix   = np.zeros((N, DIM_Z))
    
    for d in range(n_domains):                           # 0..8
        idx = np.where(domain_ids == d)[0]
        for i in idx:
            # ---------- (1) Probability vector p --------------------------------
            while True:
                p_dom  = tnorm(mu_dom,  sd_dom)
                #p_sum  = tnorm(mu_pair, sd_pair)         # The total of the two paired
                p_sum = tnorm(
                    mu_pair + (1) * sd_pair / sd_dom * (p_dom - mu_dom),
                    sd_pair * np.sqrt(1 - (1)**2)
                )
                if p_dom + p_sum < 0.95:
                    break
            pA, pB  = p_sum / 2, p_sum / 2
            remain  = 1.0 - p_dom - p_sum
            p       = np.full(C, EPS)                    # Initialize to small value
            p[d]                   = p_dom
            p[idx_pairA(d)]        = pA
            p[idx_pairB(d)]        = pB
            # Same distribution to other rare types
            p[rare_idx] += (remain - EPS * (C - 3)) / len(rare_idx)
    
            # ---------- (2) Celltype label --------------------------------
            cell_types_obs[i] = rng.choice(C, p=p)
    
            # ---------- (3) z_raw (before normalization) ---------------------------
            z = np.zeros(DIM_Z)
            # ① Domain marker 9
            z[:K1] = np.log(p[:K1])
            # ② Pair average log 9
            for j in range(K2):
                z[K1 + j] = np.log((p[idx_pairA(j)] + p[idx_pairB(j)]) / 2)
            # ③ Rare sum log
            z[-1] = np.log(p[rare_idx].sum())
            z_raw_matrix[i] = z
    
    # Scaling(option) → Put it on γ= 1 here
    z_embeddings = z_raw_matrix.copy()          # (N, 19)
    
    # ------------------------------------------------------------
    # 4) 4-neighbor adjacency matrix (sparse CSR)
    # ------------------------------------------------------------
    H_full = domains_per_side * H_dom           # 150
    W_full = domains_per_side * W_dom           # 300
    index_grid = np.arange(N).reshape(H_full, W_full)
    
    edge_i, edge_j = [], []
    for y in range(H_full):
        for x in range(W_full):
            v = index_grid[y, x]
            if x + 1 < W_full:
                u = index_grid[y, x+1]
                edge_i.extend([v, u])
                edge_j.extend([u, v])
            if y + 1 < H_full:
                u = index_grid[y+1, x]
                edge_i.extend([v, u])
                edge_j.extend([u, v])
    
    adjacency = sp.coo_matrix(
        (np.ones(len(edge_i), dtype=np.float32), (edge_i, edge_j)),
        shape=(N, N)
    ).tocsr()
    
    # ------------------------------------------------------------
    # 5) Expression (or PCA) matrix X_simulated
    #     - Fixed 20-D vector per cell type, noise X
    # ------------------------------------------------------------
    M_expr = rng.normal(0, 1, size=(C, d_expr))     # (36,20)
    X_simulated = M_expr[cell_types_obs]            # (N,20)

    obs = pd.DataFrame({
        "domain_id": domain_ids.astype(str),
        "cell_type": cell_types_obs.astype(str)
    }, index=[f"spot_{i}" for i in range(N)])
    var = pd.DataFrame(index=[f"feature_{i}" for i in range(d_expr)])
    adata = ad.AnnData(X=X_simulated, obs=obs, var=var)
    adata.obsm["spatial"] = coords
    adata.obsm["z_embeddings"] = z_embeddings
    adata.obsp["spatial_connectivities"] = adjacency
    adata.uns["simulation_params"] = {
        "K1": K1,
        "K2": K2,
        "K3": K3,
        "C": C,
        "spots_per_domain": spots_per_domain,
        "H_dom": H_dom,
        "W_dom": W_dom,
        "domains_per_side": domains_per_side,
        "DIM_Z": DIM_Z,
        "d_expr": d_expr
    }
    adata.obs['x'] = coords[:, 0]
    adata.obs['y'] = coords[:, 1]
    
    # ------------------------------------------------------------
    # 6) (Option) simple sanity check / visualization
    # ------------------------------------------------------------
    print(f"N spots              : {N}")
    print("coords shape         :", coords.shape)
    print("z_embeddings shape   :", z_embeddings.shape)
    print("X_simulated shape    :", X_simulated.shape)
    print("adjacency nnz        :", adjacency.nnz)

    def load_data_prev_simulation(X, adjacency):
        # Ensure adjacency is in COO format
        adjacency_coo = adjacency.tocoo()
        edge_index = np.vstack((adjacency_coo.row, adjacency_coo.col))
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    
        # Convert the DataFrame to a NumPy array and then to a PyTorch tensor
        x = torch.tensor(X, dtype=torch.float)
        return Data(x=x, edge_index=edge_index)
        
    data = load_data_prev_simulation(X_simulated, adjacency)

    dataloader = NeighborLoader( 
        data,
        input_nodes=torch.arange(data.num_nodes), # [0, 1, 2, ..., n_obs-1]
        num_neighbors=[10,5],                     # Node sampling for each GNN layer
        batch_size=2048,                          # Number of center nodes for each batch
        shuffle=True
    )
    
    return adata, data, dataloader, coords, domain_ids, cell_types_obs

step2_run_prev_simulation = step2_run
step3_postprocess_prev_simulation = step3_postprocess



# ----------------------------
# auxiliary functions 
# ----------------------------

def eval_coenrichment(model_ff, p, cell_types_vae, cell_types_obs, num_topics=32, THRESH_PAIR=0.05, delta=0.4):
    def get_significant_topics(p, num_topics, manual_threshold_weight=1):
        p_numpy = p.detach().cpu().numpy()
        dynamic_threshold = manual_threshold_weight * 1.0 / num_topics 
        significant_topics = set()
        for spot_weights in p_numpy:
            indices = np.where(spot_weights > dynamic_threshold)[0]
            significant_topics.update(indices)
        return significant_topics

    def make_pred_label_map(cell_types_vae, cell_types_obs, delta=delta, default_label="Mixed", sep="/"):
        cell_types_vae = np.asarray(cell_types_vae)
        cell_types_obs = np.asarray(cell_types_obs)
    
        if len(cell_types_vae) != len(cell_types_obs):
            raise ValueError("The length of cell_types_vae and cell_types_obs must be the same.")
    
        pred_label_map = {}
    
        for vae_ct in np.unique(cell_types_vae):
            obs_in_cluster = cell_types_obs[cell_types_vae == vae_ct]
    
            if len(obs_in_cluster) == 0:
                pred_label_map[int(vae_ct)] = default_label
                continue
    
            counts = Counter(obs_in_cluster)
            total = len(obs_in_cluster)
    
            labels_above_delta = []
            for label, count in counts.most_common():
                ratio = count / total
                if ratio >= delta:
                    labels_above_delta.append(label)
    
            if len(labels_above_delta) > 0:
                pred_label_map[int(vae_ct)] = sep.join(labels_above_delta)
            else:
                pred_label_map[int(vae_ct)] = default_label
    
        return pred_label_map
    
    def cell_type_pairs(p, num_topics, pred_label_map):
        target_topics = get_significant_topics(p, num_topics)
        pair_results = []
        n_celltypes, n_topics = data_matrix.shape
        
        for pp in range(n_topics):
            valid_cells = [a for a in range(n_celltypes) if data_matrix[a, pp] > THRESH_PAIR]
            for a, b in itertools.combinations(valid_cells, 2):
                pair_results.append({
                    "topic": pp,
                    "celltype_a_idx": a,
                    "celltype_a": pred_label_map[a],
                    "phi_a": data_matrix[a, pp],
                    "celltype_b_idx": b,
                    "celltype_b": pred_label_map[b],
                    "phi_b": data_matrix[b, pp],
                })
        
        pair_df = pd.DataFrame(pair_results)
        if len(pair_df) > 0:
            pair_df["min_phi"] = pair_df[["phi_a", "phi_b"]].min(axis=1)
            pair_df["mean_phi"] = pair_df[["phi_a", "phi_b"]].mean(axis=1)
            pair_df = pair_df.sort_values(["topic", "min_phi", "mean_phi"], ascending=[True, False, False]).reset_index(drop=True)
        
        print(pair_df)
        return pair_df
    
    data_matrix = F.softmax(model_ff.fc1.weight.detach().cpu(), dim=0).numpy()
    pred_label_map = make_pred_label_map(cell_types_vae, cell_types_obs)
    return cell_type_pairs(p, num_topics, pred_label_map), pred_label_map


def eval_coenrichment_prev_simulation(model_ff, p, cell_types_vae, cell_types_obs, num_topics=32, viz_threshold=0.01, manual_threshold_weight=1.0):
    '''
    Notes:
        Don't run it multiple times with the same model because model_ff changed.
    '''
    
    # ============================================================
    # 0) Functions
    # ============================================================
    def reorder_labels(
            new_labels,
            ref_labels,
            new_categories=None,
            ref_categories=None,
            EPS=1e-3):
        """
        Reorder the labels of new_labels to best match ref_labels.
        Uses the Hungarian algorithm (maximum weight bipartite matching)
        on the contingency matrix between new_labels and ref_labels.
        
        Parameters:
            new_labels: array-like
            ref_labels: array-like
            new_categories: list or array of label values (optional)
            ref_categories: list or array of label values (optional)
            EPS: small value to ensure non-zero contingency (default: 1e-3)
        
        Returns:
            re_ordering: index array for new_categories reordered to match ref_categories
            one_to_one_mapping: dict mapping from new label to reference label
        """
        import numpy as np
        import warnings
        from scipy.optimize import linear_sum_assignment
    
        # Determine unique categories if not provided
        if new_categories is None:
            new_categories = np.sort(np.unique(new_labels))
        if ref_categories is None:
            ref_categories = np.sort(np.unique(ref_labels))
    
        # Build contingency matrix [ref x new]
        cm = np.zeros((len(ref_categories), len(new_categories)), dtype=float)
        for i, r in enumerate(ref_categories):
            for j, n in enumerate(new_categories):
                cm[i, j] = np.sum((ref_labels == r) & (new_labels == n))
        cm += EPS  # ensure all entries are positive
    
        # Hungarian algorithm minimizes cost; we want to maximize overlap → negate the matrix
        cost_matrix = -cm
    
        # If the matrix is not square, pad it with dummy rows or columns
        n_rows, n_cols = cost_matrix.shape
        if n_rows < n_cols:  # pad rows
            pad = np.full((n_cols - n_rows, n_cols), -EPS)
            cost_matrix = np.vstack([cost_matrix, pad])
        elif n_cols < n_rows:  # pad columns
            pad = np.full((n_rows, n_rows - n_cols), -EPS)
            cost_matrix = np.hstack([cost_matrix, pad])
    
        # Perform assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
        # Filter only the actual label pairs (ignore padded dummy assignments)
        paired = [(r, c) for r, c in zip(row_ind, col_ind)
                  if r < len(ref_categories) and c < len(new_categories)]
    
        if len(paired) < max(len(ref_categories), len(new_categories)):
            warnings.warn("Warning: Some categories could not be matched.")
    
        # Create one-to-one label mapping
        one_to_one_mapping = {
            new_categories[c]: ref_categories[r] for r, c in paired
        }
    
        return one_to_one_mapping

    def get_significant_topics(p, num_topics):
        p_numpy = p.detach().cpu().numpy()
        dynamic_threshold = manual_threshold_weight * 1.0 / num_topics 
        significant_topics = set()
        for spot_weights in p_numpy:
            indices = np.where(spot_weights > dynamic_threshold)[0]
            significant_topics.update(indices)
        return significant_topics
    
    def get_predicted_pairs_top3(p, weights, num_topics, threshold=None, indices=None):
        H, T = weights.shape # H: hidden units, T: topics
        
        # choose which topic columns to use
        target_topics = get_significant_topics(p, num_topics) if indices is None else indices
    
        all_predicted_pairs = set()
        for col in target_topics:
            col_w = weights[:, col]
            
            if threshold is None:
                valid_rows = np.arange(H)
            else:
                valid_rows = np.where(col_w >= threshold)[0]

            if len(valid_rows) < 2:
                continue
            
            valid_weights = col_w[valid_rows]
            if len(valid_rows) > 3:
                top_idx = np.argsort(valid_weights)[::-1][:3]
            else:                
                top_idx = np.argsort(valid_weights)[::-1]
            top_rows = valid_rows[top_idx]
            
            for pair in itertools.combinations(top_rows, 2):
                all_predicted_pairs.add(tuple(sorted(pair)))
                
        return all_predicted_pairs

    def predicted_cell_type_pairs(p, model_ff, num_topics, threshold=None, indices=None):
        weights = F.softmax(model_ff.fc1.weight.detach().cpu(), dim=0).numpy()
        return get_predicted_pairs_top3(p, weights, num_topics, threshold=threshold)
    
    # ============================================================
    # 1) Modification to model_ff
    # ============================================================
    one_to_one_mapping = reorder_labels(cell_types_vae, cell_types_obs, ref_categories=[i for i in range(36)])
    
    # Copy the existing weight
    original_weight = model_ff.fc1.weight.data.clone()
    
    # Rearrange weight according to mapping order
    shuffled_weight = torch.zeros_like(original_weight)
    for i in range(original_weight.size(0)):  # based on rows
        try:
            src_idx = int(one_to_one_mapping[str(i)])
            shuffled_weight[src_idx] = original_weight[i]
        except:
            src_idx = 0
            #shuffled_weight[src_idx] = original_weight[i]
    
    # Apply reordered weight to model
    model_ff.fc1.weight.data = shuffled_weight

    W = F.softmax(model_ff.fc1.weight.detach().cpu(), dim=0).numpy()
    sns.heatmap(W, vmin=0, vmax=viz_threshold, cmap="viridis")  
    plt.show()
    plt.close()
    
    weights = F.softmax(model_ff.fc1.weight.detach().cpu(), dim=0).numpy()
    #max_vals = weights.max(axis=0)  # shape: (16,)
    thresholds = viz_threshold #viz_threshold * max_vals
    binary_weights = (weights >= thresholds).astype(int)
    sns.heatmap(binary_weights, cmap="Greys", cbar=False)
    plt.title("Binarized Heatmap")
    plt.show()
    plt.close()


    # ============================================================
    # 2) Evaluation based on viz_threshold
    # ============================================================
    print("@Evaluation based on viz_threshold")
    
    # 1. Get cell type coenrichment pairs
    print(get_significant_topics(p, num_topics))
    all_predicted_pairs = predicted_cell_type_pairs(p, model_ff, indices=None, num_topics=num_topics, threshold=viz_threshold)
    
    # 2. True pair definition
    true_pairs = {
        (0, 9), (0,10), (9,10),
        (1,11), (1,12), (11,12),
        (2,13), (2,14), (13,14),
        (3,15), (3,16), (15,16),
        (4,17), (4,18), (17,18),
        (5,19), (5,20), (19,20),
        (6,21), (6,22), (21,22),
        (7,23), (7,24), (23,24),
        (8,25), (8,26), (25,26)
    }
    
    # 3. Get all possible pairs (36 choose 2)
    all_possible_pairs = set(itertools.combinations(range(36), 2))
    
    # 4. Calculate TP, FP, FN, TN
    TP = len(all_predicted_pairs & true_pairs)
    FP = len(all_predicted_pairs - true_pairs)
    FN = len(true_pairs - all_predicted_pairs)
    TN = len(all_possible_pairs - (true_pairs | all_predicted_pairs))
    
    # 5. Calculate accuracy 
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / len(all_possible_pairs)
    f1 = 2 * precision * recall / (precision + recall)
    
    # 6. Print results
    print(f"True Positive (TP): {TP}")
    print(f"False Positive (FP): {FP}")
    print(f"False Negative (FN): {FN}")
    print(f"True Negative (TN): {TN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # ============================================================
    # 3) Evaluation irrelevant to viz_threshold
    # ============================================================
    print("@Evaluation irrelevant to viz_threshold")
    
    true_pairs = {
        (0, 9), (0,10), (9,10),
        (1,11), (1,12), (11,12),
        (2,13), (2,14), (13,14),
        (3,15), (3,16), (15,16),
        (4,17), (4,18), (17,18),
        (5,19), (5,20), (19,20),
        (6,21), (6,22), (21,22),
        (7,23), (7,24), (23,24),
        (8,25), (8,26), (25,26)
    }
    
    all_possible_pairs = set(itertools.combinations(range(36), 2))
    
    thresholds = np.linspace(0, 1.0, 200)
    
    results = []
    
    for threshold in thresholds:
        predicted_pairs = predicted_cell_type_pairs(
            p, model_ff, indices=None,
            num_topics=num_topics, threshold=threshold
        )
    
        TP = len(predicted_pairs & true_pairs)
        FP = len(predicted_pairs - true_pairs)
        FN = len(true_pairs - predicted_pairs)
        TN = len(all_possible_pairs - (true_pairs | predicted_pairs))
    
        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (TP + TN) / len(all_possible_pairs)
    
        results.append({
            "threshold": threshold,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        })
    
    df_results = pd.DataFrame(results)
    
    pr_df = (
        df_results[["threshold", "precision", "recall", "f1"]]
        .sort_values(["recall", "precision"])
        .drop_duplicates(subset=["recall", "precision"])
    )
    
    if len(pr_df) < 2:
        print("AUPRC calculation impossible: PR curve points less than 2.")
        auprc = np.nan
    else:
        auprc = auc(pr_df["recall"], pr_df["precision"])
        print(f"AUPRC: {auprc:.4f}")
    
    print(df_results)
    
    best_f1_row = df_results.loc[df_results["f1"].idxmax()]
    print("\nBest threshold by F1-score")
    print(best_f1_row)
    
    plt.figure(figsize=(6, 5))
    plt.plot(pr_df["recall"], pr_df["precision"], marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AUPRC = {auprc:.4f})" if not np.isnan(auprc) else "Precision-Recall Curve")
    plt.grid(True)
    plt.show()
    plt.close()


def viz_crosstab(cell_types_vae, cell_types_obs, n_celltypes=None, thresh=30, save=False):
    import numpy as np
    import re
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
    
    # ----------------------------
    # 0) parameters: removal threshold / distance / linkage
    # ----------------------------
    THRESH = thresh                # remove rows/columns whose sum is below this value
    DIST_METRIC = "correlation"     # 'euclidean', 'cosine', 'correlation', etc.
    LINKAGE_METHOD = "average" # 'single','complete','average','ward', etc. (if using cosine, avoid ward)
    if n_celltypes is None:
        n_celltypes = len(np.unique(cell_types_vae))
        
    # ----------------------------
    # 1) manual annotation map (`Predicted` integer → string) 
    # ----------------------------
    pred_label_map = {i: f"{i}" for i in range(20)} # modify this if needed
    
    # ----------------------------
    # 2) prepare Series
    # ----------------------------
    true_series = pd.Series(cell_types_obs, name="True")
    pred_series = pd.Series(cell_types_vae.astype(int), name="Predicted")
    
    # ----------------------------
    # 3) generate corretab (row=Predicted, column=True)
    # ----------------------------
    cont = pd.crosstab(index=pred_series, columns=true_series)
    cont.index = cont.index.map(pred_label_map)
    
    # primary sorting by a visually pleasing row order (0–19)
    pred_order = [pred_label_map[i] for i in range(20)]
    cont = cont.reindex(pred_order)
    
    # sort the columns once as well (natural-number order) 
    # → then overwrite again during the subsequent clustering-based reordering
    def natural_keys(text):
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', text)]
    cont = cont.reindex(sorted(cont.columns, key=natural_keys), axis=1)
    
    # ----------------------------
    # 4) remove rows/columns based on the threshold
    # ----------------------------
    # first column filtering
    col_mask = cont.sum(axis=0) >= THRESH
    cont = cont.loc[:, col_mask]
    
    # subsequent row filtering
    row_mask = cont.sum(axis=1) >= THRESH
    cont = cont.loc[row_mask, :]
    
    # stop if empty after filtering
    if cont.shape[0] == 0 or cont.shape[1] == 0:
        raise ValueError(
            f"After thresholding (THRESH={THRESH}), confusion matrix is empty: "
            f"{cont.shape}. Try lowering THRESH."
        )
    
    # ----------------------------
    # 5) determine the column and row order using hierarchical clustering (optimal leaf ordering)
    # ----------------------------
    def hierarchical_reorder(df, on="columns",
                             dist_metric=DIST_METRIC,
                             link_method=LINKAGE_METHOD):
        """
        df: DataFrame (counts)
        on: 'columns' or 'index' (rows)
        return: list of reordered labels
        """
        if on == "columns":
            X = df.T.values  # treat each column as a vector (along the row dimension)
            labels = df.columns.to_list()
        elif on == "index":
            X = df.values    # treat each row as a vector (along the column dimension)
            labels = df.index.to_list()
        else:
            raise ValueError("on must be 'columns' or 'index'")
    
        # check the minimum dimensionality required for clustering
        if X.shape[0] < 2:
            return labels  # as is if 1
    
        # Add a small safeguard in case distance computation becomes invalid due to all-zero vectors (zero variance), etc.
        # (Rows/columns with sum 0 have already been removed, but this also covers cases where all values are identical.)
        # Here, we proceed as is; if a problem occurs, try changing the metric to 'euclidean'.
        D = pdist(X, metric=dist_metric)             # (n*(n-1)/2,) condensed distance
        Z = linkage(D, method=link_method)           # dendrogram
        Z_opt = optimal_leaf_ordering(Z, D)          # optimal leaf order
        order = leaves_list(Z_opt)                   # index order
    
        return [labels[i] for i in order]
    
    # rearrange columns
    new_cols = hierarchical_reorder(cont, on="columns",
                                    dist_metric=DIST_METRIC,
                                    link_method=LINKAGE_METHOD)
    cont = cont.loc[:, new_cols]
    
    # rearrange rows
    new_rows = hierarchical_reorder(cont, on="index",
                                    dist_metric=DIST_METRIC,
                                    link_method=LINKAGE_METHOD)
    cont = cont.loc[new_rows, :]
    
    # ----------------------------
    # 6) vizualize heatmap
    # ----------------------------
    sns.set(context="talk", style="white")
    
    plt.figure(figsize=(16, 11))
    #cont_norm = cont.div(cont.sum(axis=0), axis=1)
    ax = sns.heatmap(
        cont,
        cmap='Spectral_r',      # 네가 쓰던 컬러맵 유지
        annot=True,
        fmt="d", #".1f"
        cbar=True,
        square=False,
        linewidths=0,
        annot_kws={"size":10},
        cbar_kws={"shrink": 0.8, "pad": 0.02}
    )
    
    ax.set_title("Confusion Matrix", pad=14, fontsize=27)
    ax.set_xlabel("Given Cell Type Label", fontsize=20)
    ax.set_ylabel("Predicted Cell Type Label", fontsize=20)
    
    # tick / label
    ax.set_xticks(np.arange(cont.shape[1]) + 0.5)
    ax.set_xticklabels(cont.columns.astype(str), rotation=90, ha="right", fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=15)
    
    # trim boundary
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.grid(False)
    
    plt.tight_layout()

    if save:
        plt.savefig("Confusion_Matrix_Given_Predicted.png", dpi=600,
                    bbox_inches='tight', pad_inches=0.01)
        plt.savefig("Confusion_Matrix_Given_Predicted.pdf", dpi=300,
                    bbox_inches='tight', pad_inches=0.01)
    
    plt.show()
    plt.close()
    

def viz_crosstab_hypothalamus(cell_types_vae, cell_types_obs, n_celltypes=None, thresh=30, save=False):
    import numpy as np
    import re
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
    
    # ----------------------------
    # 0) parameters: removal threshold / distance / linkage
    # ----------------------------
    THRESH = thresh                # remove rows/columns whose sum is below this value
    DIST_METRIC = "correlation"     # 'euclidean', 'cosine', 'correlation', etc.
    LINKAGE_METHOD = "average" # 'single','complete','average','ward', etc. (if using cosine, avoid ward)
    if n_celltypes is None:
        n_celltypes = len(np.unique(cell_types_vae))
    
    # ----------------------------
    # 1) manual annotation map (`Predicted` integer → string) 
    # ----------------------------
    pred_label_map = {i: f"{i}" for i in range(n_celltypes)} # modify this if needed
    
    # ----------------------------
    # 2) prepare Series
    # ----------------------------
    true_series = pd.Series(cell_types_obs, name="")
    pred_series = pd.Series(cell_types_vae.astype(int), name="Predicted")
    
    # ----------------------------
    # 3) generate corretab (row=Predicted, column=True)
    # ----------------------------
    cont = pd.crosstab(index=pred_series, columns=true_series)
    cont.index = cont.index.map(pred_label_map)
    
    # primary sorting by a visually pleasing row order (0–19)
    pred_order = [pred_label_map[i] for i in range(n_celltypes)]
    cont = cont.reindex(pred_order)
    
    # sort the columns once as well (natural-number order) 
    # → then overwrite again during the subsequent clustering-based reordering
    def natural_keys(text):
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', text)]
    cont = cont.reindex(sorted(cont.columns, key=natural_keys), axis=1)
    
    # ----------------------------
    # 4) remove rows/columns based on the threshold
    # ----------------------------
    # first column filtering
    col_mask = cont.sum(axis=0) >= THRESH
    cont = cont.loc[:, col_mask]
    
    # subsequent row filtering
    row_mask = cont.sum(axis=1) >= THRESH
    cont = cont.loc[row_mask, :]
    
    # stop if empty after filtering
    if cont.shape[0] == 0 or cont.shape[1] == 0:
        raise ValueError(
            f"After thresholding (THRESH={THRESH}), confusion matrix is empty: "
            f"{cont.shape}. Try lowering THRESH."
        )
    
    # ----------------------------
    # 5) determine the column and row order using hierarchical clustering (optimal leaf ordering)
    # ----------------------------
    def hierarchical_reorder(df, on="columns",
                             dist_metric=DIST_METRIC,
                             link_method=LINKAGE_METHOD):
        """
        df: DataFrame (counts)
        on: 'columns' or 'index' (rows)
        return: list of reordered labels
        """
        if on == "columns":
            X = df.T.values  # treat each column as a vector (along the row dimension)
            labels = df.columns.to_list()
        elif on == "index":
            X = df.values    # treat each row as a vector (along the column dimension)
            labels = df.index.to_list()
        else:
            raise ValueError("on must be 'columns' or 'index'")
    
        # check the minimum dimensionality required for clustering
    
        if X.shape[0] < 2:
            return labels  # as is if 1
    
        # Add a small safeguard in case distance computation becomes invalid due to all-zero vectors (zero variance), etc.
        # (Rows/columns with sum 0 have already been removed, but this also covers cases where all values are identical.)
        # Here, we proceed as is; if a problem occurs, try changing the metric to 'euclidean'.
        D = pdist(X, metric=dist_metric)             # (n*(n-1)/2,) condensed distance
        Z = linkage(D, method=link_method)           # dendrogram
        Z_opt = optimal_leaf_ordering(Z, D)          # optimal leaf order
        order = leaves_list(Z_opt)                   # index order
    
        return [labels[i] for i in order]
    
    # rearrange columns
    new_cols = hierarchical_reorder(cont, on="columns",
                                    dist_metric=DIST_METRIC,
                                    link_method=LINKAGE_METHOD)
    cont = cont.loc[:, new_cols]
    
    # rearrange rows
    new_rows = hierarchical_reorder(cont, on="index",
                                    dist_metric=DIST_METRIC,
                                    link_method=LINKAGE_METHOD)
    cont = cont.loc[new_rows, :]
    
    
    # ----------------------------
    # 6) vizualize heatmap
    # ----------------------------
    
    # rearrange columns in manual order
    manual_cols = [
        'I-16', 'I-2', 'I-23', 'I-13', 'I-9', 'I-18', 'I-15', 'I-10', 'I-11', 'I-12', 'I-14', 'I-7',
        'I-3', 'I-1', 'I-8', 'I-4',
        'E-17', 'E-12', 'E-15', 'E-8', 'E-7', 'E-4', 'E-1', 'E-24', 'E-18', 'E-3', 'E-16', 'E-19',
        'E-13', 'E-23', 'E-9', 'E-14', 
        'OD Mature 1', 'OD Mature 2', 'Microglia', 'Astrocyte',
        'Endothelial 3', 'Endothelial 1', 'Ependymal', 'OD Immature 1'
    ]
    
    # use only the columns that actually exist in `cont`, in the order listed above
    manual_cols_in_cont = [c for c in manual_cols if c in cont.columns]
    cont = cont.reindex(columns=manual_cols_in_cont)
    
    # assign group labels by column (`I-` / `E-` / others = `Non-neurons`)
    col_groups = []
    for c in cont.columns:
        if c.startswith("I-"):
            col_groups.append("Inhibitory neurons")
        elif c.startswith("E-"):
            col_groups.append("Excitatory neurons")
        else:
            col_groups.append("Non-neurons")
    
    # set the style and arrange the axes using `GridSpec`
    sns.set(context="talk", style="white")
    
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[0.985, 0.015],   # top: heatmap; bottom: group bar + x-axis ticks
        width_ratios=[20, 1],         # left: main panel; right: colorbar
        wspace=0.15,
        hspace=0.05
    )
    
    # main heatmap axis, group bar axis, and colorbar axis
    ax_hm    = fig.add_subplot(gs[0, 0])
    ax_group = fig.add_subplot(gs[1, 0], sharex=ax_hm)
    cax      = fig.add_subplot(gs[:, 1])
    
    # heatmap (keep the existing colormap and style)
    hm = sns.heatmap(
        cont,
        cmap='Spectral_r',      # keep the existing colormap
        annot=True,
        fmt="d",
        cbar=True,
        cbar_ax=cax,            # a colorbar that spans the full height on the right
        square=False,
        linewidths=0,
        annot_kws={"size":10},
        cbar_kws={"shrink": 0.8}
        ,
        ax=ax_hm
    )
    
    ax_hm.set_title("Confusion Matrix", pad=14, fontsize=27)
    ax_hm.set_ylabel("Predicted Cell Type Label", fontsize=20)
    
    # hide the heatmap’s x-axis and show it on the group bar axis below.
    ax_hm.tick_params(axis='x', bottom=False, labelbottom=False)
    
    # y tick font
    ax_hm.set_yticklabels(ax_hm.get_yticklabels(), rotation=0, fontsize=15)
    
    # clean up the borders
    for spine in ax_hm.spines.values():
        spine.set_visible(False)
    ax_hm.grid(False)
    
    # draw a group bar between the heatmap and the x-axis labels (`ax_group`)
    group_color_map = {
        "Inhibitory neurons": "#377eb8",
        "Excitatory neurons": "#e41a1c",
        "Non-neurons":       "#4daf4a",
    }
    
    n_cols = cont.shape[1]
    
    # set up the group bar axis
    ax_group.set_xlim(0, n_cols)
    ax_group.set_ylim(0, 1)
    ax_group.set_yticks([])
    ax_group.set_ylabel("")
    
    # set the x-ticks and labels here (below the heatmap)
    ax_group.set_xticks(np.arange(n_cols) + 0.5)
    ax_group.set_xticklabels(cont.columns.astype(str), rotation=90,
                             ha="right", fontsize=12)
    ax_group.set_xlabel("Given Cell Type Label", fontsize=20)
    
    # one long box for each consecutive segment of the same group
    start_idx = 0
    current_group = col_groups[0]
    
    for i in range(1, n_cols + 1):
        if i == n_cols or col_groups[i] != current_group:
            end_idx = i  # exclusive
            ax_group.add_patch(
                plt.Rectangle(
                    (start_idx, 0),           # (x0, y0)
                    end_idx - start_idx,      # width
                    1,                        # height
                    transform=ax_group.transData,
                    facecolor=group_color_map[current_group],
                    edgecolor='none',
                    alpha=0.7
                )
            )
            if i < n_cols:
                start_idx = i
                current_group = col_groups[i]
    
    # remove the borders and grid from the group bar axis
    for spine in ax_group.spines.values():
        spine.set_visible(False)
    ax_group.grid(False)
    
    # column group legend: place it below the colorbar
    handles = [
        mpatches.Patch(color=group_color_map["Inhibitory neurons"], label="Inhibitory neurons"),
        mpatches.Patch(color=group_color_map["Excitatory neurons"], label="Excitatory neurons"),
        mpatches.Patch(color=group_color_map["Non-neurons"],       label="Non-neurons"),
    ]
    
    # place the legend directly below `cax` (the colorbar axis)
    cax.legend(
        handles=handles,
        title="Column groups",
        loc="upper center",
        bbox_to_anchor=(0.8, 0),   # y<0 이면 컬러바 아래로 내려감
        frameon=False
    )
    
    plt.tight_layout()

    if save:
        plt.savefig("Confusion_Matrix_Given_Predicted_Column_Reordered.pdf", dpi=300,
                    bbox_inches='tight', pad_inches=0.01)
        plt.savefig("Confusion_Matrix_Given_Predicted_Column_Reordered.png", dpi=300,
                    bbox_inches='tight', pad_inches=0.01)
    
    plt.show()
    plt.close()


def viz_hierarchical_domain(z, x, y, K_final=15, colorspace="hsv", viz_dendrogram=True, viz_spatial=True, save_fig=False, figurename="figure.png"):
    # ───────────────────── Library ─────────────────────
    import numpy  as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.colors  as mcolors
    import matplotlib.patches as mpatches
    from collections import defaultdict
    from sklearn.preprocessing    import StandardScaler
    from sklearn.cluster          import MiniBatchKMeans
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.cluster.hierarchy import to_tree
    
    def STEP1_prototype_compression_hierarchical_clustering(z, n_proto=1000, random_state=42):
        scaler    = StandardScaler()
        z_std     = scaler.fit_transform(z)
        mbk       = MiniBatchKMeans(n_clusters=n_proto, batch_size=8192,
                                    random_state=random_state, n_init="auto")
        proto_id  = mbk.fit_predict(z_std)                 # each cell → prototype ID
        protos    = mbk.cluster_centers_                  # (n_proto, n_features)
        link_mat  = linkage(protos, method="ward", metric="euclidean")     
        return proto_id, link_mat
    
    def STEP2_specify_k_final_clusters(link_mat, K_final=K_final):
        proto_cluster  = fcluster(link_mat, K_final, criterion="maxclust")  # (n_proto,)
        clust_to_protos = defaultdict(list)
        for pid, c in enumerate(proto_cluster):
            clust_to_protos[c].append(pid)
        # "Representative leaf number" of each final cluster (smallest prototype ID)
        clust_rep_leaf = {c: min(leaf_ids) for c, leaf_ids in clust_to_protos.items()}
        rep_leaves_set = set(clust_rep_leaf.values())     
        return proto_cluster, clust_rep_leaf, rep_leaves_set
    
    def STEP3_color_palette_mapping(link_mat):
        def _assign_color(node, h0, h1, depth, out, colorspace=colorspace):
            """
            Recursively assign a unique color to each leaf.
            
            Parameters
            ----------
            node : scipy.cluster.hierarchy.ClusterNode
            h0, h1 : float
                Start and end hue (or analogous range) values
            depth : int
                Current depth in tree (not used here but could help for other color logic)
            out : dict
                Mapping of node.id → RGB tuple
            colorspace : str
                Color space to use. One of ["hsv", "hsl", "lab"]
            """
            import colorsys
            from colorsys import hsv_to_rgb, hls_to_rgb
            from matplotlib import colors as mcolors
            try:
                from colormath.color_objects import LabColor, sRGBColor
                from colormath.color_conversions import convert_color
            except ImportError:
                pass
            
            if node.is_leaf():
                h = (h0 + h1) / 2
        
                if colorspace == "hsv":
                    rgb = hsv_to_rgb(h, 0.55, 0.90)
        
                elif colorspace == "hsl":
                    rgb = hls_to_rgb(h, 0.72, 0.55)  # lightness, saturation
        
                elif colorspace == "lab":
                    # L fixed, a and b interpolated
                    a = -60 + h * 120  # -60 ~ +60
                    b =  40 - h * 80   # 40 ~ -40
                    lab = LabColor(lab_l=75, lab_a=a, lab_b=b)
                    rgb_obj = convert_color(lab, sRGBColor)
                    rgb = (rgb_obj.clamped_rgb_r, rgb_obj.clamped_rgb_g, rgb_obj.clamped_rgb_b)
        
                else:
                    raise ValueError(f"Unsupported colorspace: {colorspace}")
        
                out[node.id] = rgb
        
            else:
                mid = (h0 + h1) / 2
                _assign_color(node.get_left(),  h0,  mid, depth + 1, out, colorspace)
                _assign_color(node.get_right(), mid, h1,  depth + 1, out, colorspace)
        
        root, _ = to_tree(link_mat, rd=True)   # deterministic tree object
        proto_rgb = {}
        _assign_color(root, 0.0, 1.0, 0, proto_rgb)      # {leaf_id: (r,g,b)}
        proto_to_color = {pid: mcolors.to_hex(rgb) for pid, rgb in proto_rgb.items()}
        cluster_to_color = {c: proto_to_color[clust_rep_leaf[c]] for c in clust_rep_leaf}
        return proto_to_color, cluster_to_color
    
    def STEP4_visualization(x, y, proto_id, proto_to_color,
                            proto_cluster, clust_rep_leaf,
                            cluster_to_color, link_mat,
                            viz_dendrogram=viz_dendrogram, viz_spatial=viz_spatial,
                            save_fig=save_fig, figurename=figurename):
    
        rep_leaves_set = set(clust_rep_leaf.values())
        
        fig = plt.figure(figsize=(13, 6))
        n_cols = int(viz_dendrogram) + int(viz_spatial)
        gs = fig.add_gridspec(1, n_cols, width_ratios=[1]*n_cols, wspace=0.25)
    
        ax_idx = 0
    
        # ────── A. Dendrogram ──────
        if viz_dendrogram:
            ax_d = fig.add_subplot(gs[ax_idx])
            ax_idx += 1
    
            def leaf_color_func(id_):
                return proto_to_color.get(id_, "#555555")
    
            def leaf_label_func(id_):
                if id_ in rep_leaves_set:
                    return f"({id_})  ■"
                else:
                    return ""
    
            dendrogram(link_mat,
                       orientation="left",
                       leaf_label_func=leaf_label_func,
                       leaf_font_size=8,
                       link_color_func=leaf_color_func,
                       color_threshold=0,
                       ax=ax_d)
    
            ax_d.set_title("UPGMA dendrogram\n(representative leaves only)", fontsize=10)
            ax_d.tick_params(left=False, labelleft=False)
            ax_d.set_xlabel("Distance")
    
            for label in ax_d.get_yticklabels():
                try:
                    id_ = int(label.get_text().strip("()■ "))
                    if id_ in proto_to_color:
                        label.set_color(proto_to_color[id_])
                except:
                    continue
    
        # ────── B. Spatial Plot ──────
        if viz_spatial:
            ax_s = fig.add_subplot(gs[ax_idx])
    
            # Color and representative leaf
            hc_leaf = [clust_rep_leaf[proto_cluster[pid]] for pid in proto_id]
            hc_color = [proto_to_color[pid] for pid in proto_id]
    
            ax_s.scatter(x, y, c=hc_color, s=4, linewidth=0)
    
            # Adjust crop range 
            xr, yr = (x.max()-x.min()), (y.max()-y.min())
            
            # Visualize representative leaf number text
            cent = pd.DataFrame({"x": x, "y": y, "leaf": hc_leaf}) \
                     .groupby("leaf")[["x", "y"]].median()
    
            x_lo, x_hi = ax_s.get_xlim()
            y_lo, y_hi = ax_s.get_ylim()
    
            for lf, (cx, cy) in cent.iterrows():
                if x_lo < cx < x_hi and y_lo < cy < y_hi:
                    ax_s.text(cx, cy, f"({lf})",
                              ha="center", va="center",
                              fontsize=8, weight="bold")
    
            # Add legends
            handles = [mpatches.Patch(color=cluster_to_color[c],
                                      label=f"({clust_rep_leaf[c]})")
                       for c in sorted(cluster_to_color)]
            leg = ax_s.legend(handles=handles, title="Hierarchical leaf\n(representative)",
                              bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0.)
            leg._legend_box.align = "left"
    
            ax_s.axis("off")
            ax_s.set_title("Spatial feature plot – clusters labelled by leaf index", pad=6)
    
        # ────── Save or Show ──────
        plt.tight_layout()
        if save_fig:
            plt.savefig(filename, dpi=300)
        plt.show()
        plt.close()

    proto_id, link_mat = STEP1_prototype_compression_hierarchical_clustering(z)
    proto_cluster, clust_rep_leaf, rep_leaves_set = STEP2_specify_k_final_clusters(link_mat, K_final = K_final)
    proto_to_color, cluster_to_color = STEP3_color_palette_mapping(link_mat)
    STEP4_visualization(x, y, proto_id, proto_to_color,proto_cluster, clust_rep_leaf,
                        cluster_to_color, link_mat,viz_dendrogram, viz_spatial, save_fig)


def viz_celltype_spatial(cell_types_vae, cell_types_niche, x, y, save=False):
    pred_labels = cell_types_vae
    
    # arrange 0–19 in a fixed order on a 4×5 subplot grid, leaving blanks where missing
    labels_str = pred_labels.astype(str)
    order = [str(i) for i in range(20)]
    
    # palette
    palette_list = sns.color_palette("tab20", 20)
    palette = {str(i): palette_list[i] for i in range(20)}

    fig, axes = plt.subplots(4, 5, figsize=(18, 14), dpi=300)
    axes = axes.ravel()
    
    for i, ax in enumerate(axes):
        lab = str(i)
    
        # use a light gray background for the entire figure
        sns.scatterplot(
            x=x,
            y=y,
            color="lightgray",
            s=10,
            linewidth=0,
            ax=ax,
            legend=False
        )
    
        # overlay only the corresponding label in color
        m = (labels_str == lab)
        if np.any(m):
            sns.scatterplot(
                x=x[m],
                y=y[m],
                color=palette[lab],
                s=10,
                linewidth=0,
                ax=ax,
                legend=False
            )
    
        ax.set_title(lab, fontsize=20)
        ax.axis("off")
    
    plt.tight_layout()
    if save:
        plt.savefig("cell_types_vae_spatial.pdf", dpi=300,
                    bbox_inches='tight', pad_inches=0.01)
        plt.savefig("cell_types_vae_spatial.png", dpi=300,
                    bbox_inches='tight', pad_inches=0.01)
    plt.show()
    plt.close()
    
    pred_labels = cell_types_niche
    fig, axes = plt.subplots(4, 5, figsize=(18, 14), dpi=300)
    axes = axes.ravel()
    
    for i, ax in enumerate(axes):
        lab = str(i)
    
        # use a light gray background for the entire figure
        sns.scatterplot(
            x=x,
            y=y,
            color="lightgray",
            s=10,
            linewidth=0,
            ax=ax,
            legend=False
        )
    
        # overlay only the corresponding label in color
        m = (labels_str == lab)
        if np.any(m):
            sns.scatterplot(
                x=x[m],
                y=y[m],
                color=palette[lab],
                s=10,
                linewidth=0,
                ax=ax,
                legend=False
            )
    
        ax.set_title(lab, fontsize=20)
        ax.axis("off")
    
    plt.tight_layout()

    if save:
        plt.savefig("cell_types_niche_spatial.pdf", dpi=300,
                    bbox_inches='tight', pad_inches=0.01)
        plt.savefig("cell_types_niche_spatial.png", dpi=300,
                    bbox_inches='tight', pad_inches=0.01)
    plt.show()
    plt.close()


def viz_celltype_niche(model_ff, cell_types_vae, save=False):
    plt.rc('font', size=15)
    
    # 1) data
    data_matrix = F.softmax(model_ff.fc1.weight.detach().cpu(), dim=0).numpy()
    
    # 2) y axis label mapping
    n_celltypes = len(np.unique(cell_types_vae))
    pred_label_map = pred_label_map = {i: f"{i}" for i in range(n_celltypes)}
    
    # 3) significant/nonsignificant topic index 
    significant_topics = set(range(0, 32)) # modify it if needed e.g., {3,4,6,14}
    all_topics = set(range(data_matrix.shape[1]))
    nonsig_topics = sorted(list(all_topics - significant_topics))
    
    # 4) plot
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    hm = sns.heatmap(
        data_matrix,
        ax=ax,
        cmap='inferno',
        cbar=True,
        vmax=0.1
    )
    
    # 5) axis label
    ax.set_xlabel("Topic", fontsize=16)
    ax.set_ylabel("Cell Type", fontsize=16)
    ax.tick_params(axis='x', labelrotation=90)
    ax.tick_params(axis='y', labelrotation=0)
    
    # replace the y-axis labels (apply a 0.5 offset since the cell centers are at 0.5, 1.5, ...)
    ax.set_yticks([i + 0.5 for i in range(len(pred_label_map))])
    ax.set_yticklabels([pred_label_map[i] for i in range(len(pred_label_map))],
                       rotation=0, fontsize=10)
    ax.grid(False)
    
    # 6) shade non-significant topics with a light gray background, and set the tick colors to gray
    # lower the heatmap to the back (i.e., use a smaller `zorder`), and bring the gray spans to the front (i.e., use a larger `zorder`)
    hm.collections[0].set_zorder(1)
    
    n_cols = data_matrix.shape[1]
    assert all(0 <= j < n_cols for j in nonsig_topics)
    
    for j in nonsig_topics:
        ax.axvspan(
            j, j+1,
            ymin=0.0, ymax=1.0,             # whole column
            color='#D0D0D0', alpha=0.50,
            zorder=3,                       # 🔼 above than heatmap
            clip_on=False                   # prevent the plot edges from being clipped
        )
    
    # set the x-axis tick label color to gray
    for j, lbl in enumerate(ax.get_xticklabels()):
        if j in nonsig_topics:
            lbl.set_color('0.7')   # grey
        else:
            lbl.set_color('black')
    
    # (optional) uncomment this if you want to add a legend patch.
    # from matplotlib.lines import Line2D
    # legend_elements = [
    #     Line2D([0], [0], color='lightgray', lw=10, alpha=0.4, label='Non-significant topics'),
    # ]
    # ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1,1))
    
    # 7) save
    hm.collections[0].set_rasterized(True)  # rasterize only the heatmap in the PDF (for sharper rendering and smaller file size)
    if save:
        fig.savefig("celltype_topic_heatmap_marked.pdf", dpi=300, bbox_inches='tight', pad_inches=0.02)
        fig.savefig("celltype_topic_heatmap_marked.png", dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.show()
    plt.close()


def viz_annot_celltype_niche(model_ff, cell_types_vae, cell_types_obs, save=False):
    plt.rc('font', size=15)
    
    # 1) data
    data_matrix = F.softmax(model_ff.fc1.weight.detach().cpu(), dim=0).numpy()
    
    # 2) y axis label mapping
    def make_pred_label_map_(cell_types_vae, cell_types_obs, delta=0.2, default_label="Mixed", sep="/"):
        cell_types_vae = np.asarray(cell_types_vae)
        cell_types_obs = np.asarray(cell_types_obs)
    
        if len(cell_types_vae) != len(cell_types_obs):
            raise ValueError("The length of cell_types_vae and cell_types_obs must be the same.")
    
        pred_label_map = {}
    
        for vae_ct in np.unique(cell_types_vae):
            obs_in_cluster = cell_types_obs[cell_types_vae == vae_ct]
    
            if len(obs_in_cluster) == 0:
                pred_label_map[int(vae_ct)] = default_label
                continue
    
            counts = Counter(obs_in_cluster)
            total = len(obs_in_cluster)
    
            labels_above_delta = []
            for label, count in counts.most_common():
                ratio = count / total
                if ratio >= delta:
                    labels_above_delta.append(label)
    
            if len(labels_above_delta) > 0:
                pred_label_map[int(vae_ct)] = sep.join(labels_above_delta)
            else:
                pred_label_map[int(vae_ct)] = default_label
    
        return pred_label_map
    pred_label_map = make_pred_label_map_(cell_types_obs2, cell_types_obs)
    
    # 3) significant/nonsignificant topic index 
    significant_topics = set(range(0, 32)) # modify it if needed e.g., {3,4,6,14}
    all_topics = set(range(data_matrix.shape[1]))
    nonsig_topics = sorted(list(all_topics - significant_topics))
    
    # 4) plot
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    hm = sns.heatmap(
        data_matrix,
        ax=ax,
        cmap='inferno',
        cbar=True,
        vmax=0.1
    )
    
    # 5) axis label
    ax.set_xlabel("Topic", fontsize=16)
    ax.set_ylabel("Cell Type", fontsize=16)
    ax.tick_params(axis='x', labelrotation=90)
    ax.tick_params(axis='y', labelrotation=0)
    
    # replace the y-axis labels (apply a 0.5 offset since the cell centers are at 0.5, 1.5, ...)
    ax.set_yticks([i + 0.5 for i in range(len(pred_label_map))])
    ax.set_yticklabels([pred_label_map[i] for i in range(len(pred_label_map))],
                       rotation=0, fontsize=10)
    ax.grid(False)
    
    # 6) shade non-significant topics with a light gray background, and set the tick colors to gray
    # lower the heatmap to the back (i.e., use a smaller `zorder`), and bring the gray spans to the front (i.e., use a larger `zorder`)
    hm.collections[0].set_zorder(1)
    
    n_cols = data_matrix.shape[1]
    assert all(0 <= j < n_cols for j in nonsig_topics)
    
    for j in nonsig_topics:
        ax.axvspan(
            j, j+1,
            ymin=0.0, ymax=1.0,             # whole column
            color='#D0D0D0', alpha=0.50,
            zorder=3,                       # 🔼 above than heatmap
            clip_on=False                   # prevent the plot edges from being clipped
        )
    
    # set the x-axis tick label color to gray
    for j, lbl in enumerate(ax.get_xticklabels()):
        if j in nonsig_topics:
            lbl.set_color('0.7')   # grey
        else:
            lbl.set_color('black')
    
    # (optional) uncomment this if you want to add a legend patch.
    # from matplotlib.lines import Line2D
    # legend_elements = [
    #     Line2D([0], [0], color='lightgray', lw=10, alpha=0.4, label='Non-significant topics'),
    # ]
    # ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1,1))
    
    # 7) save
    hm.collections[0].set_rasterized(True)  # rasterize only the heatmap in the PDF (for sharper rendering and smaller file size)
    if save:
        fig.savefig("celltype_topic_heatmap_marked.pdf", dpi=300, bbox_inches='tight', pad_inches=0.02)
        fig.savefig("celltype_topic_heatmap_marked.png", dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.show()
    plt.close()


def viz_niches(p, x, y, save=False):
    n_niches = p.shape[1]   # e.g. 32
    nrows = math.ceil(n_niches / 8)
    ncols = math.ceil(n_niches / nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3), dpi=300)
    axes = axes.ravel()
    
    for idx, ax in enumerate(axes):
        if idx >= n_niches:
            ax.axis("off")
            continue
    
        channel = p[:, idx].detach().cpu().numpy()
    
        sc = ax.scatter(
            x,
            y,
            c=channel,
            s=1,
            cmap='viridis',
            linewidth=0
        )
    
        ax.set_title(f"Niche {idx}", fontsize=30)
        ax.axis('off')
    
    plt.tight_layout()

    if save:
        plt.savefig("niches_plot.png", dpi=300, bbox_inches="tight")
        plt.savefig("niches_plot.pdf", bbox_inches="tight")
    
    plt.show()
    plt.close()


def viz_single_niche(p, x, y, idx, threshold, save=False):
    channel = p[:, idx].detach().cpu().numpy()
        
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
        
    low_mask = channel < threshold
    high_mask = channel >= threshold
    
    ax.scatter(
        x[low_mask],
        y[low_mask],
        c='lightgray',
        s=1,
        linewidth=0
    )
    
    sc = ax.scatter(
        x[high_mask],
        y[high_mask],
        c=channel[high_mask],
        s=1,
        cmap='viridis',
        linewidth=0
    )
    
    ax.set_title(f"Niche {idx}", fontsize=20)
    ax.axis('off')
    fig.tight_layout()
    
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    
    if save:
        plt.savefig(f"niche_{idx}.pdf", dpi=300,
                    bbox_inches='tight', pad_inches=0.01)
        plt.savefig(f"niche_{idx}.png", dpi=300,
                    bbox_inches='tight', pad_inches=0.01)
    plt.show()
    plt.close()


def save_models(model, model_ff, model_ct, p_mat, base_dir='./', mark_time=True):
    os.makedirs(base_dir, exist_ok=True)
    if mark_time:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(model.state_dict(), os.path.join(base_dir, f"model_{timestamp}.pth"))
        torch.save(model_ff.state_dict(), os.path.join(base_dir, f"model_ff_{timestamp}.pth"))
        torch.save(model_ct.state_dict(), os.path.join(base_dir, f"model_ct_{timestamp}.pth"))
        torch.save(p_mat, os.path.join(base_dir, f"p_mat_{timestamp}.pt"))
    else:
        torch.save(model.state_dict(), os.path.join(base_dir, "model.pth"))
        torch.save(model_ff.state_dict(), os.path.join(base_dir, "model_ff.pth"))
        torch.save(model_ct.state_dict(), os.path.join(base_dir, "model_ct.pth"))
        torch.save(p_mat, os.path.join(base_dir, "p_mat.pt"))
