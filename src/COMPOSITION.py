import anndata
import cellcharter as cc
import copy
from itertools import chain
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from numba.core.errors import NumbaDeprecationWarning
import numpy as np
import pandas as pd
from pathlib import Path
import scanpy as sc
import scipy
from scipy.stats import norm, multivariate_normal, wishart, Covariance
from scipy.special import logsumexp
import seaborn
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering
import squidpy as sq
import sys
import torch
from torch import nn
from torch.autograd import Variable
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.nn import VGAE, GCNConv, InnerProductDecoder, Sequential, SAGEConv
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
from tqdm import trange
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

def step1_preprocess(adata_orig, X_pca=None, n_comps=20):
    """
    Args:
        adata_orig: Raw AnnData. If no adata_orig.layers['counts'], adata_orig.X should be raw counts

    Returns:
        data: for running w/o minibatch
        dataloader: for running w/ minibatch
    """
    
    adata = adata_orig.copy()
    if 'spatial' in adata.obsm and adata.obsm['spatial'].shape[1] == 2:
        adata.obs[['x', 'y']] = adata.obsm['spatial']
    else:
        ensure_xy_from_obs(adata)
        adata.obsm['spatial'] = adata.obs[['x', 'y']].values
    adata_orig.obs[['x', 'y']] = adata.obs[['x', 'y']] # for use outside this function
    sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, spatial_key='spatial')
    cc.gr.remove_long_links(adata)
    adjacency = adata.obsp['spatial_connectivities']

    if X_pca is not None:
        X = X_pca
    elif 'X_pca' in adata.obsm:
        X = adata.obsm['X_pca']
    else:
        if 'counts' in adata.layers:
            adata.X = adata.layers['counts'].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, zero_center=True, max_value=10)
        sc.tl.pca(adata, n_comps=n_comps)
        X = adata.obsm['X_pca']
    
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

def step2_run(data, dataloader, seed=1, hid_dim=128, num_topics=16, n_celltypes=20, minibatch=False, temperature=0.3, early_stopping=False, alpha=1, wloss_spatial=0.8, wloss_KLD=0.005, wloss_recon=1, wloss_entropy1=2.0, wloss_entropy2=0.8, tanh_thr=0.005, grad_clip=100, l1_ratio=0, lr=9e-3, epochs1=3000, epochs2=600):
    pyg.seed_everything(seed)
    
    model = VGAE(ProdLDAEncoder(data.num_features, hid_dim, num_topics))
    model_ct = VAE(data.num_features, hid_dim, 1, num_categories=n_celltypes)
    model_ff = FFPredict(num_topics, n_celltypes)
    
    if not minibatch:
        [model, model_ct, model_ff], device, loss_values = train(data, model, model_ct, model_ff, temperature=temperature, early_stopping=early_stopping, alpha=alpha, wloss_spatial=wloss_spatial, wloss_KLD=wloss_KLD, wloss_recon=wloss_recon, wloss_entropy=wloss_entropy1, tanh_thr=tanh_thr, grad_clip=grad_clip, l1_ratio=l1_ratio, lr=lr, epochs=epochs1)

        [model, model_ct, model_ff], device, loss_values = train_concat(data, model, model_ct, model_ff, temperature=temperature, early_stopping=early_stopping, alpha=alpha, wloss_spatial=wloss_spatial, wloss_KLD=wloss_KLD, wloss_recon=wloss_recon, wloss_entropy=wloss_entropy2, tanh_thr=tanh_thr, grad_clip=grad_clip, l1_ratio=l1_ratio, lr=lr, epochs=epochs1)

    else:
        [model, model_ct, model_ff], device, loss_values = train_batch(dataloader, model, model_ct, model_ff, temperature=temperature, early_stopping=early_stopping, alpha=alpha, wloss_spatial=wloss_spatial, wloss_KLD=wloss_KLD, wloss_recon=wloss_recon, wloss_entropy=wloss_entropy1, tanh_thr=tanh_thr, grad_clip=grad_clip, l1_ratio=l1_ratio, lr=lr, epochs=epochs2)

        [model, model_ct, model_ff], device, loss_values = train_batch_concat(dataloader, model, model_ct, model_ff, temperature=temperature, early_stopping=early_stopping, alpha=alpha, wloss_spatial=wloss_spatial, wloss_KLD=wloss_KLD, wloss_recon=wloss_recon, wloss_entropy=wloss_entropy2, tanh_thr=tanh_thr, grad_clip=grad_clip, l1_ratio=l1_ratio, lr=lr, epochs=epochs2)

    plt.figure()
    plt.plot(loss_values)
    plt.title("All losses")
    plt.show()
    
    plt.figure()
    plt.plot(loss_values[-200:])
    plt.title("Last 200 losses")
    plt.show()

    return model, model_ct, model_ff
    
def step3_postprocess(data, model, model_ct, model_ff, n_clusters=8):
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
    
    recon_x, logits, _ = model_ct(data.x.to(device), temperature=0.3)
    logits = logits.squeeze(1)
    cell_types_vae = F.softmax(logits, dim=1).argmax(dim=1).cpu().numpy().astype(str)

    niche_composition = F.softmax(model_ff.fc1.weight.detach().cpu(), dim=0)

    pred_domains = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(p.detach().cpu())
                                  
    return p, cell_types_niche, cell_types_vae, niche_composition, pred_domains, recon_celltype, logits
    

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

        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))

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
    Args:
        recon_x (torch.Tensor): Reconstructed input features. [real number]
        x (torch.Tensor): Original input features. [real number]
        logits (torch.Tensor): Logits from the encoder. [real number]
        beta (float): Weighting factor for KL divergence. [positive real number]

    Returns:
        torch.Tensor: Combined loss value. [positive real number]
    """
    
    ##recon_loss0 = nn.MSELoss(reduction='sum')(recon_x, x)
    # Equivalently, recon_loss0 = (x - recon_x).square().sum()
    ##recon_loss = recon_loss0 / (x.var(unbiased=False) + 1e-8) * 3 # normalization for generalizability across data

    sigma2 = torch.exp(log_sigma2) + 1e-8            # scalar
    nll_elem = 0.5 * (torch.log(sigma2) + (x - recon_x)**2 / sigma2)  # (B,G)
    recon_loss = nll_elem.sum(dim=1).sum()
    
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
def train_batch(dataloader, model, model_ct, model_ff, epochs=1500, temperature=1.0, lr=5e-3, alpha=None, betas=(0.9, 0.999), wloss_spatial=0.8, wloss_KLD=0.005, wloss_recon=1, wloss_clf=1, wloss_entropy=2.0, wtanh = None, tanh_thr = 0.005, l1_ratio=0, grad_clip=200, early_stopping=True, spotwise_celltype_probability=None):
    """
    Simultaneous model training for VGAE(model), VAE(model_ct), and FFPredict(model_ff)
    dataloader : mini-batch loader, e.g. NeighborLoader
    epochs     : number of epochs
    lr         : learning rate
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)       # move model to GPU 
    model_ct = model_ct.to(device) # move model_ct to GPU
    model_ff = model_ff.to(device) # move model_ff to GPU 
    if alpha is not None:
        model.alpha=alpha
    if wtanh is None:
        wtanh = dataloader.data.x.shape[0] / 60
                    
    if spotwise_celltype_probability is None:    
        optimizer = torch.optim.Adam( chain(model.parameters(), model_ct.parameters(), model_ff.parameters()), lr=lr, betas=betas)
    else:
        optimizer = torch.optim.Adam( chain(model.parameters(), model_ff.parameters()), lr=lr, betas=betas)
    
    model.train()                  # switch to training mode
    model_ct.train()               # switch to training mode
    model_ff.train()               # switch to training mode
                    
    # loss fun
    loss_connection = nn.CrossEntropyLoss(reduction='sum')
    loss_mse = nn.MSELoss(reduction='sum')
    loss_values = []
    log_sigma2 = 0
    log_sigma2_fixed = 0
    
    for epoch in range(epochs):
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
                    log_sigma2_fixed = model_ct.log_sigma2.data
                log_sigma2 = log_sigma2_fixed
            elif epoch < int(3*epochs/6):
                loss_entropy = 1.0 * wloss_entropy * -(p * torch.log(p + EPS)).sum()
                t_anneal = 1
                log_sigma2 = log_sigma2_fixed
            else:
                loss_entropy = 1.5 * wloss_entropy * -(p * torch.log(p + EPS)).sum()
                t_anneal = max(temperature, 1.0 - (1.0-temperature)/500 * (epoch - int(epochs/2)) )
                log_sigma2 = log_sigma2_fixed
    
            if spotwise_celltype_probability is None:
                # VAE
                recon_x, logits, logits_re = model_ct(batch.x, temperature=t_anneal)
                loss_recon = wloss_recon * vae_loss(recon_x, batch.x, logits, log_sigma2)
    
                tensor_target = logits_re.squeeze(1)
                recon_celltype = model_ff(p)
                eps = 1e-12
                log_recon_celltype = (recon_celltype.clamp_min(eps)).log()
                loss_clf = -(tensor_target * log_recon_celltype).sum() * wloss_clf
                    
            else:
                loss_recon = 0
                tensor_target = torch.from_numpy(spotwise_celltype_probability).to(torch.float32)
                tensor_target = tensor_target.to(device)
                recon_celltype = F.softmax(model_ff(p)) 
                loss_clf = -(tensor_target * (recon_celltype+1e-10).log()).sum() * wloss_clf
            
            
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
    print(f"loss_clf: {loss_clf}")
    return [model, model_ct, model_ff], device, loss_values

# train data
def train_batch_concat(dataloader, model, model_ct, model_ff, epochs=1500, temperature=1.0, lr=5e-3, alpha=None, betas=(0.9, 0.999), wloss_spatial=0.8, wloss_KLD=0.005, wloss_recon=1, wloss_clf=1, wloss_entropy=2.0, wtanh = None, tanh_thr = 0.005, l1_ratio=0, grad_clip=200, early_stopping=True, spotwise_celltype_probability=None):
    """
    Simultaneous model training for VGAE(model), VAE(model_ct), and FFPredict(model_ff)
    dataloader : mini-batch loader, e.g. NeighborLoader
    epochs     : number of epochs
    lr         : learning rate
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)       # move model to GPU 
    model_ct = model_ct.to(device) # move model_ct to GPU
    model_ff = model_ff.to(device) # move model_ff to GPU 
    if alpha is not None:
        model.alpha=alpha
    if wtanh is None:
        wtanh = dataloader.data.x.shape[0] / 100
                    
    if spotwise_celltype_probability is None:    
        optimizer = torch.optim.Adam( chain(model.parameters(), model_ff.parameters()), lr=lr, betas=betas)
    else:
        optimizer = torch.optim.Adam( chain(model.parameters(), model_ff.parameters()), lr=lr, betas=betas)
    
    model.train()                  # switch to training mode
    #model_ct.train()               # switch to training mode
    model_ff.train()               # switch to training mode
                    
    # loss fun
    loss_connection = nn.CrossEntropyLoss(reduction='sum')
    loss_mse = nn.MSELoss(reduction='sum')
    loss_values = []
    log_sigma2 = 0
    log_sigma2_fixed = 0
    
    for epoch in range(epochs):
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
            log_sigma2 = model_ct.log_sigma2.data
            t_anneal = temperature
            loss_entropy = 1.5 * wloss_entropy * -(p * torch.log(p + EPS)).sum()
    
            if spotwise_celltype_probability is None:
                # VAE
                recon_x, logits, logits_re = model_ct(batch.x, temperature=t_anneal)
                #loss_recon = wloss_recon * vae_loss(recon_x, batch.x, logits, log_sigma2)
    
                tensor_target = logits_re.squeeze(1)
                recon_celltype = model_ff(p)
                eps = 1e-12
                log_recon_celltype = (recon_celltype.clamp_min(eps)).log()
                loss_clf = -(tensor_target * log_recon_celltype).sum() * wloss_clf
                    
            else:
                loss_recon = 0
                tensor_target = torch.from_numpy(spotwise_celltype_probability).to(torch.float32)
                tensor_target = tensor_target.to(device)
                recon_celltype = F.softmax(model_ff(p)) 
                loss_clf = -(tensor_target * (recon_celltype+1e-10).log()).sum() * wloss_clf
            
            
            loss = loss_spatial + loss_KLD + loss_entropy + loss_clf + l1_ratio * z.abs().sum(axis=0).sum() -wtanh * torch.tanh(tanh_thr * p.abs().sum(dim=0)).sum()
                                                             # l1_ratio*model_ff(p).abs().sum(axis=0).sum()
            
            epoch_loss += loss.item()

            loss.backward()             # backprop
            
            if grad_clip is not None:              # Gradient Clipping for Gradient Explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                #torch.nn.utils.clip_grad_norm_(model_ct.parameters(), grad_clip)
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
    print(f"loss_clf: {loss_clf}")
    return [model, model_ct, model_ff], device, loss_values

# train data
def train(data, model, model_ct, model_ff, epochs=1500, temperature=1.0, lr=5e-3, alpha=None, betas=(0.9, 0.999), wloss_spatial=0.8, wloss_KLD=0.005, wloss_recon=1, wloss_clf=1, wloss_entropy=2.0, wtanh = None, tanh_thr = 0.005, l1_ratio=0, grad_clip=200, early_stopping=True, spotwise_celltype_probability=None):
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

    model = model.to(device)       # move model to GPU 
    model_ct = model_ct.to(device)    # move model_ct to GPU 
    model_ff = model_ff.to(device) # move model_ff to GPU
    if alpha is not None:
        model.alpha=alpha
    if wtanh is None:
        wtanh = data.x.shape[0] / 60

    if spotwise_celltype_probability is None:
        optimizer = torch.optim.Adam( chain(model.parameters(), model_ct.parameters(), model_ff.parameters()), lr=lr, betas=betas)
    else:
        optimizer = torch.optim.Adam( chain(model.parameters(), model_ff.parameters()), lr=lr, betas=betas)

    model.train()                   # switch to training mode
    model_ct.train()                # switch to training mode
    model_ff.train()                # switch to training mode
    
    # loss fun
    loss_connection = nn.CrossEntropyLoss(reduction='sum')
    loss_mse = nn.MSELoss(reduction='sum')
    loss_values = []
    log_sigma2 = 0
    log_sigma2_fixed = 0
        
    for epoch in range(epochs):
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
                log_sigma2_fixed = model_ct.log_sigma2.data
            log_sigma2 = log_sigma2_fixed
        elif epoch < int(3*epochs/6):
            loss_entropy = 1.0 * wloss_entropy * -(p * torch.log(p + EPS)).sum()
            t_anneal = 1
            log_sigma2 = log_sigma2_fixed
        else:
            loss_entropy = 1.5 * wloss_entropy * -(p * torch.log(p + EPS)).sum()
            t_anneal = max(temperature, 1.0 - (1.0-temperature)/500 * (epoch - int(epochs/2)) )
            log_sigma2 = log_sigma2_fixed
        
        if spotwise_celltype_probability is None:
            # VAE
            recon_x, logits, logits_re = model_ct(data.x, temperature=t_anneal)
            loss_recon = wloss_recon * vae_loss(recon_x, data.x, logits, log_sigma2)

            tensor_target = logits_re.squeeze(1)
            recon_celltype = model_ff(p)
            eps = 1e-12
            log_recon_celltype = (recon_celltype.clamp_min(eps)).log()
            loss_clf = -(tensor_target * log_recon_celltype).sum() * wloss_clf
            
        else:
            loss_recon = 0
            tensor_target = torch.from_numpy(spotwise_celltype_probability).to(torch.float32)
            tensor_target = tensor_target.to(device)
            recon_celltype = F.softmax(model_ff(p)) 
            loss_clf = -(tensor_target * (recon_celltype+1e-10).log()).sum() * wloss_clf
        
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
    print(f"loss_clf: {loss_clf}")
    return [model, model_ct, model_ff], device, loss_values


# train data
def train_concat(data, model, model_ct, model_ff, epochs=1500, temperature=1.0, lr=5e-3, alpha=None, betas=(0.9, 0.999), wloss_spatial=0.8, wloss_KLD=0.005, wloss_recon=1, wloss_clf=1, wloss_entropy=2.0, wtanh = None, tanh_thr = 0.005, l1_ratio=0, grad_clip=200, early_stopping=True, spotwise_celltype_probability=None):
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

    model = model.to(device)       # move model to GPU 
    model_ct = model_ct.to(device)    # move model_ct to GPU 
    model_ff = model_ff.to(device) # move model_ff to GPU
    if alpha is not None:
        model.alpha=alpha
    if wtanh is None:
        wtanh = data.x.shape[0] / 100

    if spotwise_celltype_probability is None:
        optimizer = torch.optim.Adam( chain(model.parameters(), model_ff.parameters()), lr=lr, betas=betas)
    else:
        optimizer = torch.optim.Adam( chain(model.parameters(), model_ff.parameters()), lr=lr, betas=betas)

    model.train()                   # switch to training mode
    #model_ct.train()                # switch to training mode
    model_ff.train()                # switch to training mode
    
    # loss fun
    loss_connection = nn.CrossEntropyLoss(reduction='sum')
    loss_mse = nn.MSELoss(reduction='sum')
    loss_values = []
    log_sigma2 = 0
    log_sigma2_fixed = 0
        
    for epoch in range(epochs):
        optimizer.zero_grad()       # clear previous gradients
        
        # GVAE
        z, p, posterior_mean, posterior_logvar, posterior_var = model.encoder(data.x, data.edge_index)
        
        loss_spatial = wloss_spatial * loss_mse(p[data.edge_index[0]], p[data.edge_index[1]])       
        loss_KLD = wloss_KLD * model.encoder.KLD(posterior_mean, posterior_logvar, posterior_var)

        # favor a low entropy of p
        EPS = 1e-20
        log_sigma2 = model_ct.log_sigma2.data
        t_anneal = temperature
        loss_entropy = 1.5 * wloss_entropy * -(p * torch.log(p + EPS)).sum()
        
        if spotwise_celltype_probability is None:
            # VAE
            recon_x, logits, logits_re = model_ct(data.x, temperature=t_anneal)
            #loss_recon = wloss_recon * vae_loss(recon_x, data.x, logits, log_sigma2)

            tensor_target = logits_re.squeeze(1)
            recon_celltype = model_ff(p)
            eps = 1e-12
            log_recon_celltype = (recon_celltype.clamp_min(eps)).log()
            loss_clf = -(tensor_target * log_recon_celltype).sum() * wloss_clf
            
        else:
            loss_recon = 0
            tensor_target = torch.from_numpy(spotwise_celltype_probability).to(torch.float32)
            tensor_target = tensor_target.to(device)
            recon_celltype = F.softmax(model_ff(p)) 
            loss_clf = -(tensor_target * (recon_celltype+1e-10).log()).sum() * wloss_clf
        
        loss = loss_spatial + loss_KLD + loss_entropy + loss_clf + l1_ratio *  z.abs().sum(axis=0).sum()  -wtanh * torch.tanh(tanh_thr * p.abs().sum(dim=0)).sum() #+ loss_recon 
                                        #p_cell = F.softmax(model_ff.fc1.weight, dim=0)
                                        #l1_ratio * -(p_cell * torch.log(p_cell + EPS)).sum()
                                        #l1_ratio*model_ff(p).abs().sum(axis=0).sum()
        
        loss_values.append( loss.item() )
        loss.backward()             # backprop
        
        if grad_clip is not None:              # Gradient Clipping for Gradient Explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            #torch.nn.utils.clip_grad_norm_(model_ct.parameters(), grad_clip)
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
    print(f"loss_clf: {loss_clf}")
    return [model, model_ct, model_ff], device, loss_values
