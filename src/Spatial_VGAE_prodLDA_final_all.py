import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

import sys
from scipy.stats import norm, multivariate_normal, wishart, Covariance
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering
from tqdm import trange
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
# sc.set_figure_params(dpi=120)
plt.rcParams.update({'font.size': 14})

from itertools import chain

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.nn import VGAE, GCNConv, InnerProductDecoder, Sequential
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain
import torch.nn as nn

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
def train_batch(dataloader, model, model_ct, model_ff, epochs=1500, temperature=1.0, lr=5e-3, alpha=None, betas=(0.9, 0.999), 
                wloss_spatial=0.8, wloss_KLD=0.005, wloss_recon=1, wloss_clf=1, wloss_entropy=2.0, wtanh = None, tanh_thr = 0.005, 
                l1_ratio=0, grad_clip=200, early_stopping=True, spotwise_celltype_probability=None):
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
def train_batch_concat(dataloader, model, model_ct, model_ff, epochs=1500, temperature=1.0, lr=5e-3, alpha=None, betas=(0.9, 0.999), 
                wloss_spatial=0.8, wloss_KLD=0.005, wloss_recon=1, wloss_clf=1, wloss_entropy=2.0, wtanh = None, tanh_thr = 0.005, 
                l1_ratio=0, grad_clip=200, early_stopping=True, spotwise_celltype_probability=None):
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
def train(data, model, model_ct, model_ff, epochs=1500, temperature=1.0, lr=5e-3, alpha=None, betas=(0.9, 0.999),
          wloss_spatial=0.8, wloss_KLD=0.005, wloss_recon=1, wloss_clf=1, wloss_entropy=2.0, wtanh = None, tanh_thr = 0.005,
          l1_ratio=0, grad_clip=200, early_stopping=True, spotwise_celltype_probability=None):
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
def train_concat(data, model, model_ct, model_ff, epochs=1500, temperature=1.0, lr=5e-3, alpha=None, betas=(0.9, 0.999),
          wloss_spatial=0.8, wloss_KLD=0.005, wloss_recon=1, wloss_clf=1, wloss_entropy=2.0, wtanh = None, tanh_thr = 0.005,
          l1_ratio=0, grad_clip=200, early_stopping=True, spotwise_celltype_probability=None):
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
            #ax_s.set_xlim(x.max() - xr*0.52, x.max() - xr*0.80)
            #ax_s.set_ylim(y.max() - yr*0.44, y.max() - yr*0.15)
    
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

    proto_id, link_mat = STEP1_prototype_compression_hierarchical_clustering(z)
    proto_cluster, clust_rep_leaf, rep_leaves_set = STEP2_specify_k_final_clusters(link_mat, K_final = K_final)
    proto_to_color, cluster_to_color = STEP3_color_palette_mapping(link_mat)
    STEP4_visualization(x, y, proto_id, proto_to_color,proto_cluster, clust_rep_leaf,
                        cluster_to_color, link_mat,viz_dendrogram, viz_spatial, save_fig)


def predicted_cell_type_pairs(p, model_ff, num_topics, indices = None):
    import numpy as np
    import itertools

    def get_significant_topics(p, num_topics):
        p_numpy = p.detach().cpu().numpy()
        dynamic_threshold = 1.0 / num_topics 
        significant_topics = set()
        for spot_weights in p_numpy:
            indices = np.where(spot_weights > dynamic_threshold)[0]
            significant_topics.update(indices)
        return significant_topics
    
    def get_predicted_pairs_top3(p, weights, num_topics):
        H, T = weights.shape # H: hidden units, T: topics
        
        # choose which topic columns to use
        target_topics = get_significant_topics(p, num_topics) if indices is None else indices
    
        all_predicted_pairs = set()
        for col in target_topics:
            # pick top-3 rows (largest weights) in this column
            top_rows = np.argsort(weights[:, col])[::-1][:3]
            if top_rows.size >= 2:
                for pair in itertools.combinations(top_rows, 2):
                    all_predicted_pairs.add(tuple(sorted(pair)))
        return all_predicted_pairs

    weights = F.softmax(model_ff.fc1.weight.detach().cpu(), dim=0).numpy()
    return get_predicted_pairs_top3(p, weights, num_topics)
