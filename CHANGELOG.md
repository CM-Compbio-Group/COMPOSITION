# Changelog 

All notable changes to this project will be documented in this file.

---

## [Unreleased]
#### Created - 2025-03-05
- Created the initial code combining [VGAE](https://arxiv.org/abs/1611.07308) and [VAE](https://arxiv.org/abs/1611.01144)

#### Changed - 2025-06-05
- Introduced the penalty term on z

#### Added - 2025-06-25
- Added the initial code to Github repo

#### Changed - 2025-07-03
- Changed the VGAE encoder to ProdLDA to seamlessly perform latent Dirichlet allocation

#### Changed - 2025-07-28
- Adopted the sparsity on p instead the sparsity on z

#### Changed - 2025-08-26
- Discarded the sparsity on p and used the Dirichlet prior with low alpha

#### Fixed - 2025-09-03
- Returned to the sparsity on p

#### Changed - 2025-09-12
- Modified FFPredict to follow an LDA-style from the previous ProdLDA-style to avoid overfitting

#### Fixed - 2025-10-06
- Used logits_re instead of logits for loss_3 to properly apply Gumbel-Softmax reparameterization

#### Fixed - 2025-10-16
- Renamed loss_1, loss_2, and loss_3 to loss_spatial, loss_recon, and loss_clf

#### Changed - 2025-10-24
- Variance fitting before entropy loss comes in, temperature annealing after entropy loss fixed, tanh loss for higher # of non-blanks

#### Remark - 2025-10-30
- Identified the importance of averaging multiple p matrices from a fixed model

#### Added - 2025-11-04
- Added train_concat

#### Added - 2025-11-17
- Added train_batch_concat

---

## [1.0.0] 
#### Changed - 2025-11-20 
- Initial release
