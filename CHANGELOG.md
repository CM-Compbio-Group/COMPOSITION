# Changelog 

All notable changes to this project will be documented in this file.

---

## [Unreleased]
#### Created - 2025-03-05
- Created the initial code combining VGAE and VAE

#### Changed - 2025-06-05
- Introduced the penalty term on $z$

#### Added - 2025-06-25
- Added the initial code to Github repo

#### Changed - 2025-07-03
- Changed the VGAE encoder to ProdLDA to seamlessly perform latent Dirichlet allocation

#### Changed - 2025-07-28
- Adopted the sparsity on $p$ instead of the sparsity on $z$

#### Changed - 2025-08-26
- Discarded the sparsity on $p$ and used the Dirichlet prior with low alpha

#### Fixed - 2025-09-03
- Returned to the sparsity on $p$

#### Changed - 2025-09-12
- Modified FFPredict to follow an LDA-style from the previous ProdLDA-style to avoid overfitting

#### Fixed - 2025-10-06
- Used logits_re instead of logits for loss_3 to properly apply Gumbel-Softmax reparameterization

#### Fixed - 2025-10-16
- Renamed loss_1, loss_2, and loss_3 to loss_spatial, loss_recon, and loss_clf

#### Changed - 2025-10-24
- Variance fitting before entropy loss comes in, temperature annealing after entropy loss fixed, tanh loss for higher # of non-blanks

#### Remark - 2025-10-30
- Identified the importance of averaging multiple $p$ matrices from a fixed model

#### Added - 2025-11-04
- Added train_concat

#### Added - 2025-11-17
- Added train_batch_concat

---

## [1.0.0] 
#### Changed - 2025-11-20 
- Initial release

#### Added - 2025-12-01
- Added wrapper functions for ease of use

#### Added - 2025-12-09
- Added a parameter that allows selecting among multiple optimizers

#### Fixed - 2025-12-14
- Renamed train_concat and train_batch_concat to train_2nd and train_batch_2nd

#### Fixed - 2025-12-15
- Adopted genewise variance instead of single variance

#### Fixed - 2025-12-16
- Fixed the loss_clf term for the case spotwise_celltype_probability is given

#### Fixed - 2026-01-06
- Used model_ct.log_sigma2.detach() instead of model_ct.log_sigma2.data

#### Deleted - 2026-02-18
- Deleted train_batch_2nd
- Deleted train_2nd

#### Added - 2026-02-18
- Added clf_class_weights in train and train_batch
- Added train_vae
- Added coupling_weight
- Set the default of wtanh as 0

#### Fixed - 2026-03-03
- Deleted the re-initializing of model_ct

#### Added - 2026-03-04
- Added get_clf_class_weights

#### Added - 2026-03-09
- Added predicted_cell_type_pairs

#### Fixed - 2026-03-11
- Fixed step1_preprocess
- Deleted max_value

#### Added - 2026-03-13
- Added extra_epochs

#### Added - 2026-03-15
- Integrated separate step1_prev_simulation, step4_evaluation_prev_simulation

#### Added - 2026-03-17
- Added step4_evaluation

#### Fixed - 2026-03-24
- Renamed step4_evaluation to eval_coenrichment

#### Added - 2026-03-24
- Added viz_crosstab_hypothalamus
- Added viz_hierarchical_domain
- Added viz_celltype_spatial
- Added viz_annot_celltype_niche
- Added viz_niches
- Added viz_single_niche
- Added save_models

#### Fixed - 2026-03-25
- Recovered max_value
