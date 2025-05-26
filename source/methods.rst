Methods overview
================

.. raw:: html

   <div id="taskList" style="margin-bottom:1em;">
     <strong>Filter by task:</strong>
     <ul style="list-style:none; padding:0; display:flex; flex-wrap:wrap; gap:0.5em;">
       <li><a href="#" class="task-link" data-task="all">All</a></li>
       <li><a href="#" class="task-link" data-task="['Contrastive Disentanglement', 'Perturbation Responsiveness']">['Contrastive Disentanglement', 'Perturbation Responsiveness']</a></li>
       <li><a href="#" class="task-link" data-task="['Contrastive Disentanglement']">['Contrastive Disentanglement']</a></li>
       <li><a href="#" class="task-link" data-task="['Linear Gene Programmes', 'Contrastive Disentanglement']">['Linear Gene Programmes', 'Contrastive Disentanglement']</a></li>
       <li><a href="#" class="task-link" data-task="['Multi-component Disentanglement', 'Causal Structure', 'Combinatorial Effect Prediction', 'Context Transfer', 'Seen Perturbations']">['Multi-component Disentanglement', 'Causal Structure', 'Combinatorial Effect Prediction', 'Context Transfer', 'Seen Perturbations']</a></li>
       <li><a href="#" class="task-link" data-task="['Multi-component Disentanglement', 'Causal Structure', 'Seen Perturbation Prediction', 'Combinatorial Effect Prediction']">['Multi-component Disentanglement', 'Causal Structure', 'Seen Perturbation Prediction', 'Combinatorial Effect Prediction']</a></li>
       <li><a href="#" class="task-link" data-task="['Multi-component Disentanglement', 'Non-linear Gene Programmess']">['Multi-component Disentanglement', 'Non-linear Gene Programmess']</a></li>
       <li><a href="#" class="task-link" data-task="['Non-linear Gene Programmess', 'Contrastive Disentanglement']">['Non-linear Gene Programmess', 'Contrastive Disentanglement']</a></li>
       <li><a href="#" class="task-link" data-task="['Seen Perturbation Prediction', 'Context Transfer', 'Multi-component Disentanglement']">['Seen Perturbation Prediction', 'Context Transfer', 'Multi-component Disentanglement']</a></li>
       <li><a href="#" class="task-link" data-task="['Seen Perturbation Prediction', 'Multi-component Disentanglement', 'Causal Structure', 'Non-linear Gene Programmess']">['Seen Perturbation Prediction', 'Multi-component Disentanglement', 'Causal Structure', 'Non-linear Gene Programmess']</a></li>
       <li><a href="#" class="task-link" data-task="['Unsupervised Disentanglement', 'Feature relationships']">['Unsupervised Disentanglement', 'Feature relationships']</a></li>
       <li><a href="#" class="task-link" data-task="['Unsupervised Disentanglement', 'Seen Perturbation Prediction', 'Combinatorial Effect Prediction']">['Unsupervised Disentanglement', 'Seen Perturbation Prediction', 'Combinatorial Effect Prediction']</a></li>
       <li><a href="#" class="task-link" data-task="['Unsupervised Disentanglement']">['Unsupervised Disentanglement']</a></li>
     </ul>
   </div>

.. raw:: html
   <div class="method-entry" data-tasks="['Linear Gene Programmes', 'Contrastive Disentanglement']">
...

`cPCA <https://www.nature.com/articles/s41467-018-04608-8#Sec7>`__ (2018)
-----------

**Code availability:**  
`GitHub repo <https://github.com/abidlabs/contrastive>`__

.. toggle:: Description

   A modified version of PCA, where the covariance matrix (COV) is the difference between COV(case/target) and αCOV(control/background). The hyperparameter α is used to balance having a high case variance and a low control variance. To provide some intuition, when α is 0, the model reduces to classic PCA on the case data.  Optimal alphas (equal to k clusters) are identified using spectral clustering over a range of cPCA runs with different alphas, with selection based on the similarity of cPCA outputs.

**Inspired by:**  
- PCA
- Contrastive Mixture Models (Zou et al., 2013)

**Model:**  
- Modified PCA

**Task:**  
- ['Linear Gene Programmes', 'Contrastive Disentanglement']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Linear Gene Programmes', 'Contrastive Disentanglement']">
...

`CSMF <https://academic.oup.com/nar/article/47/13/6606/5512984>`__ (2019)
-----------

**Code availability:**  
`GitHub repo <https://www.zhanglab-amss.org/homepage/software.html>`__

.. toggle:: Description

   A non-negative matrix factorisation that decomposes gene expression matrices into common and specific patterns. For each condition, the observed expression matrix is approximated as the sum of a common component - represented by a common feature matrix (Wc) with condition-specific coefficient matrices (Hc₁, Hc₂) - and a specific component unique to each condition, represented by its own feature matrix (Wsᵢ) and coefficients (Hsᵢ). The model uses an alternating approach to minimize the combined reconstruction error (squared Frobenius norm) across common and shared components.

**Inspired by:**  
- iNMF
- NMF

**Model:**  
- NMF

**Task:**  
- ['Linear Gene Programmes', 'Contrastive Disentanglement']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Linear Gene Programmes', 'Contrastive Disentanglement']">
...

`cLVM <https://arxiv.org/abs/1811.06094>`__ (2019)
-----------

**Code availability:**  
`GitHub repo <https://github.com/kseverso/contrastive-LVM>`__

.. toggle:: Description

   A family of contrastive latent variable models (cLVMs), where case data are modeled as the sum of background and salient latent embeddings, while control data are reconstructed solely from background embeddings: - cLVM with Gaussian likelihoods and priors - Sparse cLVM with horseshoe prior used to regularize the weights - Robust cLVM with a Student's t distribution - cLVM with automatic relevance determination (ARD) to regularize (select) the columns of the weight matrix - contrastive VAE, as a non-linear extension of the framework The shared concept across these models is that each model learns a shared set of latent variables for the background and target data, while salient latent variables are learnt solely for the target data.

**Inspired by:**  
- Contrastive PCA

**Model:**  
- Factor Models
- Contastive VAE

**Task:**  
- ['Linear Gene Programmes', 'Contrastive Disentanglement']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Contrastive Disentanglement']">
...

`cVAE <https://arxiv.org/pdf/1902.04601>`__ (2019)
-----------

**Code availability:**  
`GitHub repo <https://github.com/abidlabs/contrastive_vae>`__

.. toggle:: Description

   VAE with two sets of latent variables (two encoders): salient and background, each learned using amortised inference from both case and control observations, respectively. The latent variables are concatenated and then decoded simultaneously via a shared decoder. During the generative process (decoding), the control observations are reconstructed solely from the background latent space, with salient latent variables being set to 0, while the case observations are generated from both sets of latent variables. Optionally, the two sets of latent variables can be further disentagled by minimizing their total correlation, in practice done by training a discriminator to distinguish real from permuted latent samples.

**Inspired by:**  
- Contrastive PCA

**Model:**  
- Contrastive VAE

**Task:**  
- ['Contrastive Disentanglement']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Linear Gene Programmes', 'Contrastive Disentanglement']">
...

`scPCA <https://academic.oup.com/bioinformatics/article/36/11/3422/5807607>`__ (2020)
------------

**Code availability:**  
`GitHub repo <https://github.com/PhilBoileau/EHDBDscPCA>`__

.. toggle:: Description

   A sparse version of contrastive PCA that enhances interpretability in high-dimensional settings by integrating ℓ1regularization into an iterative procedure to estimate sparse loadings and principal components

**Inspired by:**  
- Contrastive PCA
- Probabilistic PCA

**Model:**  
- Modified PCA

**Task:**  
- ['Linear Gene Programmes', 'Contrastive Disentanglement']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Unsupervised Disentanglement', 'Seen Perturbation Prediction', 'Combinatorial Effect Prediction']">
...

`MichiGAN <https://link.springer.com/article/10.1186/s13059-021-02373-4>`__ (2021)
---------------

**Code availability:**  
`GitHub repo <https://github.com/welch-lab/MichiGAN>`__

.. toggle:: Description

   MichiGAN is a two-step approach that first uses a β-TCVAE - a variant of the variational autoencoder that penalizes total correlation among latent variables to promote disentangled representations. These latent representations (posterior means or samples) are then used to condition a Wasserstein GAN, the generator of which similarly to the VAE reconstructs the data from the latent variables, while attempting to 'fool' a discriminator whether the samples were real or generated. Counterfactual predictions are done via latent space arithmetics as in scGEN.

**Inspired by:**  
- scGEN
- InfoGAN

**Model:**  
- VAE
- conditioned GAN

**Task:**  
- ['Unsupervised Disentanglement', 'Seen Perturbation Prediction', 'Combinatorial Effect Prediction']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Linear Gene Programmes', 'Contrastive Disentanglement']">
...

`PCPCA <https://projecteuclid.org/journals/annals-of-applied-statistics/volume-18/issue-3/Probabilistic-contrastive-dimension-reduction-for-case-control-study-data/10.1214/24-AOAS1877.short>`__ (2024)
------------

**Code availability:**  
`GitHub repo <https://github.com/andrewcharlesjones/pcpca>`__

.. toggle:: Description

   A probabilistic model that builds on cPCA, additionally proposing a case-control-ratio-adjusted α as a more interpretable alternative to the same parameter in cPCA (see comment above).

**Inspired by:**  
- nan

**Model:**  
- modified PCA

**Task:**  
- ['Linear Gene Programmes', 'Contrastive Disentanglement']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Linear Gene Programmes', 'Contrastive Disentanglement']">
...

`CPLVMs <https://projecteuclid.org/journals/annals-of-applied-statistics/volume-16/issue-3/Contrastive-latent-variable-modeling-with-application-to-case-control-sequencing/10.1214/21-AOAS1534.short>`__ (2022)
-------------

**Code availability:**  
`GitHub repo <https://github.com/andrewcharlesjones/cplvm>`__

.. toggle:: Description

   A family of contrastive Poisson latent variable models (CPLVMs), based on a Gamma-Poisson hierarchical generative process: - CPLVM: The variational posterior is approximated using log-normal distributions, preserving non-negativity in the latent factors. - CGLVM: Extends CPLVM by allowing latent factors to take negative values, replacing Gamma priors with Gaussian priors and using a log-link function for the Poisson rates. Variational posteriors are modeled as multivariate Gaussians. The authors also propose a hypothesis testing framework, in which log-(ELBO)-Bayes is calculated between a Null model, omitting the salient latent space, and the full contrastive model. This framework is used to quantify global (across all genes) and joint expression changes in subsets of genes (akin to gene set enrichment analysis).

**Inspired by:**  
- cPCA
- cLVMs
- scVI (hypothesis testing)

**Model:**  
- NB likelihood
- Factor Models

**Task:**  
- ['Linear Gene Programmes', 'Contrastive Disentanglement']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Unsupervised Disentanglement']">
...

`sparseVAE <https://arxiv.org/pdf/2110.10804>`__ (2022)
----------------

**Code availability:**  
`GitHub repo <https://github.com/gemoran/sparse-vae-code>`__

.. toggle:: Description

   Spike and Slab Lasso applied to (non-linear) decoder weights. They show poofs of identifiability when at least 2 "anchor features" are present.

**Inspired by:**  
- oi-VAE
- VSC
- beta-VAE

**Model:**  
- VAE

**Task:**  
- ['Unsupervised Disentanglement']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Non-linear Gene Programmess', 'Contrastive Disentanglement']">
...

`ContrastiveVI <https://www.nature.com/articles/s41592-023-01955-3>`__ (2023)
--------------------

**Code availability:**  
`GitHub repo <https://github.com/scverse/scvi-tools/tree/main/src/scvi/external/contrastivevi>`__

.. toggle:: Description

   The successor to mmVAE introducing improvements: counts are modeled using a negative binomial distribution, and the MMD loss is replaced with the Wasserstein distance. More specifically, the Wasserstein distance is computed exclusively for the salient latent variables of the control data, ensuring it approaches zero. The Wasserstein penalty is optional and is set to 0 (no penalty) by default

**Inspired by:**  
- scVI / totalVI
- cVAE
- Conditional VAE
- mmVAE (theirs)

**Model:**  
- ZINB Likelihood
- Protein-Count (totalVI) Likelihood
- Contrastive VAE
- Multi-modal

**Task:**  
- ['Non-linear Gene Programmess', 'Contrastive Disentanglement']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Contrastive Disentanglement']">
...

`mmVAE <https://arxiv.org/pdf/2202.10560>`__ (2022)
------------

**Code availability:**  
`GitHub repo <https://github.com/suinleelab/MM-cVAE>`__

.. toggle:: Description

   A Contrastive VAE framework, similar to cVAE, which additionally incorporates a maximum mean discrepancy (MMD) loss to enforce salient latent variables in the control data to approach zero, while also using it to align the background latent variables between case and control conditions.

**Inspired by:**  
- nan

**Model:**  
- Contrastive VAE

**Task:**  
- ['Contrastive Disentanglement']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Non-linear Gene Programmess', 'Contrastive Disentanglement']">
...

`MultiGroupVI <https://proceedings.mlr.press/v200/weinberger22a>`__ (2022)
-------------------

**Code availability:**  
`GitHub repo <https://github.com/Genentech/multiGroupVI>`__

.. toggle:: Description

   An extension of ContrastiveVI to multi-case (multi-group) disentaglement via multiple group-specific salient encoders.

**Inspired by:**  
- ContrastiveVI (theirs)

**Model:**  
- ZINB Likelihood
- VAE
- Contrastive

**Task:**  
- ['Non-linear Gene Programmess', 'Contrastive Disentanglement']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Multi-component Disentanglement', 'Non-linear Gene Programmess']">
...

`inVAE <https://www.biorxiv.org/content/10.1101/2024.12.06.627196v1.full>`__ (2024)
------------

**Code availability:**  
`GitHub repo <https://github.com/theislab/inVAE>`__

.. toggle:: Description

   VAE model, which incorporates technical and biological covariates into two sets of latent variables:  - Z_I embeds biologically-relevant variables - Z_B embeds the unwanted variability in the data (i.e. batch effect labels) These are then fed into a shared encoder, along with the count data. The output of this shared encoder is fed to the decoder. Optionally, further disentanglement of the two latent variable sets is achieved by minimizing their total correlation, which is approximated via a minibatch-weighted estimator that quantifies the difference between the joint posterior and the product of individual marginal distributions.

**Inspired by:**  
- scVI
- iVAE
- β-TCVAE

**Model:**  
- VAE
- NB Likelihood

**Task:**  
- ['Multi-component Disentanglement', 'Non-linear Gene Programmess']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Non-linear Gene Programmess', 'Contrastive Disentanglement']">
...

`scDSA <https://openreview.net/pdf?id=fkoqMdTlEg>`__ (2023)
------------

**Code availability:**  
`GitHub repo <->`__

.. toggle:: Description

   A VAE that disentangles disease (case) from healthy (control) cells by learning invariant background and salient space representations. The background and salient representations are summed to reconstruct the count data, with an (optional) interaction term capturing the interplay between cell type and disease. As done in contrastive methods, the salient representation for control cells is set to 0 during the generative (data reconstruction) process. The invariance of the background latent variables is enforced through two GAN-style neural networks: one encouraging the prediction of cell types from the background space, while the other penalises the prediction of disease labels, ensuring that disease-specific information is isolated in the salient space.

**Inspired by:**  
- DANN
- DIVA
- CPA
- scVI 

**Model:**  
- NB likelihood
- Domain-Adversarial NNs
- VAE
- Addative Shift

**Task:**  
- ['Non-linear Gene Programmess', 'Contrastive Disentanglement']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Multi-component Disentanglement', 'Causal Structure', 'Seen Perturbation Prediction', 'Combinatorial Effect Prediction']">
...

`SAMS-VAE <https://proceedings.neurips.cc/paper_files/paper/2023/hash/0001ca33ba34ce0351e4612b744b3936-Abstract-Conference.html>`__ (2023)
---------------

**Code availability:**  
`GitHub repo <https://github.com/insitro/sams-vae>`__

.. toggle:: Description

   A VAE that encodes input data into background latent variables and learns sparse, global (salient) embeddings representing the effects of perturbations. These sparse salient embeddings are modeled using a joint relaxed straight-through (Beta-)Bernoulli distribution (mask) and a normally distributed latent space. This method captures perturbation-specific effects as an additive shift to the background representation, analogous to additive shift methods, but it can also be thought as a multi-condition extention to the contrastive framework (limited to two latent variables (case vs. control), to a more general setup capable of learning global embeddings for each perturbation. As in some contrastive methods, for perturbation samples, the perturbation (global) embeddings are added to the background latent variables to reconstruct the data, while for control samples, the perturbation embeddings are effectively set to zero. 

**Inspired by:**  
- CPA
- SVAE/SVAE+

**Model:**  
- VAE
- NB likelihood
- Conditional Latent Embeddings
- Addative Shift
- Sparse Mechanism Shift

**Task:**  
- ['Multi-component Disentanglement', 'Causal Structure', 'Seen Perturbation Prediction', 'Combinatorial Effect Prediction']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Seen Perturbation Prediction', 'Context Transfer', 'Multi-component Disentanglement']">
...

`svae-ligr <https://openreview.net/pdf?id=8hptqO7sfG>`__ (2024)
----------------

**Code availability:**  
`GitHub repo <https://github.com/theislab/svaeligr>`__

.. toggle:: Description

   A VAE  that combines the sparse mechanism shift from SVAE+ with learning a probabilistic pairing between cells and unobserved auxiliary variables. These auxilary variables correspond to the observed perturbation labels in SVAE+, but here they are learned in a data-driven way (rather than passed as static labels) which in turn enables counterfactual context-transfer scenarios.

**Inspired by:**  
- SVAE+

**Model:**  
- VAE
- NB likelihood
- Sparse Mechanism Shift
- Generative/Experience Replay

**Task:**  
- ['Seen Perturbation Prediction', 'Context Transfer', 'Multi-component Disentanglement']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Seen Perturbation Prediction', 'Multi-component Disentanglement', 'Causal Structure', 'Non-linear Gene Programmess']">
...

`sVAE+ <https://proceedings.mlr.press/v213/lopez23a/lopez23a.pdf>`__ (2023)
------------

**Code availability:**  
`GitHub repo <https://github.com/Genentech/sVAE>`__

.. toggle:: Description

   A VAE that integrates recent advances in sparse mechanism shift modeling for single-cell data, inferring a causal structure where perturbation labels identify the latent variables affected by each perturbation. The method constructs a graph identifying which latent variables are influenced by specific perturbations, promoting disentaglement and enabling biological interpretability, such as uncovering perturbations affecting shared processes. A key modelling contribution is its probabilistic sparsity approach (relaxed straight-through Beta-Bernoulli) on the global sparse embeddings (graph),  improving upon its predecessor, SVAE. As such, the latent space can be seen as being modelled from a Spike-and-Slab prior.

**Inspired by:**  
- SVAE

**Model:**  
- VAE
- NB likelihood
- Sparse Mechanism Shift

**Task:**  
- ['Seen Perturbation Prediction', 'Multi-component Disentanglement', 'Causal Structure', 'Non-linear Gene Programmess']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Multi-component Disentanglement', 'Causal Structure', 'Combinatorial Effect Prediction', 'Context Transfer', 'Seen Perturbations']">
...

`CausCell <https://www.biorxiv.org/content/biorxiv/early/2024/12/17/2024.12.11.628077.full.pdf>`__ (2024)
---------------

**Code availability:**  
`GitHub repo <->`__

.. toggle:: Description

   CausCell integrates causal representation learning with diffusion-based generative modeling to generate counterfactual single-cell data. It disentangles observed and unobserved concepts using concept-specific adversarial discriminators and links the resulting latent representations through a structural causal model encoded as a directed acyclic graph. The use of a diffusion model, instead of a traditional variational autoencoder, improves sample fidelity and better preserves underlying causal relationships during generation.

**Inspired by:**  
- AnnealVAE
- DDPM

**Model:**  
- Diffusion
- Auxilary Classifiers

**Task:**  
- ['Multi-component Disentanglement', 'Causal Structure', 'Combinatorial Effect Prediction', 'Context Transfer', 'Seen Perturbations']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Contrastive Disentanglement', 'Perturbation Responsiveness']">
...

`SC-VAE <https://www.biorxiv.org/content/10.1101/2024.01.05.574421v1.full>`__ (2024)
-------------

**Code availability:**  
`GitHub repo <->`__

.. toggle:: Description

   A VAE that combines the contrastiveVI/cVAE architecture with a classifier that learns the pairing of perturbation labels to cells. As in ContrastiveVI, unperturbed cells are drawn solely from background latent space, while cells classified as perturbed are reconstructed from both the background and salient sapces. Additionally, Hilbert-Schmidt Independence Criterion (HSIC) is used to disentagle the background and salient latent spaces.

**Inspired by:**  
- ContrastiveVI
- scVI
- cVAE

**Model:**  
- VAE
- NB likelihood

**Task:**  
- ['Contrastive Disentanglement', 'Perturbation Responsiveness']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

.. raw:: html
   <div class="method-entry" data-tasks="['Unsupervised Disentanglement', 'Feature relationships']">
...

`Celcomen <https://openreview.net/pdf?id=Tqdsruwyac>`__ (2025)
---------------

**Code availability:**  
`GitHub repo <https://github.com/Teichlab/celcomen>`__

.. toggle:: Description

   Celcomen (CCE) disentangles intra- and inter-cellular gene regulation in spatial transcriptomics data by processing gene expression through two parallel interaction functions. One function uses a graph convolution layer (k-hop GNN) to learn a gene-gene interaction matrix that captures cross-cell signaling, while the other applies a linear layer to model regulation within individual cells. During training, Celcomen combines a normalization term—computed via a mean field approximation that decomposes the overall likelihood into a mean contribution and an interaction contribution - with a similarity measure that directly compares each cell’s predicted gene expression (obtained via message passing) to its actual expression, thereby driving the model to adjust its interaction matrices so that the predictions closely match the observed data. Simcomen (SCE) then leverages these fixed, learned matrices to simulate spatial counterfactuals (e.g., gene knockouts) for in-silico experiments.

**Inspired by:**  
- -

**Model:**  
- K-hop Convolution
- Mean field estimation
- Spatially-informed

**Task:**  
- ['Unsupervised Disentanglement', 'Feature relationships']

.. raw:: html
   </div>
   <hr style="margin:2em 0;"/>

