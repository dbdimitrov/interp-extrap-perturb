All Methods
===========


.. raw:: html

     <p style="margin:0.4em 0 0.8em;">
       Press the&nbsp;
       <span style="color:#8B0000;font-weight:bold;">+</span>
       in the <em>first</em> column to expand a method’s description.
     </p>

   <style>
   td.details-control { width:20px; text-align:center; cursor:pointer; }
   td.details-control::before { content:'+'; }
   tr.shown td.details-control::before { content:'-'; }
   td.published { text-align:center; font-weight:bold; }
   .table-container { width:100%; overflow-x:auto; }
   a.github-link {
     display:inline-block; font-size:1.2em; vertical-align:middle; color:inherit;
   }
   a.github-link:hover { color:#8B0000; }
   </style>

   <div class="table-container">
     <table id="methods-table" class="display" style="width:100%">
       <thead>
         <tr>
           <th></th><th>Method</th><th>Year</th><th>Task</th>
           <th>Model</th><th>Published</th><th>Code</th>
         </tr>
       </thead>
       <tbody>
         <tr data-description="A modified version of PCA, where the covariance matrix (COV) is the difference between COV(case/perturbed) and αCOV(control/background). The hyperparameter α is used to balance having a high case variance and a low control variance. To provide some intuition, when α is 0, the model reduces to classic PCA on the case data. Optimal alphas (equal to k clusters) are identified using spectral clustering over a range of cPCA runs with different alphas, with selection based on the similarity of cPCA outputs.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-018-04608-8#Sec7">cPCA</a></td>
           <td>2018</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>Modified PCA</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/abidlabs/contrastive" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A non-negative matrix factorisation that decomposes gene expression matrices into common and condition-specific patterns. For each condition, the observed expression matrix is approximated as the sum of a common component - represented by a common feature matrix with condition-specific coefficient matrices - and a specific component unique to each condition, represented by its own feature matrix  and coefficients. The model uses an alternating approach to minimize the combined reconstruction error (squared Frobenius norm) across common and shared components.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/nar/article/47/13/6606/5512984">CSMF</a></td>
           <td>2019</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>NMF</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://www.zhanglab-amss.org/homepage/software.html" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A family of contrastive latent variable models (cLVMs), where case data are modeled as the sum of background and salient latent embeddings, while control data are reconstructed solely from background embeddings: - cLVM with Gaussian likelihoods and priors - Sparse cLVM with horseshoe prior used to regularize the weights - Robust cLVM with a Student&#39;s t distribution - cLVM with automatic relevance determination to regularise the columns of the weight matrix - contrastive VAE, as a non-linear extension of the framework The shared concept across these models is that each model learns a shared set of latent variables for the background and target data, while salient latent variables are learnt solely for the target data.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/abs/1811.06094">cLVM</a></td>
           <td>2019</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>Factor Models</li><li>Contastive VAE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/kseverso/contrastive-LVM" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="VAE with two sets of latent variables (two encoders): salient and background, each learned using amortised inference from both case and control observations, respectively. The latent variables are concatenated and then decoded simultaneously via a shared decoder. During the generative process (decoding), the control observations are reconstructed solely from the background latent space, with salient latent variables being set to 0, while the case observations are generated from both sets of latent variables. Optionally, the two sets of latent variables can be further disentagled by minimizing their total correlation, in practice done by training a discriminator to distinguish real from permuted latent samples.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/1902.04601">cVAE</a></td>
           <td>2019</td>

           <td><ul><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>Contrastive VAE</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/abidlabs/contrastive_vae" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A sparse version of contrastive PCA that enhances interpretability in high-dimensional settings by integrating l1 regularization into an iterative procedure to estimate sparse loadings and principal components">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/36/11/3422/5807607">scPCA</a></td>
           <td>2020</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>Modified PCA</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/PhilBoileau/EHDBDscPCA" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="MichiGAN is a two-step approach that first uses a β-TCVAE - a variant of the variational autoencoder that penalizes total correlation among latent variables to promote disentangled representations. These latent representations (posterior means or samples) are then used to condition a Wasserstein GAN, the generator of which similarly to the VAE reconstructs the data from the latent variables, while attempting to &#39;fool&#39; a discriminator whether the samples were real or generated. Counterfactual predictions are done via latent space arithmetics as in scGEN.">
           <td class="details-control"></td>
           <td><a href="https://link.springer.com/article/10.1186/s13059-021-02373-4">MichiGAN</a></td>
           <td>2021</td>

           <td><ul><li>Unsupervised Disentanglement</li><li>Seen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>VAE</li><li>conditioned GAN</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/welch-lab/MichiGAN" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A probabilistic model that builds on cPCA, additionally proposing a case-control-ratio-adjusted α as a more interpretable alternative to the same parameter in cPCA (see comment above).">
           <td class="details-control"></td>
           <td><a href="https://projecteuclid.org/journals/annals-of-applied-statistics/volume-18/issue-3/Probabilistic-contrastive-dimension-reduction-for-case-control-study-data/10.1214/24-AOAS1877.short">PCPCA</a></td>
           <td>2024</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>modified PCA</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/andrewcharlesjones/pcpca" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A family of contrastive Poisson latent variable models (CPLVMs), based on a Gamma-Poisson hierarchical generative process: - CPLVM: The variational posterior is approximated using log-normal distributions, preserving non-negativity in the latent factors. - CGLVM: Extends CPLVM by allowing latent factors to take negative values, replacing Gamma priors with Gaussian priors and using a log-link function for the Poisson rates. Variational posteriors are modeled as multivariate Gaussians. The authors also propose a hypothesis testing framework, in which log-(ELBO)-Bayes is calculated between a Null model, omitting the salient latent space, and the full contrastive model. This framework is used to quantify global (across all genes) and joint expression changes in subsets of genes (akin to gene set enrichment analysis).">
           <td class="details-control"></td>
           <td><a href="https://projecteuclid.org/journals/annals-of-applied-statistics/volume-16/issue-3/Contrastive-latent-variable-modeling-with-application-to-case-control-sequencing/10.1214/21-AOAS1534.short">CPLVMs</a></td>
           <td>2022</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>NB likelihood</li><li>Factor Models</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/andrewcharlesjones/cplvm" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Spike and Slab Lasso applied to (non-linear) decoder weights. They show poofs of identifiability when at least 2 &#34;anchor features&#34; are present.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/2110.10804">sparseVAE</a></td>
           <td>2022</td>

           <td><ul><li>Unsupervised Disentanglement</li></ul></td>

           <td><ul><li>VAE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/gemoran/sparse-vae-code" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="The successor to mmVAE introducing improvements: counts are modeled using a negative binomial distribution, and the MMD loss is replaced with the Wasserstein distance. More specifically, the Wasserstein distance is computed exclusively for the salient latent variables of the control data, ensuring it approaches zero. The Wasserstein penalty is optional and is set to 0 (no penalty) by default">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-023-01955-3">ContrastiveVI</a></td>
           <td>2023</td>

           <td><ul><li>Nonlinear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>ZINB Likelihood</li><li>Protein-Count (totalVI) Likelihood</li><li>Contrastive VAE</li><li>Multi-modal</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/scverse/scvi-tools/tree/main/src/scvi/external/contrastivevi" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A Contrastive VAE framework, similar to cVAE, which additionally incorporates a maximum mean discrepancy (MMD) loss to enforce salient latent variables in the control data to approach zero, while also using it to align the background latent variables between case and control conditions.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/2202.10560">mmVAE</a></td>
           <td>2022</td>

           <td><ul><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>Contrastive VAE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/suinleelab/MM-cVAE" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="An extension of ContrastiveVI to multi-case (multi-group) disentaglement via multiple group-specific salient encoders.">
           <td class="details-control"></td>
           <td><a href="https://proceedings.mlr.press/v200/weinberger22a">MultiGroupVI</a></td>
           <td>2022</td>

           <td><ul><li>Nonlinear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>ZINB Likelihood</li><li>VAE</li><li>Contrastive</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/Genentech/multiGroupVI" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="VAE model, which incorporates technical and biological covariates into two sets of latent variables:  - Z_I embeds biologically-relevant variables - Z_B embeds the unwanted variability in the data (i.e. batch effect labels) These are then fed into a shared encoder, along with the count data. The output of this shared encoder is fed to the decoder. Optionally, further disentanglement of the two latent variable sets is achieved by minimizing their total correlation, which is approximated via a minibatch-weighted estimator that quantifies the difference between the joint posterior and the product of individual marginal distributions.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.12.06.627196v1.full">inVAE</a></td>
           <td>2024</td>

           <td><ul><li>Multi-component Disentanglement</li><li>Nonlinear Gene Programmes</li></ul></td>

           <td><ul><li>VAE</li><li>NB Likelihood</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/theislab/inVAE" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A VAE that disentangles disease (case) from healthy (control) cells by learning invariant background and salient space representations. The background and salient representations are summed to reconstruct the count data, with an (optional) interaction term capturing the interplay between cell type and disease. As done in contrastive methods, the salient representation for control cells is set to 0 during the generative (data reconstruction) process. The invariance of the background latent variables is enforced through two GAN-style neural networks: one encouraging the prediction of cell types from the background space, while the other penalises the prediction of disease labels, ensuring that disease-specific information is isolated in the salient space.">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/pdf?id=fkoqMdTlEg">scDSA</a></td>
           <td>2023</td>

           <td><ul><li>Nonlinear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>NB likelihood</li><li>Domain-Adversarial NNs</li><li>VAE</li><li>Addative Shift</li></ul></td>


           <td class="published">✓</td>
            <td>✗</td>
         </tr>
         <tr data-description="A VAE that encodes input data into background latent variables and learns sparse, global (salient) embeddings representing the effects of perturbations. These sparse salient embeddings are modeled using a joint relaxed straight-through (Beta-)Bernoulli distribution (mask) and a normally distributed latent space. This method captures perturbation-specific effects as an additive shift to the background representation, analogous to additive shift methods, but it can also be thought as a multi-condition extention to the contrastive framework (limited to two latent variables (case vs. control), to a more general setup capable of learning global embeddings for each perturbation. As in some contrastive methods, for perturbation samples, the perturbation (global) embeddings are added to the background latent variables to reconstruct the data, while for control samples, the perturbation embeddings are effectively set to zero. ">
           <td class="details-control"></td>
           <td><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/0001ca33ba34ce0351e4612b744b3936-Abstract-Conference.html">SAMS-VAE</a></td>
           <td>2023</td>

           <td><ul><li>Multi-component Disentanglement</li><li>Causal Structure</li><li>Seen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>VAE</li><li>NB likelihood</li><li>Conditional Latent Embeddings</li><li>Addative Shift</li><li>Sparse Mechanism Shift</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/insitro/sams-vae" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A VAE  that combines the sparse mechanism shift from SVAE+ with learning a probabilistic pairing between cells and unobserved auxiliary variables. These auxilary variables correspond to the observed perturbation labels in SVAE+, but here they are learned in a data-driven way (rather than passed as static labels) which in turn enables counterfactual context-transfer scenarios.">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/pdf?id=8hptqO7sfG">svae-ligr</a></td>
           <td>2024</td>

           <td><ul><li>Seen Perturbation Prediction</li><li>Context Transfer</li><li>Multi-component Disentanglement</li></ul></td>

           <td><ul><li>VAE</li><li>NB likelihood</li><li>Sparse Mechanism Shift</li><li>Generative/Experience Replay</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/theislab/svaeligr" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A VAE that integrates recent advances in sparse mechanism shift modeling for single-cell data, inferring a causal structure where perturbation labels identify the latent variables affected by each perturbation. The method constructs a graph identifying which latent variables are influenced by specific perturbations, promoting disentaglement and enabling biological interpretability, such as uncovering perturbations affecting shared processes. A key modelling contribution is its probabilistic sparsity approach (relaxed straight-through Beta-Bernoulli) on the global sparse embeddings (graph),  improving upon its predecessor, SVAE. As such, the latent space can be seen as being modelled from a Spike-and-Slab prior.">
           <td class="details-control"></td>
           <td><a href="https://proceedings.mlr.press/v213/lopez23a/lopez23a.pdf">sVAE+</a></td>
           <td>2023</td>

           <td><ul><li>Seen Perturbation Prediction</li><li>Multi-component Disentanglement</li><li>Causal Structure</li><li>Nonlinear Gene Programmes</li></ul></td>

           <td><ul><li>VAE</li><li>NB likelihood</li><li>Sparse Mechanism Shift</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/Genentech/sVAE" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CausCell integrates causal representation learning with diffusion-based generative modeling to generate counterfactual single-cell data. It disentangles observed and unobserved concepts using concept-specific adversarial discriminators and links the resulting latent representations through a structural causal model encoded as a directed acyclic graph.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/biorxiv/early/2024/12/17/2024.12.11.628077.full.pdf">CausCell</a></td>
           <td>2024</td>

           <td><ul><li>Multi-component Disentanglement</li><li>Causal Structure</li><li>Combinatorial Effect Prediction</li><li>Context Transfer</li><li>Seen Perturbation Prediction</li></ul></td>

           <td><ul><li>Diffusion</li><li>Auxilary Classifiers</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/bm2-lab/CausCell" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A VAE that combines the contrastiveVI/cVAE architecture with a classifier that learns the pairing of perturbation labels to cells. As in ContrastiveVI, unperturbed cells are drawn solely from background latent space, while cells classified as perturbed are reconstructed from both the background and salient sapces. Additionally, Hilbert-Schmidt Independence Criterion (HSIC) is used to disentagle the background and salient latent spaces.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.01.05.574421v1.full">SC-VAE</a></td>
           <td>2024</td>

           <td><ul><li>Contrastive Disentanglement</li><li>Perturbation Responsiveness</li></ul></td>

           <td><ul><li>VAE</li><li>NB likelihood</li></ul></td>


           <td class="published">✓</td>
            <td>✗</td>
         </tr>
         <tr data-description="Celcomen disentangles intra- and inter-cellular gene regulation in spatial transcriptomics data by processing gene expression through two parallel interaction functions. One function uses a single graph convolution layer (1-hop GNN) to learn a gene-gene interaction matrix that captures cross-cell signaling, while the other applies a linear layer to model regulation within individual cells. Training maximises an approximate likelihood that aligns the model-predicted weight matrices to the average gene expression across all cells. Simcomen then leverages these fixed, learned matrices to simulate spatial counterfactuals (e.g., gene knockouts) for in-silico experiments.">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/pdf?id=Tqdsruwyac">Celcomen</a></td>
           <td>2025</td>

           <td><ul><li>Unsupervised Disentanglement</li><li>Feature Relationships</li></ul></td>

           <td><ul><li>K-hop Convolution</li><li>Spatially-informed</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/Teichlab/celcomen" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="An extension of ContrastiveVI that incorporates an auxiliary classifier to estimate the effects of perturbations, where the classifier operates on the salient variables and is sampled from a relaxed straight-through Bernoulli distribution. The output from the classifier also directly informs the salient latent space, indicating whether a cell expressing a gRNA successfully underwent a corresponding genetic perturbation. Additionally, Wasserstein distance is replaced by KL divergence, encouraging non-perturbed cells to map to the null region of the salient space. For datasets with a larger number of perturbations, the method also re-introduces and minimizes the Maximum Mean Discrepancy between the salient and background latent variables. This discourages the leakage of perturbation-induced information into the background latent variables, ensuring a clearer separation of perturbation effects.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/abs/2411.08072">ContrastiveVI+</a></td>
           <td>2024</td>

           <td><ul><li>Perturbation Responsiveness</li><li>Contrastive Disentanglement</li><li>Nonlinear Gene Programmes</li></ul></td>

           <td><ul><li>ZINB Likelihood</li><li>VAE</li><li>Contrastive</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/insitro/contrastive_vi_plus" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A Group Factor Analysis for multi-omics data that separates latent variables into guided factors, linked to predefined (observed) variables, and unguided factors. This structure ensures that each observed variable (known biological and technical effects) is captured by a corresponding guided factor, disentangling the observed variables from the residual information, which is in turn captured by the unguided factors. Additionally, SOFA works with both continous and categorical guiding variables and it employs a hierarchical horseshoe prior on loading weights, applying adaptive shrinkage at the view, factor, and feature levels.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.10.10.617527v3.full">SOFA</a></td>
           <td>2024</td>

           <td><ul><li>Multi-component Disentanglement</li></ul></td>

           <td><ul><li>Group Factor Analysis</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/tcapraz/SOFA" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="GSFA is a two-layer, guided Factor Analysis model that quantifies the effects of genetic perturbations on latent factors. The model first factorizes the expression matrix Y into a factor matrix Z (normal prior) and gene loadings W (normal-mixture prior). Then, it captures the effect (β) of perturbation on factors using multivariate linear regression. Spike-and-slab prior is used to enforce sparsity on β, which can also analogously be seen as a causal graph. The linearity of GSFA further enables perturbation-associated, differentially-expressed genes to be identified. GSFA uses Gibbs sampling for inference.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-023-02017-4">GSFA</a></td>
           <td>2024</td>

           <td><ul><li>Seen Perturbation Prediction</li><li>Multi-component Disentanglement</li><li>Causal Structure</li><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>Factor Analysis</li><li>Probabilistic</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/xinhe-lab/GSFA" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A VAE that partitions each cell’s latent representation into covariate-specific and covariate-agnostic (invariant) variables. It enforces disentanglement by making the covariate-specific latents more similar for positive pairs of cells (those sharing a covariate) and more dissimilar for negative pairs (those differing in that covariate). Simultaneously, TarDis maximizes or minimizes the distance between these positive/negative pairs and the covariate-agnostic latent space in a way that ensures its independence from the targeted covariates. This is accomplished via multiple distance-based loss terms for each covariate. TarDis supports both categorical and continuous covariates.">
           <td class="details-control"></td>
           <td><a href="https://link.springer.com/chapter/10.1007/978-3-031-90252-9_23">TarDis</a></td>
           <td>2024</td>

           <td><ul><li>Multi-component Disentanglement</li><li>Context Transfer</li></ul></td>

           <td><ul><li>VAE</li><li>NB likelihood</li><li>Multi-modal</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/theislab/tardis" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A VAE that learns disentangled latent representations in an unsupervised manner by employing additive decoders followed by a nonlinear pooling function (by default, log-sum-exp pooling). The decoder splits the latent vector into K variables, each decoded separately, and then aggregates these outputs. This architecture enforces disentanglement under theoretical assumptions, such as the additivity of independent processes, the existence of process-specific gene markers, and reconstruction quality, ensuring that distinct biological processes map to different latent dimensions. Additionally, DRVI performs batch-correction by optionally incorporating covariate information. Finally, DRVI enables the of ranking latent dimensions based on reconstruction and providing a gene interpretability pipeline via latent variable perturbations.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/biorxiv/early/2024/11/08/2024.11.06.622266.full.pdf">DRVI</a></td>
           <td>2024</td>

           <td><ul><li>Unsupervised Disentanglement</li><li>Nonlinear Gene Programmes</li></ul></td>

           <td><ul><li>VAE</li><li>NB likelihood</li><li>Addative Decoders</li><li>Multi-modal</li></ul></td>


           <td class="published">✗</td>
            <td><a href="http://github.com/theislab/drvi" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="The Factorized Causal Representation (FCR) framework disentangles cell representations into three latent blocks: z_x, which captures context-specific (covariate) effects and is invariant to treatment; z_t, which encodes direct treatment effects and is invariant to context; and z_{tx}, which represents interactions between treatment and context. It additionally handles interacting covariates by using a variational autoencoder framework augmented with adversarial regularization. This regularization enforces the invariance of z_x across treatments and the variability of z_t with respect to covariates. Moreover, the conditional independence of the interaction term z_{tx} from both z_x and z_t, is promoted through permutation-based discriminators.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/2410.22472">FCR</a></td>
           <td>2024</td>

           <td><ul><li>Multi-component Disentanglement</li><li>Seen Perturbation Prediction</li></ul></td>

           <td><ul><li>VAE</li><li>Adversarial</li><li>Perturbation-covariate Interactions</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/Genentech/fcr" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A VAE that encodes shared-bio latent factors that capture biological variation (e.g. cell-type differences) and unshared-bio factors that capture condition-specific signals via separate encoders. Shared factors follow a standard normal prior, while unshared factors use a condition-specific Gaussian mixture prior. The invariance of the shared latent variables is enforced via an MMD penalty, while conditon-encoding in the unshared latent variables is promoted via a classification penalty. Group lasso is used to regularise condition-specific encoders, and it&#39;s (penalty) weights are used to select key genes per condition. scDisInFac enables perturbation predictions in multi-batch, multi-condition settings using scGEN-style arithmetics on the unshared space.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-024-45227-w">scDisInFac</a></td>
           <td>2024</td>

           <td><ul><li>Contrastive Disentanglement</li><li>Nonlinear Gene Programmes</li><li>Seen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>VAE</li><li>NB likelihood</li><li>Adversarial</li><li>Gaussian Mixture Model</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/ZhangLabGT/scDisInFact" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A non-negative matrix factorization method that decomposes single-cell gene expression data into common and condition-specific gene modulees. Each sample’s expression matrix is modeled as the sum of a shared component (W₂V) and condition-specific components (W₁Hⱼ), plus residual noise. The approach minimizes a loss function combining reconstruction error (Frobenius norm) with regularization terms that control module scale and inter-condition similarity.">
           <td class="details-control"></td>
           <td><a href="https://link.springer.com/article/10.1186/s13059-022-02649-3#Sec11">scINSIGHT</a></td>
           <td>2022</td>

           <td><ul><li>Contrastive Disentanglement</li><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>NMF</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/Vivianstats/scINSIGHT" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="SIMVI is a spatially-informed VAE that disentangles gene expression variability into two latent factors: an intrinsic variable z, which captures cell type–specific signals, and a spatial variable s, which quantifies spatial effects. The spatial latent variable s is inferred by aggregating the intrinsic representations of neighboring cells via a Graph Attention Network, thereby incorporating local spatial context. To promote independence between z and s, SIMVI employs an asymmetric regularization on z using maximum mean discrepancy or, alternatively, a  mutual information estimator, ensuring that z retains minimal non-cell-intrinsic information. Furthermore, leveraging debiased machine learning principles, the model decomposes gene expression variance by treating s as a continuous treatment and z as confounding covariates, thereby quantifying the specific impact of spatial context on gene expression.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-025-58089-7">SIMVI</a></td>
           <td>2025</td>

           <td><ul><li>Nonlinear Gene Programmes</li><li>Unsupervised Disentanglement</li></ul></td>

           <td><ul><li>ZINB Likelihood</li><li>VAE</li><li>Spatially-informed</li><li>Multi-modal\n</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/KlugerLab/SIMVI" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="trVAE enhances the scGEN model by incorporating condition embeddings and leveraging maximum mean discrepancy regularization to manage distributions across binary conditions. By utilizing a conditional variational autoencoder, trVAE aims to create a compact and consistent representation of cross-condition distributions, enhancing out-of-distribution prediction accuracy. ">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/36/Supplement_2/i610/6055927#409207818">trVAE</a></td>
           <td>2020</td>

           <td><ul><li>Context Transfer</li><li>Seen Perturbation Prediction</li></ul></td>

           <td><ul><li>VAE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="github.com/theislab/trvae" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Dr.VAE uses a Variational Autoencoder architecture to predict drug response from transcriptomic perturbation signatures. It models transcription change as a linear function within a low-dimensional latent space, defined by encoder and decoder neural networks. For paired expression samples from treated and control conditions, Dr.VAE accurately predicts treated expression.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/35/19/3743/5372343">Dr.VAE</a></td>
           <td>2019</td>

           <td><ul><li>Context Transfer</li><li>Seen Perturbation Prediction</li></ul></td>

           <td><ul><li>VAE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/rampasek/DrVAE" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="scGen is VAE that uses latent space vector arithmetics to predict single-cell perturbation responses. The method first encodes high-dimensional gene expression profiles into a latent space, where it computes a difference vector (delta) representing the change between perturbed and unperturbed conditions. At inference, this delta vector is linearly added to the latent representation of unperturbed cells, and the adjusted latent vector is then decoded back into the original gene expression space, thereby simulating the perturbed state. ">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-019-0494-8#Abs1">scGEN</a></td>
           <td>2019</td>

           <td><ul><li>Context Transfer</li></ul></td>

           <td><ul><li>VAE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/theislab/scgen" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CellBox models cellular responses to perturbations, by linking molecular and phenotypic outcomes through a unified nonlinear ODE-based model, aimed at simulating dynamic cellular behavior. The framework uses gradient descent with automatic differentiation to infer ODE network interaction parameters, facilitating exposure to novel perturbations and prediction of cell responses. ">
           <td class="details-control"></td>
           <td><a href="https://www.cell.com/cell-systems/pdf/S2405-4712(20)30464-6.pdf">CellBox</a></td>
           <td>2021</td>

           <td><ul><li>Context Transfer</li><li>Seen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>ODE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/sanderlab/CellBox" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Compositional Perturbation Autoencoder (CPA) models single-cell gene expression under perturbations and covariates by decomposing expression into additive latent embeddings: a basal state, perturbation effects, and covariate effects. To ensure that the basal embedding is disentangled from perturbations and covariates, CPA employs an adversarial training scheme: auxiliary classifiers are trained to predict perturbations and covariates from the basal embedding, while the encoder is updated using a combined loss that discourages the basal representation from encoding such information. Perturbation embeddings are modulated by neural networks applied to continuous covariates (e.g., dose or time), enabling modeling of dose-response and combinatorial effects. ">
           <td class="details-control"></td>
           <td><a href="https://www.embopress.org/doi/full/10.15252/msb.202211517">CPA</a></td>
           <td>2023</td>

           <td><ul><li>Context Transfer</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>VAE</li><li>DANN-based Adversary that attempts to eliminate treatment effects/ cellular context from latent representation</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/theislab/cpa" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="scPreGAN is a deep generative model that predicts the response of single-cell expression to perturbation by integrating an autoencoder and a generative adversarial network. The model extracts common information from unperturbed and perturbed data using an encoder network, and then generates perturbed data using a generator network. scPreGAN outperforms state-of-the-art methods on three real world datasets, capturing the complicated distribution of cell expression and generating prediction data with the same expression abundance as real data.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/38/13/3377/6593485">scPreGan</a></td>
           <td>2022</td>

           <td><ul><li>Seen Perturbation Prediction</li></ul></td>

           <td><ul><li>AE</li><li>GAN</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/ JaneJiayiDong/scPreGAN" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A CPA extension that embeds prior knowlegde about the compound structure of drugs (SMILES representations), allowing it to extend CPA to unseen drug perturbations.">
           <td class="details-control"></td>
           <td><a href="https://proceedings.neurips.cc/paper_files/paper/2022/hash/aa933b5abc1be30baece1d230ec575a7-Abstract-Conference.html">ChemCPA</a></td>
           <td>2022</td>

           <td><ul><li>Unseen Perturbation Prediction</li><li>Context Transfer</li></ul></td>

           <td><ul><li>AE framework inspired by CPA</li><li>Chemical representation embeddings</li></ul></td>


           <td class="published">✓</td>
            <td><a href="github.com/theislab/chemCPA" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="MultiCPA extends CPA to predict combinatorial perturbation responses from CITE-seq data by integrating gene and protein modalities using either concatenation or a product of experts. It employs totalVI-inspired decoders and likelihoods to model modality-specific outputs.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2022.07.08.499049v1.abstract">MultiCPA</a></td>
           <td>2022</td>

           <td><ul><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>AE framework inspired by CPA</li><li>totalVI likelihood</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/theislab/multicpa" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CellCap is a deep generative model that extends CPA by incorporating cross-attention mechanisms between cell state and perturbation response (i.e., its basal latent space and the perturbation design matrix). Further, CellCap uses a variational autoencoder (VAE) framework with a linear decoder to identify sparse and interpretable latent factors.">
           <td class="details-control"></td>
           <td><a href="https://www.cell.com/cell-systems/fulltext/S2405-4712(25)00078-X">CellCap</a></td>
           <td>2024</td>

           <td><ul><li>Multi-component Disentanglement</li><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>VAE</li><li>Attention</li><li>Linear Decoder</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/broadinstitute/CellCap" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="GEARS is uses graph neural networks to learn multidimensional embeddings for genes and their perturbations by respectively leveraging gene co-expression and GO-derived similarity graphs. It first derives refined gene embeddings through a co-expression-based GNN and separately processes perturbation embeddings via a GO graph to incorporate prior biological relationships, with the latter design enabling predictions for unSeen Perturbation Prediction. These embeddings are integrated by adding the aggregated perturbation signal to the gene representations and then decoded using gene-specific layers augmented by a cross-gene context module, ultimately reconstructing the post-perturbation transcriptomic profile. The model is trained end-to-end with a combined autofocus and direction-aware loss, and it can optionally quantify uncertainty through a Gaussian likelihood framework.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41587-023-01905-6#Abs1">GEARS</a></td>
           <td>2023</td>

           <td><ul><li>Combinatorial Effect Prediction</li><li>Unseen Perturbation Prediction</li></ul></td>

           <td><ul><li>GNNs for co-expression and GO relationships</li><li>Label embeddings</li><li>PK Representations</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/snap-stanford/GEARS" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="AttentionPert is a complex generative model that utilizes attention-based mechanisms to reconstruct perturbed cellular profiles from perturbation condition and precomputed Gene2Vec embeddings. It uses two encoders to capture global and local relationships between genes and perturbations (following GEARS). The PertWeight encoder models attention-based interactions between perturbations, while the PertLocal encoder identifies localized perturbation effects using an augmented GO graph.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/40/Supplement_1/i453/7700899">AttentionPert</a></td>
           <td>2024</td>

           <td><ul><li>Combinatorial Effect Prediction</li><li>Unseen Perturbation Prediction</li></ul></td>

           <td><ul><li>Transformer Model</li><li>GNN</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/BaiDing1234/AttentionPert" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="PRnet is a generative framework to predict the transcriptional response of cells to chemical perturbations. To learn the respose, the model randomly assigns control and perturbed cell pairs which are conditioned on the smiles embedding of the chemical perturbation and the dose. PRnet consists of three components: Perturb-adapter, Perturb-encoder, and Perturb-decoder, which work together to generate a distribution of transcriptional responses. Changing the smiles embedding can be used to predict the response of cells to novel chemical perturbations at both bulk and single-cell levels.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-024-53457-1">PRNet</a></td>
           <td>2024</td>

           <td><ul><li>Unseen Perturbation Prediction</li></ul></td>

           <td><ul><li>DNN</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/Perturbation-Response-Prediction/PRnet" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CODEX uses a Deep Neural Network to map cells from control to perturbed states, learning perturbation effects in respective perturbation-dependent latent spaces. These latent spaces can be arbitrarily combined to infer unseen combinatorial effects, allowing the model to predict the outcomes of complex treatment combinations. Additionally, CODEX can leverage prior information from Gene Ontologies to inform the effects of completely unSeen Perturbation Prediction.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/40/Supplement_1/i91/7700898">CODEX</a></td>
           <td>2024</td>

           <td><ul><li>Combinatorial Effect Prediction</li><li>Unseen Perturbation Prediction</li><li>Context Transfer</li></ul></td>

           <td><ul><li>DNN</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/sschrod/CODEX" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="PrePR-CT is a framework designed to predict transcriptional responses to chemical perturbations in unobserved cell types by utilizing cell-type-specific graphs encoded within Graph Attention Networks (GANs). The approach constructs cell graph priors using metacells which are randomly associated with perturbed cells to transform the problem into a regression task.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.07.24.604816v1.full.pdf">PrePR-CT</a></td>
           <td>2024</td>

           <td><ul><li>Context Transfer</li></ul></td>

           <td><ul><li>Graph attention</li><li>Regression</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/reem12345/Cell-Type-Specific-Graphs" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="PDGrapher builds on graph neural network (GNN) to predict therapeutic perturbations that can reverse disease phenotypes, focusing directly on identifying perturbation targets rather than modeling the perturbation effects. By embedding diseased cell states into gene regulatory networks or protein-protein interaction networks, PDGrapher learns latent representations to infer optimal perturbations that drive diseased states toward desired healthy outcomes. The method utilizes dual GNNs - a response prediction module and a perturbagen discovery module - both employing causal graphs as priors and adjusting edges to model interventions. ">
           <td class="details-control"></td>
           <td><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10802439/">PDGrapher</a></td>
           <td>2025</td>

           <td><ul><li>Combinatorial Effect Prediction</li><li>Unseen Perturbation Prediction</li><li>Context Transfer</li></ul></td>

           <td><ul><li>GNN</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/mims-harvard/PDGrapher" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A deep generative model that disentangles (multi-omics) single-cell data by separating sources of variation into known and unknown decomposed latent spaces, which are then concatenated for reconstruction. It requires partial supervision through known cell attributes, such as cell type, age, or perturbation, and employs different encoding strategies for categorical and continuous attributes. A contrastive objective maximizes reconstruction accuracy while minimizing information in unknown attributes, ensuring effective disentanglement. To further constrain the unknown latent space, Biolord uses activation penalty (L2) and Gaussian noise. ">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41587-023-02079-x#Sec6">Biolord</a></td>
           <td>2024</td>

           <td><ul><li>Multi-component Disentanglement</li><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Probabilistic</li><li>ZINB likelihood</li><li>Protein-Count (totalVI) Likelihood</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/nitzanlab/biolord" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="GraphVCI employs two parallel inference branches to estimate latent variables from factual and counterfactual inputs. In the factual branch, observed gene expressions, treatments, and covariates are encoded via an MLP combined with a GCN/GAT module that integrates a gene regulatory network; its corresponding decoder then reconstructs the observed expression profile. The sparse gene regulatory network is generated using a prior-informed drop out mechanism, based on ATAC-Seq data.  A parallel branch processes counterfactual treatments to generate alternative expression profiles. Training minimizes three losses: an individual-specific reconstruction loss computed as the negative log likelihood (e.g., under a normal or negative binomial distribution) of the observed expressions; a covariate-specific loss implemented as an adversarial network using a binary cross-entropy loss on the counterfactual outputs; and a KL divergence loss that regularizes and aligns the latent space between the factual and counterfactual branches.">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/pdf?id=ICYasJBlZNs">graphVCI</a></td>
           <td>2023</td>

           <td><ul><li>Context Transfer</li></ul></td>

           <td><ul><li>Dual-branch variational bayes causal inference framework</li><li>PK Representations</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/yulun-rayn/graphVCI" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="cycle CDR uses a Cycle Consistent Learning strategy with a Complex AE architecture, consisting of two Encoder-Decoder pairs, to reconstruct control and perturbed samples. The two submodels are used in an alternating order to reconstruct the perturbed samples, and a GAN loss is applied to remove irrelevant information in the latent space. Additionally, chemical representations are added to the latent representation of each submodel to enhance the model&#39;s ability to capture chemical information.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/40/Supplement_1/i462/7700878">cycleCDR</a></td>
           <td>2024</td>

           <td><ul><li>Unseen Perturbation Prediction</li></ul></td>

           <td><ul><li>Autoencoder</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/hliulab/cycleCDR" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="GraphVCI predecessor, almost identical architecture, excluding the prior knowledge graphs.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/abs/2209.05935">VCI</a></td>
           <td>2024</td>

           <td><ul><li>Context Transfer</li><li>Seen Perturbation Prediction</li></ul></td>

           <td><ul><li>Dual-branch variational bayes causal inference framework</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/yulun-rayn/variational-causal-inference" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="SALT &amp; PEPER represents a straightforward two-step approach. The initial SALT model assumes additive effects of individual perturbations. Building on this foundation, PEPER leverages a neural network to learn a non-linear correction, effectively accounting for non-additive combinatorial effects. Notably, despite its simplicity, this approach has demonstrated impressive performance on standard extrapolation benchmarks.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/abs/2404.16907">SALT&PEPER</a></td>
           <td>2024</td>

           <td><ul><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>Additive Model</li><li>DNN</li></ul></td>


           <td class="published">✓</td>
            <td>✗</td>
         </tr>
         <tr data-description="Squidiff integrates a diffusion model with a variational autoencoder (VAE) to modulating cellular states and conditions using latent variables. Squidiff can accurately capture and reproduce cellular states, and can be used to generate new single-cell gene expression data over time and in response to stimuli">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.11.16.623974v1">Squidiff</a></td>
           <td>2024</td>

           <td><ul><li>Combinatorial Effect Prediction</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Diffusion Model</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/siyuh/squidiff" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="LEMUR is a PCA based algorithm that defines condition dependent embedings to analyze differences in differentialy expressed genes across conditions. For each condition a separate embeding matrix is learned and reconstructed using a shared matrix. This is used to generate counterfactual estimates for each cell and condition, which is used to infer label-free DE genes and neighborhoods. ">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41588-024-01996-0">LEMUR</a></td>
           <td>2025</td>

           <td><ul><li>Seen Perturbation Prediction</li></ul></td>

           <td><ul><li>Multi-condition PCA</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/const-ae/pylemur" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Expimap uses a nonlinear encoder and a masked linear decoder, where the latent space’s dimensions are set equal to the number of gene programs, and decoder weights are masked according to prior knowledge to ensure that each latent variable reconstructs only genes associated with the geneset (fixed membership), with L1 sparsity regularization allowing soft membership for additional genes, not included in the prior knowledge. Group lasso is additionally used to &#39;deactivate&#39; uniformative Gene Programmes.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41556-022-01072-x">Expimap</a></td>
           <td>2023</td>

           <td><ul><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>VAE</li><li>Linear Decoder</li><li>NB likelihood</li><li>PK Representations</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/theislab/scarches" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="In pmVAE, each predefined pathway is modeled as a VAE that learns a (local) multidimensional latent embedding for the genes in that pathway. Each VAE module minimizes a size-weighted local reconstruction loss based solely on its pathway’s genes, while the (local) latent embeddings from all pathways are concatenated to form a global representation. ">
           <td class="details-control"></td>
           <td><a href="https://icml-compbio.github.io/2021/papers/WCBICML2021_paper_24.pdf">pmVAE</a></td>
           <td>2021</td>

           <td><ul><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>Multiple VAEs</li><li>PK Representations</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/ratschlab/pmvae " class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="ontoVAE uses a multi-layer, linear decoder, structured to represent hierarchical prior knowledge - e.g. layers can represent gene ontology level.  To preserve connections beyond adjacent layers, the decoder concatenates outputs from previous layers with the current layer’s input, with binary masks ensuring that only valid parent–child and gene set relationships are captured. Decoder weights are constrained to be positive to preserve directional pathway activity, with each ontology term represented by three neurons whose average activation reflects its activity.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/39/6/btad387/7199588">ontoVAE</a></td>
           <td>2023</td>

           <td><ul><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>VAE</li><li>Linear Decoder</li><li>PK Representations</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/hdsu-bioquant/onto-vae" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="VEGA replaces conventional fully connected decoder with a sparse linear decoder that uses a binary gene membership mask, assingning latent variables to a pre-defined collection of gene sets.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-021-26017-0">VEGA</a></td>
           <td>2023</td>

           <td><ul><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>VAE</li><li>Linear Decoder</li><li>PK Representations</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/LucasESBS/vega/" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="NicheCompass, the spatial sucessor of ExpiMap, employs multiple decoders: one graph decoder reconstructs the spatial adjacency matrix via an adjacency loss to ensure that spatially-neighboring observations have similar latent representations, while separate (masked) decoders - one for each cell’s own features and one for its aggregated neighborhood features - reconstruct the omics data. By masking the data reconstruction according to prior knowledge, each latent variable is associated with a gene program (subclassified according inter- or  intracellular signalling). Additionally, it learns de novo gene programs that capture novel, spatially coherent expression patterns, not covered by the prior knowledge. By default, it replaces the Group lasso loss of Expimap with a a dropout mechanism to prune uninformative prior knowledge sets.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41588-025-02120-6">NicheCompass</a></td>
           <td>2024</td>

           <td><ul><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>Graph VAE</li><li>Linear Decoder</li><li>NB Likelihood</li><li>Spatially-informed</li><li>PK Representations</li><li>Multi-modal</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/Lotfollahi-lab/nichecompass." class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="EXPORT builds on the VEGA architecture by adding an auxiliary decoder that functions as an ordinal regressor, with an additional cumulative link loss to explicitly model dose-dependent response. ">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/forum?id=f4nMJPKMkQ&referrer=%5Bthe%20profile%20of%20Xiaoning%20Qian%5D(%2Fprofile%3Fid%3D~Xiaoning_Qian1)">EXPORT</a></td>
           <td>2024</td>

           <td><ul><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>VAE</li><li>Linear Decoder</li><li>PK Representations</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/namini94/EXPORT" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="MuVI is a multi-view factor analysis that encodes prior knowledge by imposing structured sparsity on view‐specific factor loadings via a weighted, regularized horseshoe prior. Specifically, it uses a weight parameter that controls the variance of each loading; e.g., by default, it is set to 0.99 for genes known to belong to a gene set and 0.01 for genes which do not (are uknown). Using this hieararchical regulairisation strategy, MuVI directly associates latent factors with corresponding gene sets while still allowing for the de novo identification of additional genes relevant to a given factor.">
           <td class="details-control"></td>
           <td><a href="https://proceedings.mlr.press/v206/qoku23a.html">MuVi</a></td>
           <td>2023</td>

           <td><ul><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>Group Factor Model</li><li>PK Representations</li><li>Multi-modal</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/MLO-lab/MuVI" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="scETM uses a standard VAE encoder with a softmax layer to obtain a cell-by-topic matrix, paired with a linear decoder based on matrix tri-factorization that reconstructs the data from the cell-by-topic matrix, along with topics-by-embedding α, and embedding-by-genes ρ matrices. This structure allows the latent topics to be directly interpreted as groups of co-expressed genes and can optionally integrate prior pathway (prior knowledge) information as a binary mask.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-021-25534-2">scETM</a></td>
           <td>2021</td>

           <td><ul><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>VAE</li><li>Embedding Topic Model</li><li>Linear Decoder</li><li>PK Representations</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/hui2000ji/scETM" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Spectra decomposes a gene expression matrix into cell‐by‐factor and factor‐by‐gene matrices, while integrating prior knowledge gene sets and cell-type labels. It explicitly models both global and cell-type–specific factors by incorporating cell-type labels, thereby disentagling the typically dominating cell-type variation from shared Gene Programmes. Gene sets are represented as a gene–gene knowledge graph, and a penalty term based on a weighted Bernoulli likelihood, guides the factorization toward preserving this graph. Yet, it also permits the data-driven discovery of novel programs by &#39;detaching&#39; factors from the prior. Spectra can also include cell-type-specific prior knowledge gene sets (e.g. T cell antigen receptor activation programmes can be limited to T cells)">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41587-023-01940-3">Spectra</a></td>
           <td>2023</td>

           <td><ul><li>Linear Gene Programmes</li><li>Multi-component Disentanglement</li></ul></td>

           <td><ul><li>Poisson Likelihood</li><li>Factor Analysis</li><li>PK Representations</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/dpeerlab/spectra" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CellDrift fits a negative binomial GLM to scRNA-seq counts using cell type, perturbation, and their interaction as independent (predictor) variables, while also incorporating library size and batch effects. Pairwise contrast coefficients are then derived to quantify the difference between perturbed and control states across time points. These time series of contrast coefficients, representing the temporal trajectory of perturbation effects per gene, are subsequently analyzed using Fuzzy C-means clustering to group similar temporal patterns and Functional PCA to extract the dominant modes of temporal variance.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bib/article/23/5/bbac324/6673850#373524408">CellDrift</a></td>
           <td>2022</td>

           <td><ul><li>Differential Analysis</li></ul></td>

           <td><ul><li>Generalised Linear Model</li><li>NB Likelihood</li><li>Functional PCA</li><li>Fuzzy Clustering</li><li>Time-resolved</li><li>Perturbation-Covariate Interactions</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/KANG-BIOINFO/CellDrift" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CINEMA‐OT disentangles perturbation effects from confounding variation by decomposing the data with independent component analysis (ICA); ICA components correlated with the perturbation labels are identified using Chatterjee’s coefficient and excluded, yielding a background (confounder) latent space that predominantly reflects confounding factors. Optimal transport is then applied to this background space to align perturbed and control cells, thereby generating counterfactual cell pairs, and this OT map is used in downstream analyses. They also propose a reweighting variant (CINEMA‐OT‐W) to address differential cell type abundance by pre-aligning treated cells with k‐nearest neighbor controls and balancing clusters prior to ICA and optimal transport.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-023-02040-5#Sec11">CINEMA-OT</a></td>
           <td>2023</td>

           <td><ul><li>Trace Cell Populations</li><li>Perturbation Responsiveness</li><li>Unsupervised Disentanglement</li></ul></td>

           <td><ul><li>Unbalanced OT</li><li>Entropy‐regularized Sinkhorn</li><li>ICA</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/vandijklab/CINEMA-OT" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CellOT learns mappings between control and perturbed cell state distributions by solving a dual formulation of the optimal transport problem. The approach learns optimal transport maps as the gradient of a convex potential function, which is approximated using input convex neural networks - (briefly) a specific type of neural network with convex-preserving constraints, such as non-negative weights and a predefined set of activation functions (e.g. ReLU). Instead of relying on regularisation-based OT (e.g. Entropy-regularised Sinkhorn), it jointly optimizes dual potentials (a pair of functions) via a max–min loss.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-023-01969-x">CellOT</a></td>
           <td>2023</td>

           <td><ul><li>Trace Cell Populations</li><li>Perturbation Responsiveness</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Dual (min-max) Formulation OT</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/bunnech/cellot" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CondOT builds on CellOT to learn context-aware optimal transport maps by conditioning on an auxiliary variable. Instead of learning a fixed transport map, it learns a context-dependent transport map that adapts based on this auxiliary information. The OT map is modeled as the gradient of a convex potential using partially input convex neural networks, which ensures mathematical properties required for parametrised optimal transport. The auxiliary variables can be of different types: continuous (like dosage or spatial coordinates), categorical (like treatment groups, represented via one-hot encoding), or learned embeddings learned. Additionally, CondOT includes a separate neural module, a combinator network, for combinatorial predictions.">
           <td class="details-control"></td>
           <td><a href="https://proceedings.neurips.cc/paper_files/paper/2022/file/2d880acd7b31e25d45097455c8e8257f-Paper-Conference.pdf">CondOT</a></td>
           <td>2022</td>

           <td><ul><li>Trace Cell Populations</li><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Conditioned Dual (min-max) Formulation OT</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/bunnech/condot/tree/main" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="The paper extends entropic Gromov-Wasserstein Optimal Transport and Co-Optimal Transport to incorporate perturbation labels for aligning data across different modalities from large-scale perturbation screens. The core innovation involves constraining the learned cross-modality coupling matrix to be &#34;label-compatible&#34;, meaning that the transport plan is informed by the perturbation labels and is only allowed to match between cells that have received the same perturbation label, which is achieved by modifying the Sinkhorn algorithm. This label-compatible alignment is used to train a model to estimate cellular responses to perturbations when measurements are available in only one modality.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/2405.00838">GWOT</a></td>
           <td>2025</td>

           <td><ul><li>Trace Cell Populations</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Optimal Transport</li><li>Multi-modal</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://genentech.github.io/Perturb-OT/" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="MMFM (Multi-Marginal Flow Matching) builds on Flow Matching to model cell trajectories across time and conditions. MMFM generalizes the Conditional Flow Matching framework to incorporate multiple time points using a spline-based conditional probability path. Moreover, it leverages ideas from classifier-free guidance to incorporate multiple conditions.">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/pdf?id=hwnObmOTrV">MMFM</a></td>
           <td>2024</td>

           <td><ul><li>Trace Cell Populations</li><li>Context Transfer</li><li>Seen Perturbation Prediction</li></ul></td>

           <td><ul><li>Flow Matching</li><li>Optimal Transport</li></ul></td>


           <td class="published">✓</td>
            <td><a href="github.com/Genentech/MMFM" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Metric Flow Matching (MFM) constructs probability paths between source and target distributions by interpolating geodesics following a data-dependent Riemannian metric, ensuring that interpolations remain close to the data manifold rather than being straight lines in Euclidean space. MFM first learns these geodesics by minimizing a special cost function, and then regresses a vector field along a geodesic-based corrected path using a conditional flow matching objective.">
           <td class="details-control"></td>
           <td><a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/f381114cf5aba4e45552869863deaaa7-Paper-Conference.pdf">MFM</a></td>
           <td>2024</td>

           <td><ul><li>Trace Cell Populations</li><li>Context Transfer</li><li>Seen Perturbation Prediction</li></ul></td>

           <td><ul><li>Flow Matching</li><li>Optimal Transport</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/kksniak/metric-flow-matching.git" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="scDiffusion employs a Latent Diffusion Model for generating single-cell RNA sequencing data, using a three-part framework: a fine-tuned autoencoder for initial data transformation, a skip-connected multilayer perceptron denoising network, and a condition controller for cell-type-specific data generation. ">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/40/9/btae518/7738782">scDiffusion</a></td>
           <td>2024</td>

           <td><ul><li>Context Transfer</li></ul></td>

           <td><ul><li>Diffusion</li><li>VAE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/EperLuo/scDiffusion" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CFGen is a flow-based model for producing multi-modal scRNA-seq data. CFGen builds on CellFlow and explicitly models the discrete, over-dispersed nature of single-cell counts when generating synthetic data.">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/forum?id=3MnMGLctKb">CFGen</a></td>
           <td>2024</td>

           <td><ul><li>Trace Cell Populations</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Optimal Transport</li><li>Multi-modal</li><li>Conditional Flow Matching</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/theislab/CFGen" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CellFlow learns a vector field to predict time-dependent expression profiles under diverse conditions. The model encodes various covariates (perturbation, dosage, batch, etc.), aggregates the embeddings via attention and deep sets, and uses a conditional flow matching framework to learn the underlying flow of the effect.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2025.04.11.648220v1.full.pdf">cellFlow</a></td>
           <td>2024</td>

           <td><ul><li>Trace Cell Populations</li><li>Context Transfer</li><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>Conditional Flow Matching</li><li>Optimal Transport</li></ul></td>


           <td class="published">✗</td>
            <td>✗</td>
         </tr>
         <tr data-description="Waddington-OT models developmental processes as time‐varying probability distributions in gene expression space and infers temporal couplings by solving an entropy‐regularized, unbalanced optimal transport problem. Growth rate, estimated leveraging expression levels of genes associated with proliferation and apoptosis, is taken into consideration via unbalanced OT. Additionally, uses spectral clustering to obtain Gene Programmes, and subsequently associate those to predictive TFs.">
           <td class="details-control"></td>
           <td><a href="https://www.sciencedirect.com/science/article/pii/S009286741930039X?via%3Dihub">Waddington-OT</a></td>
           <td>2019</td>

           <td><ul><li>Trace Cell Populations</li><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>Unbalanced OT</li><li>Entropy‐regularized Sinkhorn</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/broadinstitute/wot" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="The paper introduces two key Flow Matching variants: (i) Optimal Transport CFM (OT-CFM), which uses optimal transport couplings (approximated with minibatch OT), to produce more robust flow inference and (ii) Schrödinger Bridge CFM solving the Schrödinger Bridge problem by using entropy-regularized OT couplings. ">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/forum?id=CD9Snc73AW">OT-CFM</a></td>
           <td>2024</td>

           <td><ul><li>Trace Cell Populations</li></ul></td>

           <td><ul><li>Flow Matching</li><li>Schrödinger Bridge</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/atong01/conditional-flow-matching" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="SBALIGN solves a Diffusion Schrödinger Bridge Problem with aligned data, i.e. where each sample from the source distribution is paired with a corresponding sample from the target distribution. It combines classical SB theory with Doob&#39;s h-transform to derive a novel loss -- parameterizing the drift and h-function with neural networks, enabling more stable training than iterative procedures.">
           <td class="details-control"></td>
           <td><a href="https://proceedings.mlr.press/v216/somnath23a/somnath23a.pdf">SBALIGN</a></td>
           <td>2023</td>

           <td><ul><li>Trace Cell Populations</li></ul></td>

           <td><ul><li>Schrödinger Bridge</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/vsomnath/aligned_diffusion_bridges " class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="MIOFlow learns (stochastic) continuous dynamics from snapshot data by combining neural ODEs, manifold learning, and optimal transport. The method first trains a Geodesic Autoencoder to embed data into a latent space where geodesic-based distances are preserved, then models trajectories via a neural ODE and an ODE solver, and optimizes the Wasserstein loss between actual and predicted distributions.">
           <td class="details-control"></td>
           <td><a href="https://proceedings.neurips.cc/paper_files/paper/2022/file/bfc03f077688d8885c0a9389d77616d0-Paper-Conference.pdf">MioFlow</a></td>
           <td>2022</td>

           <td><ul><li>Trace Cell Populations</li></ul></td>

           <td><ul><li>Neural ODE</li><li>Geodasic Autoencoder</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/KrishnaswamyLab/MIOFlow" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Moscot is a broad and scalable framework that recasts various single-cell mapping tasks as optimal transport problems, supporting formulations that compare distributions in shared (Wasserstein-type OT), distinct (Gromov-Wasserstein OT), and partially-overlapping feature spaces (fused-Gromov–Wasserstein OT). Beyond Entropy-regularized sinkhorn (Cuturi et al., 2013), moscot provides a user-friendly API to more recent OT strategies, such as low-rank and sparse Monge maps.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41586-024-08453-2">moscot</a></td>
           <td>2025</td>

           <td><ul><li>Trace Cell Populations</li></ul></td>

           <td><ul><li>Unbalanced OT</li><li>Entropy‐regularized Sinkhorn</li><li>Low-rank OT</li><li>Sparse Map OT</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/theislab/moscot" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Geneformer is a context-aware transformer encoder comprising six layers of full dense self-attention over an input sequence of up to 2,048 genes, producing embeddings for genes and cells. Genes in each single-cell transcriptome are encoded as  rank value vectors - each gene’s expression is ranked within each cell. Pretraining uses a self-supervised masked learning objective (masking 15% of gene tokens and minimizing a prediction loss to recover their identities).">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41586-023-06139-9">Geneformer</a></td>
           <td>2023</td>

           <td><ul><li>GRN Inference</li></ul></td>

           <td><ul><li>Foundational Gene expression embeddings (from ~30M human cells)</li><li>Self-supervised masked regression</li><li>Standard transformer attention</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/jkobject/geneformer" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="scGPT processes each cell as a sequence of gene tokens, expression-value tokens and condition tokens (e.g., batch, perturbation or modality), embedding each and summing before feeding them into stacked transformer blocks whose specialised, masked multi-head attention layers enable autoregressive prediction of masked gene expressions from non-sequential data. scGPT is pretrained using a masked gene expression-prediction objective that jointly optimizes cell and gene embeddings, and can be fine-tuned on smaller datasets with task-specific supervised losses. For gene regulatory network inference, scGPT derives k-nearest neighbor similarity graphs from learned gene embeddings and analyses attention maps to extract context-specific Gene Programmes and gene-gene interactions.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-024-02201-0">scGPT</a></td>
           <td>2024</td>

           <td><ul><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li><li>GRN Inference</li></ul></td>

           <td><ul><li>Foundational Gene expression embeddings (from >33M human cells)</li><li>Self-supervised masked expression prediction</li><li>Customised non-sequential (flash) attention</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/bowang-lab/scGPT" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="scELMo first converts gene and cell metadata into textual descriptions and uses GPT-3.5 to generate fixed-length embeddings, which are integrated with normalised expression values by arithmetic or weighted averaging in a zero-shot framework to yield cell embeddings. For some tasks, these embeddings and are fine-tuned via a compact neural adaptor trained with combined classification and contrastive losses. These embeddings are also fed into CPA’s conditional variational autoencoder and GEARS’s graph neural network for perturbation response prediction">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2023.12.07.569910v2">scELMo</a></td>
           <td>2024</td>

           <td><ul><li>Seen Perturbation Prediction</li></ul></td>

           <td><ul><li>Converts gene/cell metadata into text embeddings</li><li>Integrates text and expression embeddings</li><li>Fine-tunes embeddings via a lightweight neural adaptor</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/HelloWorldLTY/scELMo" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="C2S-Scale is a family of large language models (LLMs) for single-cell RNA-seq analysis that extends the Cell2Sentence (C2S) framework by converting cell gene-expression profiles into ordered “cell sentences” for natural-language processing.Each C2S-Scale model is initialized from a publicly released Gemma-2 or Pythia checkpoint, i.e. leverages pre-existing language representations, and is then further pre-trained on a multimodal corpus of over a billion tokens. Each cell sentence is paired with the abstract (and, where available, additional free-text annotations) from the same study, allowing the model to learn matched transcriptomic and experimental context. ">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2025.04.14.648850v1">C2S-Scale</a></td>
           <td>2025</td>

           <td><ul><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>Family of LLMs with to 27B parameters</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/vandijklab/cell2sentence" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="LPM is a decoder-only deep neural network designed for large-scale integration and prediction across heterogeneous perturbation datasets. LPM encodes perturbation (P), readout (R), and context (C) as discrete variables, each with its own embedding space implemented via learnable look-up tables. These embeddings are concatenated to and used for inference">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/2503.23535">LPM</a></td>
           <td>2025</td>

           <td><ul><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li><li>GRN Inference</li></ul></td>

           <td><ul><li>DNN Decoder</li></ul></td>


           <td class="published">✗</td>
            <td>✗</td>
         </tr>
         <tr data-description="scGenePT combines CRISPR single‐cell RNA‐seq perturbation data with language‐based gene embeddings. It builds on a pretrained scGPT by adding gene‐level text embeddings from NCBI Gene/UniProt summaries or GO annotations, to the token, count, and perturbation embeddings of the model during fine-tuning on perturbational data.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.10.23.619972v1">scGenePT</a></td>
           <td>2025</td>

           <td><ul><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li><li>GRN Inference</li></ul></td>

           <td><ul><li>scGPT</li><li>ChatGPT prompts</li></ul></td>


           <td class="published">✗</td>
            <td>✗</td>
         </tr>
         <tr data-description="scFoundation uses an asymmetric transformer encoder–decoder: its embedding module converts each continuous gene expression scalar directly into a high-dimensional learnable vector without discretization; the encoder takes as input only nonzero and unmasked embeddings through vanilla transformer blocks to model gene–gene dependencies efficiently. The zero and masked gene embeddings, along with the encoder embeddings, are passed to the decoder, which uses Performer-style attention to reconstruct transcriptome-wide representations, specifically those of masked genes. Specifically, scFoundation is trained using a masked regression objective on both raw and downsampled count vectors, with two total-count tokens concatenated to inputs to account for sequencing depth variance. The decoder-derived gene context embeddings are then used as node features in GEARS for single-cell perturbation response prediction.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-024-02305-7">scFoundation</a></td>
           <td>2024</td>

           <td><ul><li>Nonlinear Gene Programmes</li><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>Foundational Gene expression embeddings (from >50M human cells)</li><li>Self-supervised masked regression with down-sampling</li><li>Sparse transformer encoder</li><li>Performer-style attention decoder</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/biomap-research/scFoundation" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="GeneCompass is a knowledge-informed, cross-species foundation model. During pre-training it integrates four types of prior biological knowledge - gene regulatory networks (ENCODE PECA2‐derived GRNs), promoter sequences (fine‐tuned DNABert embeddings), gene family annotations (gene2vec HGNC/esnembl embeddings), and gene co-expression relationships (Pearson Correlations in their dataset) - into a unified embedding space. It employs a masked-language-modeling strategy by randomly masking 15 % of gene inputs and simultaneously reconstructs both gene identities and expression values; this is optimized via a multi-task loss combining mean squared error for expression recovery and cross-entropy for gene ID prediction, balanced by a weighting hyperparameter β. Combined with GEARS for extrapolation tasks.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41422-024-01034-y">GeneCompass</a></td>
           <td>2024</td>

           <td><ul><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li><li>GRN Inference</li></ul></td>

           <td><ul><li>Foundational Gene expression embeddings (from >50M human cells)</li><li>Self-supervised masked regression with down-sampling</li><li>Sparse transformer encoder</li><li>Performer-style attention decoder</li><li>PK-informed</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/xCompass-AI/GeneCompass" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="scPRINT is implemented as a bidirectional transformer, focusing on scalable zero-shot applications to new datasets. During pre-training, it optimises a single composite loss that sums: (1) a denoising objective, which up-samples down-sampled transcript counts via a zero-inflated negative-binomial decoder; (2) a bottleneck reconstruction objective, where the model must regenerate full expression profiles from its compressed cell embedding; and (3) a hierarchical label-prediction objective that forces disentanglement of latent factors for cell type, disease, platform and other metadata. Each gene token is the sum of: a learned protein embedding for its gene ID; an MLP encoding of its log-normalized count; and a positional encoding of its genomic locus . Pre-training contexts consist of 2,200 randomly sampled expressed genes per cell. At inference, cell-specific gene networks are derived from the model’s multi-head attention maps by either averaging all heads or selecting a subset post hoc based on correlation with external priors (e.g., protein–protein interaction databases, ChIP-seq, perturbation-ground-truth networks).">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-025-58699-1">scPrint</a></td>
           <td>2025</td>

           <td><ul><li>Gene Programmes</li><li>GRN Inference</li><li>Multi-component Disentanglement</li></ul></td>

           <td><ul><li>Foundational Gene expression embeddings (from >50M human cells)</li><li>BERT-like Bidirectional transformers (with flashattention2)</li><li>Self-supervised masked regression</li><li>A classifier decoder</li><li>ZINB likelihood decoder</li><li>PK Representations</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/cantinilab/scPRINT" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="The method embeds each gene using two LLM-derived representations - GPT-3.5 text embeddings of NCBI gene descriptions and ProtT5 protein sequence embeddings; and, after reducing them to the top 50 principal components, uses these as inputs to a multi-output Gaussian Process regression model with an RBF kernel to predict the differential expression response to single-gene knockouts. ">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/forum?id=eb3ndUlkt4">LLM+GP</a></td>
           <td>2024</td>

           <td><ul><li>Unseen Perturbation Prediction</li></ul></td>

           <td><ul><li>Gaussian Process Model</li><li>Language embeddings</li></ul></td>


           <td class="published">✓</td>
            <td>✗</td>
         </tr>
         <tr data-description="TODO">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s42256-023-00719-0">CIV</a></td>
           <td>2023</td>

           <td><ul><li>Causal Structure</li></ul></td>

           <td><ul><li>Active Learning</li><li>Structural Causal Model</li><li>DAG-Bayesian linear regression</li></ul></td>


           <td class="published">✓</td>
            <td>✗</td>
         </tr>
         <tr data-description="TODO">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41587-024-02182-7">LINGER</a></td>
           <td>2024</td>

           <td><ul><li>GRN Inference</li></ul></td>

           <td><ul><li>Multi-modal</li><li>Prior Knowledge Informed</li><li>Shapley values</li><li>DNN</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/Durenlab/LINGER" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="TODO">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-023-01938-4">SCENIC+</a></td>
           <td>2022</td>

           <td><ul><li>GRN Inference</li></ul></td>

           <td><ul><li>Multi-modal</li><li>Prior Knowledge Informed</li><li>Gradient Boosting</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/aertslab/scenicplus" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="TODO">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41586-022-05688-9">CellOracle</a></td>
           <td>2023</td>

           <td><ul><li>GRN Inference</li></ul></td>

           <td><ul><li>Multi-modal</li><li>Prior Knowledge Informed</li><li>Regularised (Linear) Regression</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/morris-lab/CellOracle" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="NOTEARS replaced traditional statistical DAG learning techniques for observational data with a continuous optimization problem, by reformulating the acyclicity constraint. This reduces the computational complexity and facilitated first small scale biological applications. ">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/abs/1803.01422">NOTEARS</a></td>
           <td>2018</td>

           <td><ul><li>Causal Structure</li></ul></td>

           <td><ul><li>Continuous optimisation for acyclicity</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/xunzheng/notears" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="NOTEARS-MLP  further generalized the continuous DAG objective introduced by NOTEARS to nonparametric and semi-parametric models, such as deep neural networks (DNNs), to better facilitate non-linear data.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/abs/1909.13189">NOTEARS-MLP</a></td>
           <td>2020</td>

           <td><ul><li>Causal Structure</li></ul></td>

           <td><ul><li>Continuous optimisation for acyclicity</li><li>DNN</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/xunzheng/notears" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="DAG-GNN introduced a polynomial alternative for the acyclicity constraint of NOTEARS, and encodes the DAG in a Graph Neural Network. Experimental results on synthetic data sets indicate that DAG-GNN learns more accurate graphs for non-linearly generated samples. ">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/abs/1904.10098">DAG-GNN</a></td>
           <td>2019</td>

           <td><ul><li>Causal Structure</li></ul></td>

           <td><ul><li>Continuous optimisation for acyclicity</li><li>GNN</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/fishmoon1234/DAG-GNN" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="DCDI advanced DAG learning by introducing a framework for causal discovery using interventional data. DCDI encoding interventions using a binary adjacency matrix, to replicate the interventional effects directly the DAG and uses neural networks to model the conditional densities. Further, the authors provided theoretical guarantees for DAG learning using interventional data and showed that the inferred graphs can scale to 100 nodes.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/2007.01754">DCDI</a></td>
           <td>2020</td>

           <td><ul><li>Causal Structure</li></ul></td>

           <td><ul><li>Graph interventions</li><li>DNN</li><li>Normalizing-Flows</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/slachapelle/dcdi" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="NODAGS-Flow utilizes contractive residual flows to model perturbational data as generated from the steady state of a dynamical system with explicit noise. Following DCDI, NODAGS-Flow replicates perturbations on the graph. Further, NODAGS-Flow drops the acyclicity constraint to model cyclic causal models and better explain the feedback loops inherent to biological data.">
           <td class="details-control"></td>
           <td><a href="https://proceedings.mlr.press/v206/sethuraman23a/sethuraman23a.pdf">NODAGS-Flow</a></td>
           <td>2023</td>

           <td><ul><li>Causal Structure</li></ul></td>

           <td><ul><li>Graph interventions</li><li>DNN</li><li>Residual Flow\nSteady-State ODE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/Genentech/nodags-flows" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Bicycle addresses the challenge of robustly identifying cyclic causal graphs, particularly in domains like single-cell genomics, by leveraging perturbation data and explicitly replicating the perturbations on the graph. Following Dictys Bicycle assumes the perturbed cell states to be the steady-state solution of the Ornstein-Uhlenbeck process.">
           <td class="details-control"></td>
           <td><a href="https://proceedings.mlr.press/v236/rohbeck24a.html">Bicycle</a></td>
           <td>2023</td>

           <td><ul><li>Causal Structure</li></ul></td>

           <td><ul><li>Graph interventions</li><li>Ornstein-Uhlenbeck process</li><li>Steady-State ODE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/PMBio/bicycle" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A VAE that disentangles control and pertubed cells into a latent space organized by a causal DAG. The encoder produces a Gaussian latent code z, while an intervention encoder transforms intervention one-hot encodings into two embeddings - a soft assignment vector that targets specific latent dimensions and a scalar capturing the intervention’s magnitude. Multiplying and adding these embeddings to z yields a modified latent vector that simulates a soft intervention, whereas zeroing them recovers the control condition. A causal layer then processes the latent vectors using an upper-triangular matrix G, which enforces an acyclic causal structure and propagates intervention effects among the latent factors. The decoder is applied twice - once to the modified latent code to generate virtual counterfactual outputs that reconstruct interventional outcomes, and once to the unmodified code to recover control samples. This dual decoding forces the model to disentangle intervention-specific effects from the intrinsic data distribution. The training objective combines reconstruction error to reconstruct control samples, a discrepancy loss (e.g., MMD) to align virtual counterfactuals with observed interventional data, KL divergence on the latent space, and an L1 penalty on G to enforce sparsity.">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/forum?id=o16sYKHk3S&noteId=2EQ6cmfPHg">discrepancy-VAE</a></td>
           <td>2023</td>

           <td><ul><li>Causal Structure</li><li>Multi-component Disentanglement</li><li>Seen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li><li>Causal Structure</li></ul></td>

           <td><ul><li>VAE</li><li>Disentanglement via Virtual Counterfactuals</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/uhlerlab/discrepancy_vae" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="DCD-FG leverages a Gaussian low-rank structural equation model to model factor directed acyclic graphs (f-DAGs). The f-DAG assumption posits that many nodes share a similar set of parents and children, reflecting the behavior of genes acting collectively in biological programs. This method restricts the search space to low-rank causal interactions to improve causal discovery accuracy and scalability for high-dimensional data. ">
           <td class="details-control"></td>
           <td><a href="https://proceedings.neurips.cc/paper_files/paper/2022/file/7a8fa1382ea068f3f402b72081df16be-Paper-Conference.pdf">DCD-FG</a></td>
           <td>2022</td>

           <td><ul><li>Causal Structure</li></ul></td>

           <td><ul><li>Factor Model</li><li>DAGs</li><li>Latent DAGs</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/Genentech/dcdfg" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Dictys integrates scRNA-seq and scATAC-seq data to infer gene regulatory networks (GRNs) and their changes across multiple conditions. By leveraging multiomic data, Dictys infers context-specific networks and dynamic GRNs using steady-state solutions of the Ornstein-Uhlenbeck process to model transcriptional kinetics and account for feedback loops. It reconstructs GRNs by detecting transcription factor (TF) binding sites and refining these networks with single-cell transcriptomic data, capturing regulatory shifts that reflect TF activity beyond expression levels.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-023-01971-3">Dictys</a></td>
           <td>2023</td>

           <td><ul><li>GRN Inference</li><li>Causal Structure</li></ul></td>

           <td><ul><li>Ornstein–Uhlenbeck process</li><li>Steady-State ODE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/pinellolab/dictys" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="AVICI proposes an amortized causal discovery approach, attempting to directly predict causal structures from observational or interventional data using variational inference rather than performing costly searches over possible structures. Since no ground truth is not available for real data, the mode is pre-trained using simulated data with known causal graphs and subsequently applied to real data.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/abs/2205.12934">AVICI</a></td>
           <td>2022</td>

           <td><ul><li>Causal Structure</li></ul></td>

           <td><ul><li>Amortized pre-training</li><li>Variational Inference</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/larslorch/avici" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="DCI introduced a reformulated version of the PC algorithm. Rather than inferring the Causal Graph directly DCI attempts to identify causal differences between condition-dependent gene regulatory networks (GRNs) by focusing on edges that appear, disappear, or change between conditions. This significantly reduces the computational complexity in comparison to the original PC algorithm.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/37/18/3067/6168117">DCI</a></td>
           <td>2021</td>

           <td><ul><li>Causal Structure</li></ul></td>

           <td><ul><li>PC Algorithm</li></ul></td>


           <td class="published">✓</td>
            <td><a href="http://uhlerlab.github.io/causaldag/dci" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="SEA predicts large causal graphs by leveraging small graphs generated from subsets of variables using standard causal discovery algorithms like FCI or GIES. To tackle the challenges of causal discovery with large variable sets, SEA employs an amortized learning approach and utilizes a complex architecture, including transformer modules and diverse embeddings, to aggregate the subgraphs. SEA is pre-trained on synthetic data with known causal structures and encodes interventions by replicating the effects on the encoded graph.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/abs/2402.01929">SEA</a></td>
           <td>2024</td>

           <td><ul><li>Causal Structure</li></ul></td>

           <td><ul><li>Amortized pre-training</li><li>Transfomer</li><li>Graph Attention</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/rmwu/sea" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="SENA replaces discrepancy‑VAE&#39;s encoder by using a gene-to-pathway mask that applies a soft weighting, via the pathway activity scores α, to the gene expression inputs. In this design, each weight in the encoder is elementwise multiplied by a mask M that assigns full weight to genes known to belong to a pathway and a tunable, lower weight (λ) to genes outside the pathway. This allows the model to primarily capture the signal of annotated genes while still letting unannotated genes contribute, thereby forming interpretable latent factors as linear combinations of pathway activities. ">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/forum?id=NjlafBAahz">SENA</a></td>
           <td>2024</td>

           <td><ul><li>Causal Structure</li><li>Multi-component Disentanglement</li><li>Seen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>Discrepancy-VAE architecture</li><li>VAE</li><li>PK Representations</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/ML4BM-Lab/SENA" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="RiTINI employs graph ordinary differential equations (graph-ODEs) to infer time-varying interaction graphs from multivariate time series data. RiTINI integrates dual attention mechanisms to enhance dynamic modeling and defines interaction graph inference as identifying a directed graph. Further, RiTINI utilizes prior knowledge to initialize the causal graph and by penalizing deviations the prior.Additionally, RiTINI simulates perturbations in silico to further refine the graph structure.">
           <td class="details-control"></td>
           <td><a href="https://proceedings.mlr.press/v231/bhaskar24a.html">RiTINI</a></td>
           <td>2024</td>

           <td><ul><li>GRN Inference</li><li>Causal Structure</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Graph interventions</li><li>Graph-ODE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/KrishnaswamyLab/RiTINI" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="scMAGeCK is a framework with two modules: 1) scMAGeCK-RRA ranks cells by marker expression and uses rank aggregation, with a dropout filtering step, to detect enrichment of specific perturbations; 2) scMAGeCK-LR applies ridge regression on the expression matrix to compute the relevance of perturbations, including in cells with multiple perturbations. Both modules rely on permutation tests and Benjamini-Hochberg correction.">
           <td class="details-control"></td>
           <td><a href="https://link.springer.com/article/10.1186/s13059-020-1928-4#Sec10">scMAGeCK</a></td>
           <td>2020</td>

           <td><ul><li>Differential Analysis</li></ul></td>

           <td><ul><li>Robust Rank Aggregate</li><li>Ridge Regression</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://bitbucket.org/weililab/scmageck/src/master/" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="For each gene-gRNA pair, SCEPTRE fits a negative binomial regression where the response is the gene’s expression across cells and the predictors are binary indicator denoting gRNA presence, plus technical covariates. Concurrently, a logistic regression using the same technical factors estimates π - the probability of detecting the gRNA in a cell. In a conditional resampling step, gRNA assignments are independently redrawn per cell based on π, generating an empirical null distribution of z‐scores; a skew‑t distribution is then fitted to this null to yield calibrated p‑values.">
           <td class="details-control"></td>
           <td><a href="https://link.springer.com/article/10.1186/s13059-021-02545-2">SPECTRE</a></td>
           <td>2021</td>

           <td><ul><li>Differential Analysis</li><li>Perturbation Responsiveness</li></ul></td>

           <td><ul><li>Conditional Resampling</li><li>Generalised Linear Model</li><li>NB Likelihood</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/Katsevich-Lab/sceptre" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Mixscale extends Mixscape by converting the binary perturbed/non‐perturbed assignment into a continuous perturbation score. As it&#39;s predecessor, it first identifies DE genes between gRNA-targeted and non-targeting control cells, then computes perturbation vector and projects each cell’s expression profile onto this vector to yield a quantitative score (computed independently per cell line). A weighted multivariate regression is then applied where each cell’s contribution is scaled according to its perturbation score, so that cells with weaker perturbation (and thus lower scores) have reduced influence on the model. This regression also incorporates covariates such as cell line identity and sequencing depth, and uses a leave-one-feature-out procedure.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41556-025-01622-z">Mixscale</a></td>
           <td>2025</td>

           <td><ul><li>Differential Analysis</li><li>Perturbation Responsiveness</li></ul></td>

           <td><ul><li>Gaussian Mixture Model</li><li>Weighted multivariate regression</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://longmanz.github.io/Mixscale/" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="MELD models cells as samples drawn from a probability density defined in a low-dimensional space (manifold). Each cell is assigned to a one-hot indicator according to its sample origin (e.g. treatment or control), normalized by the total cell count in that sample. A cell (transcriptomic) similarity graph is then built using a decaying kernel, and the normalized indicator vectors are smoothed across the graph, such that each cell’s value is updated by averaging with its neighbors to yield a density estimate for each sample (condition) for that cell. Normalizing these estimates produces a perturbation-associated relative likelihood for each cell. Vertex Frequency Clustering (VFC) then uses these likelihoods, cell indicator vectors, and similarity graphs to cluster cells with similar transcriptomics and perturbation profiles.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41587-020-00803-5#Sec13">MELD(-VCF)</a></td>
           <td>2021</td>

           <td><ul><li>Perturbation Responsiveness</li><li>Perturbation Responsiveness</li></ul></td>

           <td><ul><li>Manifold Learning</li><li>Vertex-frequency analysis</li><li>Graph Diffusion</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/KrishnaswamyLab/MELD" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="GEDI learns a shared latent space and, for each sample, estimates a specific reconstruction function that maps latent states to observed gene expression profiles. This design captures inter-sample variability and enables differential expression analysis along continuous cell-state gradients without relying on predefined clusters. Optionally, it can incorporate prior knowledge.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-024-50963-0?fromPaywallRec=false">GEDI</a></td>
           <td>2024</td>

           <td><ul><li>Differential Analysis</li><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>Probabilistic</li><li>Sample-specific Decoders</li><li>PK Representations (optional)</li><li>RNA Velocity (optional)</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/csglab/GEDI" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Memento is a differential expression framework that uses method-of-moments estimators under a multivariate hypergeometric model, where a gene’s mean is derived from Good-Turing corrected counts scaled by total cell counts. Differential variability is quantified as the variance remaining after accounting for mean-dependent effects (residual variance), while the covariance (pairwise association) between genes is estimated from the off-diagonal elements of the resulting variance-covariance matrix. Efficient permutation is achieved through a bootstrapping strategy that leverages the sparsity of unique transcript counts.">
           <td class="details-control"></td>
           <td><a href="https://www.cell.com/cell/fulltext/S0092-8674(24)01144-9">Memento</a></td>
           <td>2024</td>

           <td><ul><li>Differential Analysis</li><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>Hypergeometric test</li><li>Probabilistic</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/yelabucsf/scrna-parameter-estimation" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="scITD constructs a three-dimensional tensor (donors × genes × cell types) by generating donor-by-gene pseudobulk matrices for each cell type. Tucker decomposition then decomposes this tensor into separate factor matrices for donors, genes, and cell types, along with a core tensor that captures their interactions as latent multicellular expression patterns. The gene factors and core tensor are rearranged into a loading tensor analogous to PCA loadings, while the donor factor matrix represents sample scores. Finally, to improve interpretability, a two-step rotation is carried out - first applying ICA to the gene factors and then varimax to the donor factors.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41587-024-02411-z">scITD</a></td>
           <td>2024</td>

           <td><ul><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>Tensor Decomposition (Tucker)</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/kharchenkolab/scITD" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="scRank infers cell type-specific Gene Programmes from untreated scRNA-seq data by constructing co-expression networks via principal component regression with random subsampling and integrating them using tensor decomposition. It simulates drug perturbation by modifying the drug targets&#39; outgoing edges to generate an in-sillico perturbed network, and then aligns the untreated and perturbed networks via Laplacian eigen-decomposition. In this low-dimensional space, the distances between corresponding gene nodes quantify gene-level changes due to the perturbation. These distances, weighted by network connectivity (e.g., outgoing edge strength normalized by node degree) and extended through two-hop diffusion, are aggregated to yield a composite perturbation score that ranks cell types by their predicted drug responsiveness.">
           <td class="details-control"></td>
           <td><a href="https://www.cell.com/cell-reports-medicine/fulltext/S2666-3791(24)00260-X">scRANK</a></td>
           <td>2024</td>

           <td><ul><li>Linear Gene Programmes</li><li>Perturbation Responsiveness</li><li>GRN Inference</li></ul></td>

           <td><ul><li>PC Regression</li><li>Tensor Decomposition (PARAFAC)</li><li>Network Diffusion</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/ZJUFanLab/scRank" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Taichi identifies perturbation-relevant cell niches in spatial omics data without predefined spatial clustering. It first constructs spatially-informed embeddings using MENDER, which are then used in a logistic regression model to predict slice-level condition labels. Using the trained model each cell (niche) is assigned a probability of belonging to the condition group. These probabilities are clustered using k-means (k=2) to separate condition-relevant and control-like niches. Graph heat diffusion is applied to refine these labels by propagating information across spatially adjacent cells. Finally, a second k-means clustering step is performed on the diffused results to define the final niche segmentation.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.05.30.596656v1.abstract">Taichi</a></td>
           <td>2024</td>

           <td><ul><li>Perturbation Responsiveness</li><li>Perturbation Responsiveness</li><li>Differential Analysis</li></ul></td>

           <td><ul><li>Graph Diffusion</li><li>K-means</li><li>Logistic Regression</li><li>Spatially-informed</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/C0nc/TAICHI" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="River identifies condition-relevant genes from spatial omics data across multiple slices or conditions. It learns gene expression and spatial coordinate embeddings using separate MLP-based encoders, which are then concatenated and used to predict condition labels. Spatial alignment is thus required as a preprocessing step. In a second step, River uses Integrated Gradients, DeepLift, and GradientShap to attribute model predictions to input genes at the cell level. These attribution scores are aggregated using rank aggregation to prioritize condition-relevant genes.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.08.04.606512v1.abstract">River</a></td>
           <td>2024</td>

           <td><ul><li>Differential Analysis</li></ul></td>

           <td><ul><li>Non-linear Classifier</li><li>Feature Attribution</li><li>Spatially-informed</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/C0nc/River" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="MUSIC evaluates sgRNA knockout efficiency and summarises perturbation effects using topic modeling. Following preprocessing steps, MUSIC removes low-efficiency (non-targeted) cells based on the cosine similarity of their differential expression genes, excluding perturbed cells with profiles more similar to controls. Next, highly dispersed DE genes are selected and their normalized expression values are used as to fit a topic model, where cells are treated as documents and gene counts as words. Topics are then ranked according to overall effect, their relevance to each perturbation, and perturbation similarities.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-019-10216-x">MUSIC</a></td>
           <td>2019</td>

           <td><ul><li>Perturbation Responsiveness</li><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>Topic Model</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/bm2-lab/MUSIC" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Mixscape aims to classify CRISPR-targeted cells into perturbed and not perturbed (escaping). To eachive that, Mixscape computes a local perturbation signature by subtracting each cell’s mRNA expression from the average of its k nearest NT (non-targeted) control neighbors. Differential expression testing between targeted and NT cells then identifies a set of DEGs that capture the perturbation response. These DEGs are used to define a perturbation vector-essentially, the average difference in expression between targeted and NT cells, which projects each cell’s DEG expression onto a single perturbation score. The Gaussian mixture model is applied to these perturbation scores, with one component fixed to match the NT distribution, while the other represents the perturbation effect. This model assigns probabilities that classify each targeted cell as either perturbed or escaping. Additionally, the authors propose visualization with Linear Discriminant Analysis and UMAP, aiming to identify a low-dimensional subspace that maximally discriminates the mixscape-derived classes.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41588-021-00778-2#Sec11">Mixscape</a></td>
           <td>2021</td>

           <td><ul><li>Perturbation Responsiveness</li><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>Gaussian Mixture Model</li><li>LDA\n</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/satijalab/seurat" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Perturbation Score (PS) quantifies single-cell responses to (&#34;dosage&#34;-informed) perturbations in three steps. First, a signature gene set is defined for each perturbation - either via differential expression or pre-defined gene sets. Second, scMAGeCK’s regression framework estimates gene-level coefficients (β) reflecting the average effect of each perturbation on its target genes. Third, a regularised regression model is fitted per cell to estimate a scalar PS [0-to-1], reflecting how well the cell’s gene expression profile matches the β-weighted perturbation signature. This is done through constrained optimisation, where scores are inferred only for cells annotated as receiving the perturbation.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41556-025-01626-9">Perturbation Score</a></td>
           <td>2025</td>

           <td><ul><li>Perturbation Responsiveness</li><li>Differential Analysis</li></ul></td>

           <td><ul><li>Robust Rank Aggregate</li><li>Ridge Regression</li><li>Regularised Linear Models</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/davidliwei/PS" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="For each gene-gRNA pair, SCEPTRE fits a negative binomial regression where the response is the gene’s expression across cells and the predictors are binary indicator denoting gRNA presence, plus technical covariates. Concurrently, a logistic regression using the same technical factors estimates π - the probability of detecting the gRNA in a cell. In a conditional resampling step, gRNA assignments are independently redrawn per cell based on π, generating an empirical null distribution of z‐scores; a skew‑t distribution is then fitted to this null to yield calibrated p‑values.">
           <td class="details-control"></td>
           <td><a href="https://link.springer.com/article/10.1186/s13059-021-02545-2#Sec11">SCEPTRE</a></td>
           <td>2021</td>

           <td><ul><li>Perturbation Responsiveness</li><li>Differential Analysis</li></ul></td>

           <td><ul><li>Conditional Resampling</li><li>Generalised Linear Model</li><li>NB Likelihood</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/Katsevich-Lab/sceptre" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Multicellular factor analysis repurposes MOFA by treating pseudobulked cell types as views. Each patient is represented by multiple views - one per cell type - summarizing gene expression. MOFA+ ised then used to identify latent factors that capture coordinated variability across these views, with loadings indicating cell-type-specific gene contributions. ">
           <td class="details-control"></td>
           <td><a href="https://elifesciences.org/articles/93161">MOFAcell</a></td>
           <td>2023</td>

           <td><ul><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>Group Factor Analysis (MOFA+)</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/saezlab/MOFAcellulaR" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Vespucci builds on Augur, and similarly it trains a random forest classifier to predict perturbation labels based on gene expression but extends this to spatial barcodes, using cross-validation within small, neighbouring regions to compute the area under the ROC curve (AUC) as a measure of transcriptional separability per observation. To overcome the computational inefficiency of classification across all observations, Vespucci employs a meta-learning approach: it first performs exhaustive classification on a subset of barcodes, then trains a random forrest regression model on derived distance metrics (e.g., Pearson correlation, Spearman correlation) between all pairs of observations to impute AUCs across the full dataset. This is done by iteratively expanding the number of observations in the training set until convergence (according to prediction similarity to the previous iteration). Finally, perturbation-responsive genes are identified by splitting the data (using an independent set of observations) to avoid bias, then using negative binomial mixed models to link gene expression to AUC scores.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.06.13.598641v2.full">Vespucci</a></td>
           <td>2024</td>

           <td><ul><li>Perturbation Responsiveness</li><li>Differential Analysis</li></ul></td>

           <td><ul><li>Random Forrest</li><li>Spatially-Informed</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/neurorestore/Vespucci" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="GEASS is a causal feature selection framework in high-dimensional spatal &amp; temporal omics data that identifies nonlinear Granger causal interactions by maximizing a sparsity-regularized modified transfer entropy. It enforces sparsity using combinatorial stochastic gate layers that allow it to select a minimal subset of features with causal interactions - i.e. two sets of of non-overlapping genes as drivers (source) and receivers (sink). ">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/forum?id=aKcS3xojnwY">GEASS</a></td>
           <td>2023</td>

           <td><ul><li>Nonlinear Gene Programmes</li></ul></td>

           <td><ul><li>Non-linear Granger Causality</li><li>Stochastic Gate Layers (Feature Selectors)</li><li>Time-resolved / Spatially-informed</li></ul></td>


           <td class="published">✓</td>
            <td>✗</td>
         </tr>
         <tr data-description="GEASS is a causal feature selection framework in high-dimensional spatal &amp; temporal omics data that identifies nonlinear Granger causal interactions by maximizing a sparsity-regularized modified transfer entropy. It enforces sparsity using combinatorial stochastic gate layers that allow it to select a minimal subset of features with causal interactions - i.e. two sets of of non-overlapping genes as drivers (source) and receivers (sink). ">
           <td class="details-control"></td>
           <td><a href="https://link.springer.com/article/10.1186/s13059-024-03334-3">MiloDE</a></td>
           <td>2024</td>

           <td><ul><li>Differential Analysis</li></ul></td>

           <td><ul><li>Generalised Linear Model</li><li>NB Likelihood</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/MarioniLab/miloDE" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Hotspot proposes a modified autocorrelation metrics that detect genes with coherent expression among neighboring cells (K-nearest neighbours graph in a latent space, spatial proximities, or lineage). By comparing these local autocorrelation scores to a permutation-free null model (e.g. using negative binomial or Bernoulli assumptions), it calculates the significance of autocorrelated genes. Additionally, for module detection, Hotspot computes pairwise correlations that capture how similarly two genes are expressed across nearby cells and then applies hierarchical clustering to group genes into biologically coherent modules.">
           <td class="details-control"></td>
           <td><a href="https://www.sciencedirect.com/science/article/pii/S2405471221001149">Hotspot</a></td>
           <td>2021</td>

           <td><ul><li>Differential Analysis</li><li>Nonlinear Gene Programmes</li></ul></td>

           <td><ul><li>Autocorrelation</li><li>Pairwise Local Correlations</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/YosefLab/Hotspot/tree/master" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="DIALOGUE identifies shared multicellular patterns across cell types and samples. It first constructs cell-type–specific data matrices by averaging features (e.g., gene expression or PCs) over samples or spatial niches. Then it applies multi-factor sparse canonical correlation analysis (referred to as penalized matrix decomposition (PMD)) to derive latent feature matrices that maximize cross-cell-type correlations under LASSO constraints. Following this initial PMD step, DIALOGUE employs correlation coefficients and permutation tests to determine which cell types contribute to each multicellular progarmmes (MCP). It then re-applies the PMD procedure in both a multi-way and a pairwise fashion, incorporating programs unique to the pairwise analysis into the downstream modeling. Finally, gene associated with MCPs are first identified using partial Spearman correlation and then refined through hierarchical mixed-effects modeling with covariate control.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41587-022-01288-0">DIALOGUE</a></td>
           <td>2022</td>

           <td><ul><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>Sparse CCA</li><li>Partial Correlations</li><li>Mixed Linear Model</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/livnatje/DIALOGUE" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Augur rank cell types by quantifying how accurately perturbation labels can be predicted from gene expression profiles using a random forest classifier (or regressor depending on the perturbation label). For each cell type, it repeatedly subsamples a fixed number of cells to mitigate biases from uneven cell numbers. It also employs a two-step feature selection procedure, first, identifying highly variable genes via local polynomial regression on the mean–variance relationship, and second, random downsampling. AUGUR then uses cross-validation to compute the area under the ROC curve (AUC) as a model performance metric that is used to quantify the perturbation effect on each cell type. It also provides (gene) feature importances. For multi-class or continous perturbations, cell-type effects (model performance) are measured using macro-averaged AUC or concordance correlation coefficient, respectively.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41587-020-0605-1#Abs1">AUGUR</a></td>
           <td>2020</td>

           <td><ul><li>Perturbation Responsiveness</li><li>Differential Analysis</li></ul></td>

           <td><ul><li>Random Forrest</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/neurorestore/Augur" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="scDist is a statistical framework that uses linear mixed-effects models to estimate gene-level condition effects while accounting for individual and technical variability. In the model, baseline expression levels are first captured, and then a parameter representing the condition-induced change is estimated. The overall shift between conditions is quantified by computing the Euclidean distance between the condition-specific mean expression profiles - essentially, by taking the norm of the condition effect vector. This high-dimensional metric is then efficiently approximated in a lower-dimensional space via principal component analysis.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-024-51649-3">scDIST</a></td>
           <td>2024</td>

           <td><ul><li>Perturbation Responsiveness</li><li>Differential Analysis</li></ul></td>

           <td><ul><li>Generalised Linear Model</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/phillipnicol/scDist" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="SubCell is a set of self-supervised vision transformer (ViT) models trained on low-plex single-cell immunofluorescence images from the Human Protein Atlas to learn biologically meaningful representations of protein localisation and tissue morphology. The models are optimised using a multi-task objective that combines masked autoencoding for spatial reconstruction, a cell-specific contrastive loss to enforce consistency across augmented views of the same cell, and a protein-specific contrastive loss to align embeddings of cells stained for the same protein across different cell lines and experiments. An attention pooling module is used to priotise informative subcellular regions. The resulting models, particularly ViT-ProtS-Pool and MAE-CellS-ProtS-Pool, are shown to generalise across datasets, imaging modalities, cell types, and perturbations without fine-tuning.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.12.06.627299v1.abstract">SubCell</a></td>
           <td>2024</td>

           <td><ul><li>Context Transfer</li><li>Nonlinear Gene Programmes</li></ul></td>

           <td><ul><li>A collection of Vision Transformer Models</li><li>Contrastive loss</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/czi-ai/sub-cell-embed" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="VirTues is a multi-modal foundation model based on a vision transformer architecture, trained on multiplex spatial proteomics data from lung, breast, and melanoma tumors. It combines image representations with protein language model (PLM) embeddings of molecular markers and constructs hierarchical summary tokens at the cell, niche, and tissue levels. This PLM-based tokenisation enables the model to predict previously unseen markers. The architecture employs a sparse attention mechanism that factorises attention into spatial and marker components to manage the computational complexity of high-dimensional input. Training is performed using a masked autoencoding objective - i.e. it reconstructs missing subsets of spatial and protein data.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/abs/2501.06039">VirTues</a></td>
           <td>2025</td>

           <td><ul><li>Context Transfer</li></ul></td>

           <td><ul><li>ViT</li><li>Maked Autoencoder</li><li>Multi-modal</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/bunnelab/virtues" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="OmiCLIP is a dual-encoder foundation model that encodes H&amp;E histology patches with a Vision Transformer and “gene-sentence” representations of (10X Visium) spatial transcriptomics (ST) using a Transformer initialised on LAION-5B. It projects both modalities into a shared latent space using symmetric contrastive learning, which pulls matched histology-transcriptome pairs. The model is pretrained on paired H&amp;E images and ST data across diverse organs and disease states. These aligned embeddings underpin downstream tasks such as spatial registration of histology to transcriptomic spots, zero-shot tissue annotation, cell-type deconvolution, retrieval of matching transcriptomic profiles for novel histology inputs, and prediction of spatial gene-expression patterns directly from histology.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-025-02707-1">OmiCLIP</a></td>
           <td>2025</td>

           <td><ul><li>Context Transfer</li></ul></td>

           <td><ul><li>ViT</li><li>Contrastive Learning</li><li>Multi-modal</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/GuangyuWangLab2021/Loki" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Prophet represents each experiment as a set of three axes - cellular state (cell lines), treatments (perturbations), and phenotypic readouts - and projects diverse prior knowledge types (e.g., CCLE bulk RNA-seq for cell lines; chemical fingerprints or transcriptomic/genomic vectors for perturbations; learnable embeddings for readouts) into a shared token space. It is pre-trained on a set of diverse perturbation experiments covering readouts such as cell viability, compound IC50, Cell Painting morphology features, mRNA transcript abundance, and cell type proportions. A transformer-based encoder integrates these tokenised inputs, feeding a regression head that’s trained end-to-end to minimise mean squared error across all outcome types. The model is fine-tuned for assay-specific data applications.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.08.12.607533v2">Prophet</a></td>
           <td>2024</td>

           <td><ul><li>Unseen Perturbation Prediction</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Transformer</li><li>Multi-modal</li><li>Knowledge Informed</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/theislab/prophet" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Iterpert is an active learning framework for Perturb-seq experiments that uses GEARS to predict gene expression gene expression perturbation effects. The method iteratively retrains GEARS on new data and selects the next batch of perturbations using an enhanced kernel, which is constructed by fusing the GEARS-derived kernel with kernels from six prior information sources (additional Perturb-seq data, optical pooled screens, scRNA-seq atlases, protein structures, protein–protein interaction networks, and literature-derived features). Each prior source is mapped into a kernel matrix, normalized, and combined with the model kernel via a mean fusion operator. The fused kernel is then used with a greedy distance maximization rule to select perturbation batches under budget constraints (a limited set of experiments per round). ">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2023.12.12.571389v1.full.pdf">IterPert</a></td>
           <td>2024</td>

           <td><ul><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>GEARS</li><li>Active Learning</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/Genentech/iterative-perturb-seq" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Decipher is a hierarchical deep generative model to integrate and visualize single-cell RNA-seq data from both normal and perturbed conditions, identifying shared and disrupted cell-state trajectories. Its architecture includes dual latent spaces -a low-dimensional state for detailed cell-state modeling and a two-dimensional space for visualization-connected to gene expression through linear or single-layer neural network transformations. The model aligns trajectories by maintaining shared transcriptional programs for common biological processes across conditions.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2023.11.11.566719v2.full">Decipher</a></td>
           <td>2024</td>

           <td><ul><li>Unsupervised Disentanglement</li><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>VAE</li><li>Linear Decoder</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/azizilab/decipher" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="SpatialDIVA learns distinct latent spaces capturing intrinsic (transcriptomic), morphological (histology), spatial neighborhood, technical (batch), and residual variations. To promote disentanglement, the model employs auxiliary classification heads - using cell type labels to supervise the transcriptomic latent space, pathology annotations to guide the histology latent space, and batch labels to capture technical variation. Additionally, an auxiliary regression head with mean squared error (MSE) loss is trained to ensure that the spatial latent space accurately reconstructs a PCA-based representation derived from concatenated histology and transcriptomic profiles from k-nearest spatial neighbours, thereby capturing both imaging and expression data from adjacent spots.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2025.02.19.638201v1.full.pdf">SpatialDIVA</a></td>
           <td>2025</td>

           <td><ul><li>Multi-component Disentanglement</li></ul></td>

           <td><ul><li>VAE</li><li>Spatially-informed</li><li>NB Likelihood</li><li>Multi-modal</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/hsmaan/SpatialDIVA" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="FLeCS models single-cell gene expression dynamics using coupled ordinary differential equations (ODEs) parameterized by a gene regulatory network. Cells are grouped into temporal bins—either via pseudotime inference or experimental timestamps—and aligned across time with optimal transport to form (pseudo)time series. To model interventions FLeCS replicates interventions in the learned graph.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/2503.20027">FLeCS</a></td>
           <td>2025</td>

           <td><ul><li>Context Transfer</li><li>GRN Inference</li><li>Causal Structure</li></ul></td>

           <td><ul><li>ODE</li><li>Optimal Transp</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/Bertinus/FLeCS" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="RENGE attempts to infer gene regulatory networks from time-series single-cell CRISPR knockout data. It models changes in gene expression following a knockout by propagating the effects through direct and higher-order (indirect) regulatory paths, where the gene network is represented as a matrix of regulatory strengths between gene pairs.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s42003-023-05594-4">RENGE</a></td>
           <td>2023</td>

           <td><ul><li>Context Transfer</li><li>GRN Inference</li><li>Causal Structure</li></ul></td>

           <td><ul><li>Regression model</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/masastat/RENGE" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="ARTEMIS combines variational autoencoder with an unbalanced Diffusion Schrödinger Bridge to reconstruct continuous trajectories from time-series data. The methods trains a VAE first, then jointly trains the VAE and uDSB by solving the forward-backward SDEs in the latent space using neural networks.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/biorxiv/early/2025/01/26/2025.01.23.634618.full.pdf">ARTEMIS</a></td>
           <td>2025</td>

           <td><ul><li>Trajectory Inference</li></ul></td>

           <td><ul><li>VAE</li><li>Schrödinger Bridge</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/daifengwanglab/ARTEMIS" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="scPRAM is a computational framework for predicting single-cell gene expression changes in response to perturbations. The method integrates three main components: a variational autoencoder (VAE), optimal transport, and an attention mechanism. The VAE encodes the gene expression data into a latent space. Optimal transport is applied in this latent space to match unpaired cells before and after perturbation by finding an optimal coupling between their distributions. For each test cell, the attention mechanism computes a perturbation vector by comparing its latent representation (query) against those of matched training cells (keys and values). The predicted post-perturbation response is generated by adding the perturbation vector to the query and decoding it back to gene expression space using the VAE decoder.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/40/5/btae265/7646141">scPRAM</a></td>
           <td>2024</td>

           <td><ul><li>Context Transfer</li><li>Trace Cell Populations</li></ul></td>

           <td><ul><li>VAE</li><li>OT</li><li>Attention</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/jiang-q19/scPRAM" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="MISTy extracts intra- and intercellular relationships from spatial omics data by learning multivariate interactions through a multi-view approach, where each view represents a collection of variables (e.g., a modality or an aggragation of a spatial niche). It jointly models spatial and functional aspects of the data, supporting any number of views with arbitrary numbers of variables. Target variables (intrinsic view) are predicted using random forests (by default), either via leave-feature-one-out within the intrinsic view or using the remaining (extrinsic) views.">
           <td class="details-control"></td>
           <td><a href="https://link.springer.com/article/10.1186/s13059-022-02663-5">MISTy</a></td>
           <td>2022</td>

           <td><ul><li>Feature Relationships</li></ul></td>

           <td><ul><li>Spatially-informed</li><li>Random Forrest (or other regression models)</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://saezlab.github.io/mistyR/" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="SpaCeNet aims to untangle the complex relationships between molecular interactions within and between cells by analyzing spatially resolved single-cell data. To achieve this, SpaCeNet leverages an adaptation of probabilistic graphical models  to enable spatially resolved conditional independence testing. This approach allows for the identification of direct and indirect dependencies, as well as the removal of spurious gene association patterns. Additionally, SpaCeNet incorporates explicit cell-cell distance information to differentiate between short- and long-range interactions, thereby distinguishing between baseline cellular variability and interactions influenced by a cell&#39;s microenvironment.">
           <td class="details-control"></td>
           <td><a href="https://genome.cshlp.org/content/34/9/1371">SpaCeNet</a></td>
           <td>2024</td>

           <td><ul><li>Feature Relationships</li></ul></td>

           <td><ul><li>Generalised Gaussian Graphical Model</li><li>Spatially-informed</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/sschrod/SpaCeNet" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Kasumi extends MISTy by focusing on identifying localised relationship patterns that are persistent across tissue samples. Instead of modeling global relationships, it uses a sliding-window approach to learn representations of local tissue patches (neighborhoods), characterized by multivariate, potentially non-linear relationships across views. These window-specific relationship signatures are clustered (using graph-based community detection) into spatial patterns, which are consistently observed across multiple samples. This enables Kasumi to represent each sample as a distribution over interpretable, shared local patterns, facilitating tasks like patient stratification while maintaining model explainability.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-025-59448-0">Kasumi</a></td>
           <td>2025</td>

           <td><ul><li>Feature Relationships</li></ul></td>

           <td><ul><li>Spatially-informed</li><li>Random Forrest (or other regression models)</li><li>Convolution Operations</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://www.github.com/jtanevski/kasumi" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="PRESCIENT models cellular differentiation as a stochastic diffusion process, where the drift term is parameterised as the negative gradient of a neural network-learned potential function. The model is trained using time-series single-cell RNA-seq data, and fits the potential function by minimizing the regularised Wasserstein distance between simulated and observed cell populations at each time point, explicitly incorporating cell proliferation by weighting cells according to their expected number of descendants. PRESCIENT can simulate differentiation trajectories for both observed and (in silico) perturbed cell states, enabling the prediction of cell fate outcomes under various genetic interventions.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-021-23518-w">Prescient</a></td>
           <td>2021</td>

           <td><ul><li>Trace Cell Populations</li><li>Seen Perturbation Prediction</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Diffusion</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/gifford-lab/prescient" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
       </tbody>
     </table>
   </div>

.. raw:: html

   <script>
   jQuery(function($){
      $('#methods-table').DataTable({
        order:      [[2,'desc']],
        pageLength: 5,
        lengthMenu: [5,10,20,50,200],
        scrollX:    true,
        autoWidth:  false
      });
     $('#methods-table tbody').on('click','td.details-control',function(){
       var tr = $(this).closest('tr'),
           row = $('#methods-table').DataTable().row(tr);
       if(row.child.isShown()){
         row.child.hide(); tr.removeClass('shown');
       } else {
         row.child('<div style="padding:0.5em;">'+tr.data('description')+'</div>').show();
         tr.addClass('shown');
       }
     });
   });
   </script>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   causal_structure
   combinatorial_effect_prediction
   context_transfer
   contrastive_disentanglement
   differential_analysis
   feature_relationships
   grn_inference
   linear_gene_programmes
   multi_component_disentanglement
   nonlinear_gene_programmes
   perturbation_responsiveness
   seen_perturbation_prediction
   trace_cell_populations
   unseen_perturbation_prediction
   unsupervised_disentanglement
