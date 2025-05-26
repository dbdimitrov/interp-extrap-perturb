All Methods
===========

.. raw:: html

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
           <th>Expand</th><th>Method</th><th>Year</th><th>Task</th>
           <th>Model</th><th>Inspired by</th><th>Published</th><th>Code</th>
         </tr>
       </thead>
       <tbody>
         <tr data-description="A modified version of PCA, where the covariance matrix (COV) is the difference between COV(case/target) and αCOV(control/background). The hyperparameter α is used to balance having a high case variance and a low control variance. To provide some intuition, when α is 0, the model reduces to classic PCA on the case data.  Optimal alphas (equal to k clusters) are identified using spectral clustering over a range of cPCA runs with different alphas, with selection based on the similarity of cPCA outputs.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-018-04608-8#Sec7">cPCA</a></td>
           <td>2018</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>Modified PCA</li></ul></td>

           <td><ul><li>PCA</li><li>Contrastive Mixture Models (Zou et al.</li><li>2013)</li></ul></td>

           <td class="published">✓</td>
           <td>
             <a href="https://github.com/abidlabs/contrastive" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="A non-negative matrix factorisation that decomposes gene expression matrices into common and specific patterns. For each condition, the observed expression matrix is approximated as the sum of a common component - represented by a common feature matrix (Wc) with condition-specific coefficient matrices (Hc₁, Hc₂) - and a specific component unique to each condition, represented by its own feature matrix (Wsᵢ) and coefficients (Hsᵢ). The model uses an alternating approach to minimize the combined reconstruction error (squared Frobenius norm) across common and shared components.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/nar/article/47/13/6606/5512984">CSMF</a></td>
           <td>2019</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>NMF</li></ul></td>

           <td><ul><li>iNMF</li><li>NMF</li></ul></td>

           <td class="published">✓</td>
           <td>
             <a href="https://www.zhanglab-amss.org/homepage/software.html" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="A family of contrastive latent variable models (cLVMs), where case data are modeled as the sum of background and salient latent embeddings, while control data are reconstructed solely from background embeddings: - cLVM with Gaussian likelihoods and priors - Sparse cLVM with horseshoe prior used to regularize the weights - Robust cLVM with a Student&#39;s t distribution - cLVM with automatic relevance determination (ARD) to regularize (select) the columns of the weight matrix - contrastive VAE, as a non-linear extension of the framework The shared concept across these models is that each model learns a shared set of latent variables for the background and target data, while salient latent variables are learnt solely for the target data.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/abs/1811.06094">cLVM</a></td>
           <td>2019</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>Factor Models</li><li>Contastive VAE</li></ul></td>

           <td><ul><li>Contrastive PCA</li></ul></td>

           <td class="published">✓</td>
           <td>
             <a href="https://github.com/kseverso/contrastive-LVM" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="VAE with two sets of latent variables (two encoders): salient and background, each learned using amortised inference from both case and control observations, respectively. The latent variables are concatenated and then decoded simultaneously via a shared decoder. During the generative process (decoding), the control observations are reconstructed solely from the background latent space, with salient latent variables being set to 0, while the case observations are generated from both sets of latent variables. Optionally, the two sets of latent variables can be further disentagled by minimizing their total correlation, in practice done by training a discriminator to distinguish real from permuted latent samples.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/1902.04601">cVAE</a></td>
           <td>2019</td>

           <td><ul><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>Contrastive VAE</li></ul></td>

           <td><ul><li>Contrastive PCA</li></ul></td>

           <td class="published">✗</td>
           <td>
             <a href="https://github.com/abidlabs/contrastive_vae" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="A sparse version of contrastive PCA that enhances interpretability in high-dimensional settings by integrating ℓ1regularization into an iterative procedure to estimate sparse loadings and principal components">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/36/11/3422/5807607">scPCA</a></td>
           <td>2020</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>Modified PCA</li></ul></td>

           <td><ul><li>Contrastive PCA</li><li>Probabilistic PCA</li></ul></td>

           <td class="published">✓</td>
           <td>
             <a href="https://github.com/PhilBoileau/EHDBDscPCA" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="MichiGAN is a two-step approach that first uses a β-TCVAE - a variant of the variational autoencoder that penalizes total correlation among latent variables to promote disentangled representations. These latent representations (posterior means or samples) are then used to condition a Wasserstein GAN, the generator of which similarly to the VAE reconstructs the data from the latent variables, while attempting to &#39;fool&#39; a discriminator whether the samples were real or generated. Counterfactual predictions are done via latent space arithmetics as in scGEN.">
           <td class="details-control"></td>
           <td><a href="https://link.springer.com/article/10.1186/s13059-021-02373-4">MichiGAN</a></td>
           <td>2021</td>

           <td><ul><li>Unsupervised Disentanglement</li><li>Seen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>VAE</li><li>conditioned GAN</li></ul></td>

           <td><ul><li>scGEN</li><li>InfoGAN</li></ul></td>

           <td class="published">✓</td>
           <td>
             <a href="https://github.com/welch-lab/MichiGAN" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="A probabilistic model that builds on cPCA, additionally proposing a case-control-ratio-adjusted α as a more interpretable alternative to the same parameter in cPCA (see comment above).">
           <td class="details-control"></td>
           <td><a href="https://projecteuclid.org/journals/annals-of-applied-statistics/volume-18/issue-3/Probabilistic-contrastive-dimension-reduction-for-case-control-study-data/10.1214/24-AOAS1877.short">PCPCA</a></td>
           <td>2024</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>modified PCA</li></ul></td>

           <td>&nbsp;</td>

           <td class="published">✓</td>
           <td>
             <a href="https://github.com/andrewcharlesjones/pcpca" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="A family of contrastive Poisson latent variable models (CPLVMs), based on a Gamma-Poisson hierarchical generative process: - CPLVM: The variational posterior is approximated using log-normal distributions, preserving non-negativity in the latent factors. - CGLVM: Extends CPLVM by allowing latent factors to take negative values, replacing Gamma priors with Gaussian priors and using a log-link function for the Poisson rates. Variational posteriors are modeled as multivariate Gaussians. The authors also propose a hypothesis testing framework, in which log-(ELBO)-Bayes is calculated between a Null model, omitting the salient latent space, and the full contrastive model. This framework is used to quantify global (across all genes) and joint expression changes in subsets of genes (akin to gene set enrichment analysis).">
           <td class="details-control"></td>
           <td><a href="https://projecteuclid.org/journals/annals-of-applied-statistics/volume-16/issue-3/Contrastive-latent-variable-modeling-with-application-to-case-control-sequencing/10.1214/21-AOAS1534.short">CPLVMs</a></td>
           <td>2022</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>NB likelihood</li><li>Factor Models</li></ul></td>

           <td><ul><li>cPCA</li><li>cLVMs</li><li>scVI (hypothesis testing)</li></ul></td>

           <td class="published">✓</td>
           <td>
             <a href="https://github.com/andrewcharlesjones/cplvm" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="Spike and Slab Lasso applied to (non-linear) decoder weights. They show poofs of identifiability when at least 2 &#34;anchor features&#34; are present.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/2110.10804">sparseVAE</a></td>
           <td>2022</td>

           <td><ul><li>Unsupervised Disentanglement</li></ul></td>

           <td><ul><li>VAE</li></ul></td>

           <td><ul><li>oi-VAE</li><li>VSC</li><li>beta-VAE</li></ul></td>

           <td class="published">✓</td>
           <td>
             <a href="https://github.com/gemoran/sparse-vae-code" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="The successor to mmVAE introducing improvements: counts are modeled using a negative binomial distribution, and the MMD loss is replaced with the Wasserstein distance. More specifically, the Wasserstein distance is computed exclusively for the salient latent variables of the control data, ensuring it approaches zero. The Wasserstein penalty is optional and is set to 0 (no penalty) by default">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-023-01955-3">ContrastiveVI</a></td>
           <td>2023</td>

           <td><ul><li>Non-linear Gene Programmess</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>ZINB Likelihood</li><li>Protein-Count (totalVI) Likelihood</li><li>Contrastive VAE</li><li>Multi-modal</li></ul></td>

           <td><ul><li>scVI / totalVI</li><li>cVAE</li><li>Conditional VAE</li><li>mmVAE (theirs)</li></ul></td>

           <td class="published">✓</td>
           <td>
             <a href="https://github.com/scverse/scvi-tools/tree/main/src/scvi/external/contrastivevi" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="A Contrastive VAE framework, similar to cVAE, which additionally incorporates a maximum mean discrepancy (MMD) loss to enforce salient latent variables in the control data to approach zero, while also using it to align the background latent variables between case and control conditions.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/2202.10560">mmVAE</a></td>
           <td>2022</td>

           <td><ul><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>Contrastive VAE</li></ul></td>

           <td>&nbsp;</td>

           <td class="published">✓</td>
           <td>
             <a href="https://github.com/suinleelab/MM-cVAE" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="An extension of ContrastiveVI to multi-case (multi-group) disentaglement via multiple group-specific salient encoders.">
           <td class="details-control"></td>
           <td><a href="https://proceedings.mlr.press/v200/weinberger22a">MultiGroupVI</a></td>
           <td>2022</td>

           <td><ul><li>Non-linear Gene Programmess</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>ZINB Likelihood</li><li>VAE</li><li>Contrastive</li></ul></td>

           <td><ul><li>ContrastiveVI (theirs)</li></ul></td>

           <td class="published">✓</td>
           <td>
             <a href="https://github.com/Genentech/multiGroupVI" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="VAE model, which incorporates technical and biological covariates into two sets of latent variables:  - Z_I embeds biologically-relevant variables - Z_B embeds the unwanted variability in the data (i.e. batch effect labels) These are then fed into a shared encoder, along with the count data. The output of this shared encoder is fed to the decoder. Optionally, further disentanglement of the two latent variable sets is achieved by minimizing their total correlation, which is approximated via a minibatch-weighted estimator that quantifies the difference between the joint posterior and the product of individual marginal distributions.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.12.06.627196v1.full">inVAE</a></td>
           <td>2024</td>

           <td><ul><li>Multi-component Disentanglement</li><li>Non-linear Gene Programmess</li></ul></td>

           <td><ul><li>VAE</li><li>NB Likelihood</li></ul></td>

           <td><ul><li>scVI</li><li>iVAE</li><li>β-TCVAE</li></ul></td>

           <td class="published">✗</td>
           <td>
             <a href="https://github.com/theislab/inVAE" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="A VAE that disentangles disease (case) from healthy (control) cells by learning invariant background and salient space representations. The background and salient representations are summed to reconstruct the count data, with an (optional) interaction term capturing the interplay between cell type and disease. As done in contrastive methods, the salient representation for control cells is set to 0 during the generative (data reconstruction) process. The invariance of the background latent variables is enforced through two GAN-style neural networks: one encouraging the prediction of cell types from the background space, while the other penalises the prediction of disease labels, ensuring that disease-specific information is isolated in the salient space.">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/pdf?id=fkoqMdTlEg">scDSA</a></td>
           <td>2023</td>

           <td><ul><li>Non-linear Gene Programmess</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>NB likelihood</li><li>Domain-Adversarial NNs</li><li>VAE</li><li>Addative Shift</li></ul></td>

           <td><ul><li>DANN</li><li>DIVA</li><li>CPA</li><li>scVI</li></ul></td>

           <td class="published">✓</td>
           <td>
             <a href="-" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="A VAE that encodes input data into background latent variables and learns sparse, global (salient) embeddings representing the effects of perturbations. These sparse salient embeddings are modeled using a joint relaxed straight-through (Beta-)Bernoulli distribution (mask) and a normally distributed latent space. This method captures perturbation-specific effects as an additive shift to the background representation, analogous to additive shift methods, but it can also be thought as a multi-condition extention to the contrastive framework (limited to two latent variables (case vs. control), to a more general setup capable of learning global embeddings for each perturbation. As in some contrastive methods, for perturbation samples, the perturbation (global) embeddings are added to the background latent variables to reconstruct the data, while for control samples, the perturbation embeddings are effectively set to zero. ">
           <td class="details-control"></td>
           <td><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/0001ca33ba34ce0351e4612b744b3936-Abstract-Conference.html">SAMS-VAE</a></td>
           <td>2023</td>

           <td><ul><li>Multi-component Disentanglement</li><li>Causal Structure</li><li>Seen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>VAE</li><li>NB likelihood</li><li>Conditional Latent Embeddings</li><li>Addative Shift</li><li>Sparse Mechanism Shift</li></ul></td>

           <td><ul><li>CPA</li><li>SVAE/SVAE+</li></ul></td>

           <td class="published">✓</td>
           <td>
             <a href="https://github.com/insitro/sams-vae" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="A VAE  that combines the sparse mechanism shift from SVAE+ with learning a probabilistic pairing between cells and unobserved auxiliary variables. These auxilary variables correspond to the observed perturbation labels in SVAE+, but here they are learned in a data-driven way (rather than passed as static labels) which in turn enables counterfactual context-transfer scenarios.">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/pdf?id=8hptqO7sfG">svae-ligr</a></td>
           <td>2024</td>

           <td><ul><li>Seen Perturbation Prediction</li><li>Context Transfer</li><li>Multi-component Disentanglement</li></ul></td>

           <td><ul><li>VAE</li><li>NB likelihood</li><li>Sparse Mechanism Shift</li><li>Generative/Experience Replay</li></ul></td>

           <td><ul><li>SVAE+</li></ul></td>

           <td class="published">✓</td>
           <td>
             <a href="https://github.com/theislab/svaeligr" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="A VAE that integrates recent advances in sparse mechanism shift modeling for single-cell data, inferring a causal structure where perturbation labels identify the latent variables affected by each perturbation. The method constructs a graph identifying which latent variables are influenced by specific perturbations, promoting disentaglement and enabling biological interpretability, such as uncovering perturbations affecting shared processes. A key modelling contribution is its probabilistic sparsity approach (relaxed straight-through Beta-Bernoulli) on the global sparse embeddings (graph),  improving upon its predecessor, SVAE. As such, the latent space can be seen as being modelled from a Spike-and-Slab prior.">
           <td class="details-control"></td>
           <td><a href="https://proceedings.mlr.press/v213/lopez23a/lopez23a.pdf">sVAE+</a></td>
           <td>2023</td>

           <td><ul><li>Seen Perturbation Prediction</li><li>Multi-component Disentanglement</li><li>Causal Structure</li><li>Non-linear Gene Programmess</li></ul></td>

           <td><ul><li>VAE</li><li>NB likelihood</li><li>Sparse Mechanism Shift</li></ul></td>

           <td><ul><li>SVAE</li></ul></td>

           <td class="published">✓</td>
           <td>
             <a href="https://github.com/Genentech/sVAE" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="CausCell integrates causal representation learning with diffusion-based generative modeling to generate counterfactual single-cell data. It disentangles observed and unobserved concepts using concept-specific adversarial discriminators and links the resulting latent representations through a structural causal model encoded as a directed acyclic graph. The use of a diffusion model, instead of a traditional variational autoencoder, improves sample fidelity and better preserves underlying causal relationships during generation.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/biorxiv/early/2024/12/17/2024.12.11.628077.full.pdf">CausCell</a></td>
           <td>2024</td>

           <td><ul><li>Multi-component Disentanglement</li><li>Causal Structure</li><li>Combinatorial Effect Prediction</li><li>Context Transfer</li><li>Seen Perturbations</li></ul></td>

           <td><ul><li>Diffusion</li><li>Auxilary Classifiers</li></ul></td>

           <td><ul><li>AnnealVAE</li><li>DDPM</li></ul></td>

           <td class="published">✗</td>
           <td>
             <a href="-" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="A VAE that combines the contrastiveVI/cVAE architecture with a classifier that learns the pairing of perturbation labels to cells. As in ContrastiveVI, unperturbed cells are drawn solely from background latent space, while cells classified as perturbed are reconstructed from both the background and salient sapces. Additionally, Hilbert-Schmidt Independence Criterion (HSIC) is used to disentagle the background and salient latent spaces.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.01.05.574421v1.full">SC-VAE</a></td>
           <td>2024</td>

           <td><ul><li>Contrastive Disentanglement</li><li>Perturbation Responsiveness</li></ul></td>

           <td><ul><li>VAE</li><li>NB likelihood</li></ul></td>

           <td><ul><li>ContrastiveVI</li><li>scVI</li><li>cVAE</li></ul></td>

           <td class="published">✓</td>
           <td>
             <a href="-" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
         <tr data-description="Celcomen (CCE) disentangles intra- and inter-cellular gene regulation in spatial transcriptomics data by processing gene expression through two parallel interaction functions. One function uses a graph convolution layer (k-hop GNN) to learn a gene-gene interaction matrix that captures cross-cell signaling, while the other applies a linear layer to model regulation within individual cells. During training, Celcomen combines a normalization term—computed via a mean field approximation that decomposes the overall likelihood into a mean contribution and an interaction contribution - with a similarity measure that directly compares each cell’s predicted gene expression (obtained via message passing) to its actual expression, thereby driving the model to adjust its interaction matrices so that the predictions closely match the observed data. Simcomen (SCE) then leverages these fixed, learned matrices to simulate spatial counterfactuals (e.g., gene knockouts) for in-silico experiments.">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/pdf?id=Tqdsruwyac">Celcomen</a></td>
           <td>2025</td>

           <td><ul><li>Unsupervised Disentanglement</li><li>Feature relationships</li></ul></td>

           <td><ul><li>K-hop Convolution</li><li>Mean field estimation</li><li>Spatially-informed</li></ul></td>

           <td><ul><li>-</li></ul></td>

           <td class="published">✓</td>
           <td>
             <a href="https://github.com/Teichlab/celcomen" class="github-link">
               <i class="fab fa-github" aria-hidden="true"></i>
               <span class="sr-only">GitHub</span>
             </a>
           </td>
         </tr>
       </tbody>
     </table>
   </div>

.. raw:: html

   <script>
   jQuery(function($){
     $('#methods-table').DataTable({
       columns: [null,null,null,null,null,null,null,null],
       order:      [[2,'desc']],
       pageLength: 20,
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