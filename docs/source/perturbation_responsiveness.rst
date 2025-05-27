Perturbation Responsiveness
===========================


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
         <tr data-description="A VAE that combines the contrastiveVI/cVAE architecture with a classifier that learns the pairing of perturbation labels to cells. As in ContrastiveVI, unperturbed cells are drawn solely from background latent space, while cells classified as perturbed are reconstructed from both the background and salient sapces. Additionally, Hilbert-Schmidt Independence Criterion (HSIC) is used to disentagle the background and salient latent spaces.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.01.05.574421v1.full">SC-VAE</a></td>
           <td>2024</td>

           <td><ul><li>Contrastive Disentanglement</li><li>Perturbation Responsiveness</li></ul></td>

           <td><ul><li>VAE</li><li>NB likelihood</li></ul></td>

           <td><ul><li>ContrastiveVI</li><li>scVI</li><li>cVAE</li></ul></td>

           <td class="published">✓</td>
            <td>✗</td>
         </tr>
         <tr data-description="An extension of ContrastiveVI that incorporates an auxiliary classifier to estimate the effects of perturbations, where the classifier operates on the salient variables and is sampled from a relaxed straight-through Bernoulli distribution. The output from the classifier also directly informs the salient latent space, indicating whether a cell expressing a gRNA successfully underwent a corresponding genetic perturbation. Additionally, Wasserstein distance is replaced by KL divergence, encouraging non-perturbed cells to map to the null region of the salient space. For datasets with a larger number of perturbations, the method also re-introduces and minimizes the Maximum Mean Discrepancy (MMD) between the salient and background latent variables. This discourages the leakage of perturbation-induced information into the background latent variables, ensuring a clearer separation of perturbation effects.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/abs/2411.08072">ContrastiveVI+</a></td>
           <td>2024</td>

           <td><ul><li>Perturbation Responsiveness</li><li>Contrastive Disentanglement</li><li>Nonlinear Gene Programmes</li></ul></td>

           <td><ul><li>ZINB Likelihood</li><li>VAE</li><li>Contrastive</li></ul></td>

           <td><ul><li>ContrastiveVI (theirs)</li></ul></td>

           <td class="published">✓</td>
            <td><a href="https://github.com/insitro/contrastive_vi_plus" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CINEMA‐OT disentangles perturbation effects from confounding variation by decomposing the data with independent component analysis (ICA); ICA components correlated with the perturbation labels are identified using Chatterjee’s coefficient and excluded, yielding a background (confounder) latent space that predominantly reflects confounding factors. Optimal transport is then applied to this background space to align perturbed and control cells, thereby generating counterfactual cell pairs, and this OT map is used in downstream analyses. They also propose a reweighting variant (CINEMA‐OT‐W) to address differential cell type abundance by pre-aligning treated cells with k‐nearest neighbor controls and balancing clusters prior to ICA and optimal transport.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-023-02040-5#Sec11">CINEMA-OT</a></td>
           <td>2023</td>

           <td><ul><li>Counterfactual Cell Maps</li><li>Perturbation Responsiveness</li><li>Unsupervised Disentanglement</li></ul></td>

           <td><ul><li>Unbalanced OT</li><li>Entropy‐regularized Sinkhorn</li><li>ICA</li></ul></td>

           <td><ul><li>Mixscape</li><li>OTT</li></ul></td>

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

           <td><ul><li>Counterfactual Cell Maps</li><li>Perturbation Responsiveness</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Dual (min-max) Formulation OT</li></ul></td>

           <td><ul><li>Makkuva et al</li><li>2020</li></ul></td>

           <td class="published">✓</td>
            <td><a href="https://github.com/bunnech/cellot" class="github-link">
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

           <td><ul><li>-</li></ul></td>

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

           <td><ul><li>-</li></ul></td>

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

           <td><ul><li>PLIER (PK representation)</li></ul></td>

           <td class="published">✓</td>
            <td><a href="https://github.com/KrishnaswamyLab/MELD" class="github-link">
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

           <td><ul><li>-</li></ul></td>

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

           <td><ul><li>MELD</li></ul></td>

           <td class="published">✗</td>
            <td><a href="https://github.com/C0nc/TAICHI" class="github-link">
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

           <td><ul><li>LDA</li><li>Correlated topic model</li></ul></td>

           <td class="published">✓</td>
            <td><a href="https://github.com/bm2-lab/MUSIC" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Mixscape aims to classify CRISPR-targeted cells into perturbed and not perturbed (escaping). To eachive that, Mixscape computes a local perturbation signature by subtracting each cell’s mRNA expression from the average of its k nearest NT (non-targeted) control neighbors. Differential expression testing between targeted and NT cells then identifies a set of DEGs that capture the perturbation response. These DEGs are used to define a perturbation vector—essentially, the average difference in expression between targeted and NT cells—which projects each cell’s DEG expression onto a single perturbation score. The Gaussian mixture model is applied to these perturbation scores, with one component fixed to match the NT distribution, while the other represents the perturbation effect. This model assigns probabilities that classify each targeted cell as either perturbed or escaping. Additionally, the authors propose visualization with Linear Discriminant Analysis (LDA) and UMAP, aiming to identify a low-dimensional subspace that maximally discriminates the mixscape-derived classes.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41588-021-00778-2#Sec11">Mixscape</a></td>
           <td>2021</td>

           <td><ul><li>Perturbation Responsiveness</li><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>Gaussian Mixture Model</li><li>LDA\n</li></ul></td>

           <td><ul><li>-</li></ul></td>

           <td class="published">✓</td>
            <td><a href="https://github.com/satijalab/seurat" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Perturbation Score (PS) quantifies single-cell responses to perturbations in three steps. First, differentially expressed genes (DEGs) are identified. Second, existing algorithms, such as MUSIC, MIMOSCA, scMAGeCK or SCEPTRE, are used to infer the average perturbation effect on these genes. Finally, each cell is assigned a Perturbation Score by minimizing the error between predicted and observed changes in gene expression.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41556-025-01626-9">Perturbation Score</a></td>
           <td>2025</td>

           <td><ul><li>Perturbation Responsiveness</li><li>Differential Analysis</li></ul></td>

           <td><ul><li>Pipeline</li></ul></td>

           <td><ul><li>scMageck (theirs)</li></ul></td>

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

           <td><ul><li>-</li></ul></td>

           <td class="published">✓</td>
            <td><a href="https://github.com/Katsevich-Lab/sceptre" class="github-link">
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

           <td><ul><li>Augur (theirs)</li></ul></td>

           <td class="published">✗</td>
            <td><a href="https://github.com/neurorestore/Vespucci" class="github-link">
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

           <td><ul><li>-</li></ul></td>

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

           <td><ul><li>AUGUR</li></ul></td>

           <td class="published">✓</td>
            <td><a href="https://github.com/phillipnicol/scDist" class="github-link">
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
       columns: [null,null,null,null,null,null,null,null],
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

