Differential Analysis
=====================


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
         <tr data-description="CellDrift fits a negative binomial GLM to scRNA-seq counts using cell type, perturbation, and their interaction as independent (predictor) variables, while also incorporating library size and batch effects. Pairwise contrast coefficients are then derived to quantify the difference between perturbed and control states across time points. These time series of contrast coefficients, representing the temporal trajectory of perturbation effects per gene, are subsequently analyzed using Fuzzy C-means clustering to group similar temporal patterns and Functional PCA to extract the dominant modes of temporal variance.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bib/article/23/5/bbac324/6673850#373524408">CellDrift</a></td>
           <td>2022</td>

           <td><ul><li>Differential Analysis</li></ul></td>

           <td><ul><li>Generalised Linear Model</li><li>NB Likelihood</li><li>Functional PCA</li><li>Fuzzy Clustering</li><li>Time-resolved</li><li>Perturbation-Covariate Interactions</li></ul></td>

           <td><ul><li>-</li></ul></td>

           <td class="published">✓</td>
            <td><a href="https://github.com/KANG-BIOINFO/CellDrift" class="github-link">
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

           <td><ul><li>-</li></ul></td>

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
         <tr data-description="GEDI learns a shared latent space and, for each sample, estimates a specific reconstruction function that maps latent states to observed gene expression profiles. This design captures inter-sample variability and enables differential expression analysis along continuous cell-state gradients without relying on predefined clusters. Optionally, it can incorporate prior knowledge.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-024-50963-0?fromPaywallRec=false">GEDI</a></td>
           <td>2024</td>

           <td><ul><li>Differential Analysis</li><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>Probabilistic</li><li>Sample-specific Decoders</li><li>PK Representations (optional)</li><li>RNA Velocity (optional)</li></ul></td>

           <td><ul><li>PLIER (PK representation)</li></ul></td>

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

           <td><ul><li>-</li></ul></td>

           <td class="published">✓</td>
            <td><a href="https://github.com/yelabucsf/scrna-parameter-estimation" class="github-link">
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
         <tr data-description="River identifies condition-relevant genes from spatial omics data across multiple slices or conditions. It learns gene expression and spatial coordinate embeddings using separate MLP-based encoders, which are then concatenated and used to predict condition labels. Spatial alignment is thus required as a preprocessing step. In a second step, River uses Integrated Gradients, DeepLift, and GradientShap to attribute model predictions to input genes at the cell level. These attribution scores are aggregated using rank aggregation to prioritize condition-relevant genes.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.08.04.606512v1.abstract">River</a></td>
           <td>2024</td>

           <td><ul><li>Differential Analysis</li></ul></td>

           <td><ul><li>Non-linear Classifier</li><li>Feature Attribution</li><li>Spatially-informed</li></ul></td>

           <td><ul><li>-</li></ul></td>

           <td class="published">✗</td>
            <td><a href="https://github.com/C0nc/River" class="github-link">
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
         <tr data-description="GEASS is a causal feature selection framework in high-dimensional spatal &amp; temporal omics data that identifies nonlinear Granger causal interactions by maximizing a sparsity-regularized modified transfer entropy. It enforces sparsity using combinatorial stochastic gate layers that allow it to select a minimal subset of features with causal interactions - i.e. two sets of of non-overlapping genes as drivers (source) and receivers (sink). ">
           <td class="details-control"></td>
           <td><a href="https://link.springer.com/article/10.1186/s13059-024-03334-3">MiloDE</a></td>
           <td>2024</td>

           <td><ul><li>Differential Analysis</li></ul></td>

           <td><ul><li>Generalised Linear Model</li><li>NB Likelihood</li></ul></td>

           <td><ul><li>Milo (theirs)</li><li>edgeR</li><li>Cydar</li></ul></td>

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

           <td><ul><li>Spatial Autocorrelation Metrics (e.g. Morans I)</li></ul></td>

           <td class="published">✓</td>
            <td><a href="https://github.com/YosefLab/Hotspot/tree/master" class="github-link">
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

