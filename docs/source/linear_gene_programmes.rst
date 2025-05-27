Linear Gene Programmes
======================


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
         <tr data-description="A modified version of PCA, where the covariance matrix (COV) is the difference between COV(case/target) and αCOV(control/background). The hyperparameter α is used to balance having a high case variance and a low control variance. To provide some intuition, when α is 0, the model reduces to classic PCA on the case data.  Optimal alphas (equal to k clusters) are identified using spectral clustering over a range of cPCA runs with different alphas, with selection based on the similarity of cPCA outputs.">
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
         <tr data-description="A non-negative matrix factorisation that decomposes gene expression matrices into common and specific patterns. For each condition, the observed expression matrix is approximated as the sum of a common component - represented by a common feature matrix (Wc) with condition-specific coefficient matrices (Hc₁, Hc₂) - and a specific component unique to each condition, represented by its own feature matrix (Wsᵢ) and coefficients (Hsᵢ). The model uses an alternating approach to minimize the combined reconstruction error (squared Frobenius norm) across common and shared components.">
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
         <tr data-description="A family of contrastive latent variable models (cLVMs), where case data are modeled as the sum of background and salient latent embeddings, while control data are reconstructed solely from background embeddings: - cLVM with Gaussian likelihoods and priors - Sparse cLVM with horseshoe prior used to regularize the weights - Robust cLVM with a Student&#39;s t distribution - cLVM with automatic relevance determination (ARD) to regularize (select) the columns of the weight matrix - contrastive VAE, as a non-linear extension of the framework The shared concept across these models is that each model learns a shared set of latent variables for the background and target data, while salient latent variables are learnt solely for the target data.">
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
         <tr data-description="A sparse version of contrastive PCA that enhances interpretability in high-dimensional settings by integrating ℓ1regularization into an iterative procedure to estimate sparse loadings and principal components">
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
         <tr data-description="CellCap is a deep generative model that extends CPA by incorporating cross-attention mechanisms between cell states, aimed at understanding transcriptional response programs and reconstructing perturbed profiles. Further, CellCap uses a variational autoencoder (VAE) framework with a linear decoder to identify sparse and interpretable latent factors.">
           <td class="details-control"></td>
           <td><a href="https://www.cell.com/cell-systems/fulltext/S2405-4712(25)00078-X">CellCap</a></td>
           <td>2024</td>

           <td><ul><li>Multi-component Disentanglement</li><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>VAE</li><li>Attention</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/broadinstitute/CellCap" class="github-link">
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
           <td><a href="https://scholar.google.com/scholar_url?url=https://www.nature.com/articles/s41588-025-02120-6&hl=en&sa=X&d=4385431678967561370&ei=jfHcZ9q4G5yV6rQP4Zul2Qo&scisig=AFWwaea2QWdmQLBJLz29SV6YD2cm&oi=scholaralrt&hist=dDujacgAAAAJ:12160454169637496643:AFWwaebf1S6Ukws-5zfatGcdJi9a&html=&pos=0&folt=art">NicheCompass</a></td>
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
         <tr data-description="Waddington-OT models developmental processes as time‐varying probability distributions in gene expression space and infers temporal couplings by solving an entropy‐regularized, unbalanced optimal transport problem. Growth rate, estimated leveraging expression levels of genes associated with proliferation and apoptosis, is taken into consideration via unbalanced OT. Additionally, uses spectral clustering to obtain Gene Programmes, and subsequently associate those to predictive TFs.">
           <td class="details-control"></td>
           <td><a href="https://www.sciencedirect.com/science/article/pii/S009286741930039X?via%3Dihub">Waddington-OT</a></td>
           <td>2019</td>

           <td><ul><li>Counterfactual Cell Maps</li><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>Unbalanced OT</li><li>Entropy‐regularized Sinkhorn</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/broadinstitute/wot" class="github-link">
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
         <tr data-description="Mixscape aims to classify CRISPR-targeted cells into perturbed and not perturbed (escaping). To eachive that, Mixscape computes a local perturbation signature by subtracting each cell’s mRNA expression from the average of its k nearest NT (non-targeted) control neighbors. Differential expression testing between targeted and NT cells then identifies a set of DEGs that capture the perturbation response. These DEGs are used to define a perturbation vector—essentially, the average difference in expression between targeted and NT cells—which projects each cell’s DEG expression onto a single perturbation score. The Gaussian mixture model is applied to these perturbation scores, with one component fixed to match the NT distribution, while the other represents the perturbation effect. This model assigns probabilities that classify each targeted cell as either perturbed or escaping. Additionally, the authors propose visualization with Linear Discriminant Analysis (LDA) and UMAP, aiming to identify a low-dimensional subspace that maximally discriminates the mixscape-derived classes.">
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

