Multi-component Disentanglement
===============================


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
         <tr data-description="CausCell integrates causal representation learning with diffusion-based generative modeling to generate counterfactual single-cell data. It disentangles observed and unobserved concepts using concept-specific adversarial discriminators and links the resulting latent representations through a structural causal model encoded as a directed acyclic graph. The use of a diffusion model, instead of a traditional variational autoencoder, improves sample fidelity and better preserves underlying causal relationships during generation.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/biorxiv/early/2024/12/17/2024.12.11.628077.full.pdf">CausCell</a></td>
           <td>2024</td>

           <td><ul><li>Multi-component Disentanglement</li><li>Causal Structure</li><li>Combinatorial Effect Prediction</li><li>Context Transfer</li><li>Seen Perturbation Prediction</li></ul></td>

           <td><ul><li>Diffusion</li><li>Auxilary Classifiers</li></ul></td>


           <td class="published">✗</td>
            <td>✗</td>
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
         <tr data-description="A VAE that disentangles control and pertubed cells into a latent space organized by a causal DAG. The encoder produces a Gaussian latent code z, while an intervention encoder transforms intervention one-hot encodings into two embeddings—a soft assignment vector that targets specific latent dimensions and a scalar capturing the intervention’s magnitude. Multiplying and adding these embeddings to z yields a modified latent vector that simulates a soft intervention, whereas zeroing them recovers the control condition. A causal layer then processes the latent vectors using an upper-triangular matrix G, which enforces an acyclic causal structure and propagates intervention effects among the latent factors. The decoder is applied twice—once to the modified latent code to generate virtual counterfactual outputs that reconstruct interventional outcomes, and once to the unmodified code to recover control samples. This dual decoding forces the model to disentangle intervention-specific effects from the intrinsic data distribution. The training objective combines reconstruction error to reconstruct control samples, a discrepancy loss (e.g., MMD) to align virtual counterfactuals with observed interventional data, KL divergence on the latent space, and an L1 penalty on G to enforce sparsity.">
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

