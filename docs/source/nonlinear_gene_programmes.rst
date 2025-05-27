Nonlinear Gene Programmes
=========================


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
           <th>Model</th><th>Published</th><th>Code</th>
         </tr>
       </thead>
       <tbody>
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
         <tr data-description="An extension of ContrastiveVI that incorporates an auxiliary classifier to estimate the effects of perturbations, where the classifier operates on the salient variables and is sampled from a relaxed straight-through Bernoulli distribution. The output from the classifier also directly informs the salient latent space, indicating whether a cell expressing a gRNA successfully underwent a corresponding genetic perturbation. Additionally, Wasserstein distance is replaced by KL divergence, encouraging non-perturbed cells to map to the null region of the salient space. For datasets with a larger number of perturbations, the method also re-introduces and minimizes the Maximum Mean Discrepancy (MMD) between the salient and background latent variables. This discourages the leakage of perturbation-induced information into the background latent variables, ensuring a clearer separation of perturbation effects.">
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
         <tr data-description="SIMVI is a spatially-informed VAE that disentangles gene expression variability into two latent factors: an intrinsic variable z, which captures cell type–specific signals, and a spatial variable s, which quantifies spatial effects. The spatial latent variable s is inferred by aggregating the intrinsic representations of neighboring cells via a Graph Attention Network (GAT), thereby incorporating local spatial context. To promote independence between z and s, SIMVI employs an asymmetric regularization on z using maximum mean discrepancy (MMD) or, alternatively, a  mutual information estimator, ensuring that z retains minimal non-cell-intrinsic information. Furthermore, leveraging debiased machine learning principles, the model decomposes gene expression variance by treating s as a continuous treatment and z as confounding covariates, thereby quantifying the specific impact of spatial context on gene expression.">
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
         <tr data-description="scGPT processes each cell as a sequence of gene tokens, expression-value tokens and condition tokens (e.g., batch, perturbation or modality), embedding each and summing before feeding them into stacked transformer blocks whose specialised, masked multi-head attention layers enable autoregressive prediction of masked gene expressions from non-sequential data. scGPT is pretrained using a masked gene expression-prediction objective that jointly optimizes cell and gene embeddings, and can be fine-tuned on smaller datasets with task-specific supervised losses. For gene regulatory network inference, scGPT derives k-nearest neighbor similarity graphs from learned gene embeddings and analyses attention maps to extract context-specific Gene Programmes and gene-gene interactions.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-024-02201-0">scGPT</a></td>
           <td>2024</td>

           <td><ul><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li><li>GRN Inference</li><li>Nonlinear Gene Programmes</li></ul></td>

           <td><ul><li>Foundational Gene expression embeddings (from >33M human cells)</li><li>Self-supervised masked expression prediction</li><li>Customised non-sequential (flash) attention</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/bowang-lab/scGPT" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="scFoundation uses an asymmetric transformer encoder–decoder: its embedding module converts each continuous gene expression scalar directly into a high-dimensional learnable vector without discretization; the encoder takes as input only nonzero and unmasked embeddings through vanilla transformer blocks to model gene–gene dependencies efficiently. The zero and masked gene embeddings, along with the encoder embeddings, are passed to the decoder, which uses Performer-style attention to reconstruct transcriptome-wide representations, specifically those of masked genes. Specifically, scFoundation is trained using a masked regression objective on both raw and downsampled count vectors, with two total-count tokens concatenated to inputs to account for sequencing depth variance. The decoder-derived gene context embeddings are then used as node features in GEARS for single-cell perturbation response prediction.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-024-02305-7">scFoundation</a></td>
           <td>2024</td>

           <td><ul><li>Nonlinear Gene Programmes</li><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li><li>Feature Relationships</li></ul></td>

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

           <td><ul><li>Nonlinear Gene Programmes</li><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li><li>GRN Inference</li></ul></td>

           <td><ul><li>Foundational Gene expression embeddings (from >50M human cells)</li><li>Self-supervised masked regression with down-sampling</li><li>Sparse transformer encoder</li><li>Performer-style attention decoder</li><li>PK-informed</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/xCompass-AI/GeneCompass" class="github-link">
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

