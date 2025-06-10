Unsupervised Disentanglement
============================


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

