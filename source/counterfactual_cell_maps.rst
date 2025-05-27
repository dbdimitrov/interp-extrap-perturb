Counterfactual Cell Maps
========================


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
         <tr data-description="CondOT builds on CellOT to learn context-aware optimal transport maps by conditioning on an auxiliary variable. Instead of learning a fixed transport map, it learns a context-dependent transport map that adapts based on this auxiliary information. For each condition, CondOT learns how to transform a source distribution so that it closely matches a corresponding target distribution. The OT map is modeled as the gradient of a convex potential using partially input convex neural networks (PICNN), which ensures mathematical properties required for parametrised optimal transport. The auxiliary variables can be of different types: continuous (like dosage or spatial coordinates), categorical (like treatment groups, represented via one-hot encoding), or learned embeddings learned. Additionally, CondOT includes a separate neural module, a combinator network, for combinatorial predictions.">
           <td class="details-control"></td>
           <td><a href="https://proceedings.neurips.cc/paper_files/paper/2022/file/2d880acd7b31e25d45097455c8e8257f-Paper-Conference.pdf">CondOT</a></td>
           <td>2022</td>

           <td><ul><li>Counterfactual Cell Maps</li><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Conditioned Dual (min-max) Formulation OT</li></ul></td>

           <td><ul><li>Amos et al.</li><li>2017</li><li>Makkuva et al.</li><li>2020</li><li>CellOT (theirs)</li></ul></td>

           <td class="published">✓</td>
            <td><a href="https://github.com/bunnech/condot/tree/main" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="TODO">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/2405.00838">GWOT</a></td>
           <td>2025</td>

           <td><ul><li>Counterfactual Cell Maps</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Optimal Transport</li><li>Multi-modal</li></ul></td>

           <td><ul><li>-</li></ul></td>

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

           <td><ul><li>Counterfactual Cell Maps</li><li>Context Transfer</li><li>Seen Perturbation Prediction</li></ul></td>

           <td><ul><li>Flow Matching</li><li>Optimal Transport</li></ul></td>

           <td><ul><li>Conditional Flow Matching</li><li>Optimal Transport</li></ul></td>

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

           <td><ul><li>Counterfactual Cell Maps</li><li>Context Transfer</li><li>Seen Perturbation Prediction</li></ul></td>

           <td><ul><li>Flow Matching</li><li>Optimal Transport</li></ul></td>

           <td><ul><li>Conditional Flow Matching</li><li>Optimal Transport</li></ul></td>

           <td class="published">✓</td>
            <td><a href="https://github.com/kksniak/metric-flow-matching.git" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CFGen is a flow-based model for producing multi-modal scRNA-seq data. CFGen builds on CellFlow and explicitly models the discrete, over-dispersed nature of single-cell counts when generating synthetic data.">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/forum?id=3MnMGLctKb">CFGen</a></td>
           <td>2024</td>

           <td><ul><li>Counterfactual Cell Maps</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Optimal Transport</li><li>Multi-modal</li><li>Conditional Flow Matching</li></ul></td>

           <td><ul><li>CellFlow</li></ul></td>

           <td class="published">✓</td>
            <td><a href="https://github.com/theislab/CFGen" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CellFlow learns a vector field to predict time-dependent expression profiles under diverse conditions. The model encodes various covariates (perturbation, dosage, batch, etc.) , aggregates the embeddings via attention and deep sets, and uses a conditional flow matching framework to learn the underlying flow of the effect.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2025.04.11.648220v1.full.pdf">cellFlow</a></td>
           <td>2024</td>

           <td><ul><li>Counterfactual Cell Maps</li><li>Context Transfer</li><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>Conditional Flow Matching</li><li>Optimal Transport</li></ul></td>

           <td><ul><li>CellOT</li></ul></td>

           <td class="published">✗</td>
            <td>✗</td>
         </tr>
         <tr data-description="Waddington-OT models developmental processes as time‐varying probability distributions in gene expression space and infers temporal couplings by solving an entropy‐regularized, unbalanced optimal transport problem. Growth rate, estimated leveraging expression levels of genes associated with proliferation and apoptosis, is taken into consideration via unbalanced OT. Additionally, uses spectral clustering to obtain Gene Programmes, and subsequently associate those to predictive TFs.">
           <td class="details-control"></td>
           <td><a href="https://www.sciencedirect.com/science/article/pii/S009286741930039X?via%3Dihub">Waddington-OT</a></td>
           <td>2019</td>

           <td><ul><li>Counterfactual Cell Maps</li><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>Unbalanced OT</li><li>Entropy‐regularized Sinkhorn</li></ul></td>

           <td><ul><li>-</li></ul></td>

           <td class="published">✓</td>
            <td><a href="https://github.com/broadinstitute/wot" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Moscot is a broad and scalable framework that recasts various single-cell mapping tasks as optimal transport problems, supporting formulations that compare distributions in shared (Wasserstein-type OT), distinct (Gromov-Wasserstein OT), and partially-overlapping feature spaces (fused-Gromov–Wasserstein OT). Beyond Entropy-regularized sinkhorn (Cuturi et al., 2013), moscot provides a user-friendly API to more recent OT strategies, such as low-rank and sparse Monge maps.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41586-024-08453-2">moscot</a></td>
           <td>2025</td>

           <td><ul><li>Counterfactual Cell Maps</li></ul></td>

           <td><ul><li>Unbalanced OT</li><li>Entropy‐regularized Sinkhorn</li><li>Low-rank OT</li><li>Sparse Map OT</li></ul></td>

           <td><ul><li>Waddington-OT</li><li>NovoSpaRc</li><li>PASTE</li><li>OTT</li></ul></td>

           <td class="published">✓</td>
            <td><a href="https://github.com/theislab/moscot" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="scPRAM is a computational framework for predicting single-cell gene expression changes in response to perturbations. The method integrates three main components: a variational autoencoder (VAE), optimal transport, and an attention mechanism. The VAE encodes high-dimensional, sparse gene expression data into a latent space. Optimal transport is applied in this latent space to match unpaired cells before and after perturbation by finding an optimal coupling between their distributions. For each test cell, the attention mechanism computes a perturbation vector by comparing its latent representation (query) against those of matched training cells (keys and values). The predicted post-perturbation response is generated by adding the perturbation vector to the query and decoding it back to gene expression space using the VAE decoder.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/40/5/btae265/7646141">scPRAM</a></td>
           <td>2024</td>

           <td><ul><li>Context Transfer</li><li>Counterfactual Cell Maps</li></ul></td>

           <td><ul><li>VAE</li><li>OT</li><li>Attention</li></ul></td>

           <td><ul><li>-</li></ul></td>

           <td class="published">✓</td>
            <td><a href="https://github.com/jiang-q19/scPRAM" class="github-link">
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

