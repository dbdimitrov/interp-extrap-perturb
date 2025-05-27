Causal Structure
================


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
         <tr data-description="NOTEARS replaced traditional statistical DAG learning techniques for observational data with a continuous optimization problem, by reformulating the acyclicity constraint. This reduces the computational complexity and facilitated first small scale biological applications. ">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/abs/1803.01422">NOTEARS</a></td>
           <td>2018</td>

           <td><ul><li>Causal Structure</li></ul></td>

           <td><ul><li>Continuous optimization for acyclicity</li></ul></td>


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

           <td><ul><li>Continuous optimization for acyclicity</li><li>DNN</li></ul></td>


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

           <td><ul><li>Continuous optimization for acyclicity</li><li>GNN</li></ul></td>


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

           <td><ul><li>Graph interventions</li><li>Ornstein–Uhlenbeck process\n-Steady-State ODE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/PMBio/bicycle" class="github-link">
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
         <tr data-description="Dictys integrates scRNA-seq and scATAC-seq data to infer gene regulatory networks (GRNs) and their changes across multiple conditions. By leveraging multiomic data, Dictys infers context-specific networks and dynamic GRNs using steady-state solutions of the Ornstein-Uhlenbeck process to model transcriptional kinetics and account for feedback loops. It reconstructs undirected GRNs by detecting transcription factor (TF) binding sites and refining these networks with single-cell transcriptomic data, capturing regulatory shifts that reflect TF activity beyond expression levels.">
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
         <tr data-description="FLeCS models single-cell gene expression dynamics using coupled ordinary differential equations (ODEs) parameterized by a gene regulatory network. Cells are grouped into temporal bins—either via pseudotime inference or experimental timestamps—and aligned across time with optimal transport to form (pseudo)time series. To model interventions FLeCS replicates interventions in the learned graph.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/2503.20027">FLeCS</a></td>
           <td>2025</td>

           <td><ul><li>Context Transfer</li><li>GRN Inference</li><li>Causal Structure</li></ul></td>

           <td><ul><li>ODE</li><li>Optimal Transp</li></ul></td>


           <td class="published">✗</td>
            <td>✗</td>
         </tr>
         <tr data-description="RENGE attempts to infer gene regulatory networks (GRNs) from time-series single-cell CRISPR knockout data. It models changes in gene expression following a knockout by propagating the effects through direct and higher-order (indirect) regulatory paths, where the gene network is represented as a matrix of regulatory strengths between gene pairs.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s42003-023-05594-4">RENGE</a></td>
           <td>2023</td>

           <td><ul><li>Context Transfer</li><li>GRN Inference</li><li>Causal Structure</li></ul></td>

           <td><ul><li>Regression model</li></ul></td>


           <td class="published">✗</td>
            <td>✗</td>
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

