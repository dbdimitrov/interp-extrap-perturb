Seen Perturbation Prediction
============================


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
         <tr data-description="trVAE enhances the scGEN model by incorporating condition embeddings and leveraging maximum mean discrepancy (MMD) regularization to manage distributions across binary conditions. By utilizing a conditional variational autoencoder (CVAE), trVAE aims to create a compact and consistent representation of cross-condition distributions, enhancing out-of-distribution (OOD) prediction accuracy. ">
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
         <tr data-description="Dr.VAE uses a Variational Autoencoder (VAE) architecture to predict drug response from transcriptomic perturbation signatures. It models transcription change as a linear function within a low-dimensional latent space, defined by encoder and decoder neural networks. For paired expression samples from treated and control conditions, Dr.VAE accurately predicts treated expression.">
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
         <tr data-description="CellBox models cellular responses to perturbations, by linking molecular and phenotypic outcomes through a unified nonlinear ODE-based model, aimed at simulating dynamic cellular behavior. The framework uses gradient descent with automatic differentiation to infer ODE network interaction parameters, facilitating exposure to novel perturbations and prediction of cell responses. ">
           <td class="details-control"></td>
           <td><a href="https://www.cell.com/cell-systems/pdf/S2405-4712(20)30464-6.pdf">CellBox</a></td>
           <td>2021</td>

           <td><ul><li>Context Transfer</li><li>Seen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>-ODE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/sanderlab/CellBox" class="github-link">
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
         <tr data-description="LEMUR is a PCA based algorithm that defines condition dependent embedings to analyze differences in differentialy expressed genes across conditions. For each condition a separate embeding matrix is learned and reconstructed using a shared matrix. This is used to generate counterfactual estimates for each cell and condition, which is used to infer DE neighborhoods. ">
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
         <tr data-description="MMFM (Multi-Marginal Flow Matching) builds on Flow Matching to model cell trajectories across time and conditions. MMFM generalizes the Conditional Flow Matching framework to incorporate multiple time points using a spline-based conditional probability path. Moreover, it leverages ideas from classifier-free guidance to incorporate multiple conditions.">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/pdf?id=hwnObmOTrV">MMFM</a></td>
           <td>2024</td>

           <td><ul><li>Counterfactual Cell Maps</li><li>Context Transfer</li><li>Seen Perturbation Prediction</li></ul></td>

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

           <td><ul><li>Counterfactual Cell Maps</li><li>Context Transfer</li><li>Seen Perturbation Prediction</li></ul></td>

           <td><ul><li>Flow Matching</li><li>Optimal Transport</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/kksniak/metric-flow-matching.git" class="github-link">
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

