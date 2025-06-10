Context Transfer
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
         <tr data-description="CausCell integrates causal representation learning with diffusion-based generative modeling to generate counterfactual single-cell data. It disentangles observed and unobserved concepts using concept-specific adversarial discriminators and links the resulting latent representations through a structural causal model encoded as a directed acyclic graph.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/biorxiv/early/2024/12/17/2024.12.11.628077.full.pdf">CausCell</a></td>
           <td>2024</td>

           <td><ul><li>Multi-component Disentanglement</li><li>Causal Structure</li><li>Combinatorial Effect Prediction</li><li>Context Transfer</li><li>Seen Perturbation Prediction</li></ul></td>

           <td><ul><li>Diffusion</li><li>Auxilary Classifiers</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/bm2-lab/CausCell" class="github-link">
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
         <tr data-description="trVAE enhances the scGEN model by incorporating condition embeddings and leveraging maximum mean discrepancy regularization to manage distributions across binary conditions. By utilizing a conditional variational autoencoder, trVAE aims to create a compact and consistent representation of cross-condition distributions, enhancing out-of-distribution prediction accuracy. ">
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
         <tr data-description="Dr.VAE uses a Variational Autoencoder architecture to predict drug response from transcriptomic perturbation signatures. It models transcription change as a linear function within a low-dimensional latent space, defined by encoder and decoder neural networks. For paired expression samples from treated and control conditions, Dr.VAE accurately predicts treated expression.">
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
         <tr data-description="scGen is VAE that uses latent space vector arithmetics to predict single-cell perturbation responses. The method first encodes high-dimensional gene expression profiles into a latent space, where it computes a difference vector (delta) representing the change between perturbed and unperturbed conditions. At inference, this delta vector is linearly added to the latent representation of unperturbed cells, and the adjusted latent vector is then decoded back into the original gene expression space, thereby simulating the perturbed state. ">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-019-0494-8#Abs1">scGEN</a></td>
           <td>2019</td>

           <td><ul><li>Context Transfer</li></ul></td>

           <td><ul><li>VAE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/theislab/scgen" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CellBox models cellular responses to perturbations, by linking molecular and phenotypic outcomes through a unified nonlinear ODE-based model, aimed at simulating dynamic cellular behavior. The framework uses gradient descent with automatic differentiation to infer ODE network interaction parameters, facilitating exposure to novel perturbations and prediction of cell responses. ">
           <td class="details-control"></td>
           <td><a href="https://www.cell.com/cell-systems/pdf/S2405-4712(20)30464-6.pdf">CellBox</a></td>
           <td>2021</td>

           <td><ul><li>Context Transfer</li><li>Seen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>ODE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/sanderlab/CellBox" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Compositional Perturbation Autoencoder (CPA) models single-cell gene expression under perturbations and covariates by decomposing expression into additive latent embeddings: a basal state, perturbation effects, and covariate effects. To ensure that the basal embedding is disentangled from perturbations and covariates, CPA employs an adversarial training scheme: auxiliary classifiers are trained to predict perturbations and covariates from the basal embedding, while the encoder is updated using a combined loss that discourages the basal representation from encoding such information. Perturbation embeddings are modulated by neural networks applied to continuous covariates (e.g., dose or time), enabling modeling of dose-response and combinatorial effects. ">
           <td class="details-control"></td>
           <td><a href="https://www.embopress.org/doi/full/10.15252/msb.202211517">CPA</a></td>
           <td>2023</td>

           <td><ul><li>Context Transfer</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>VAE</li><li>DANN-based Adversary that attempts to eliminate treatment effects/ cellular context from latent representation</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/theislab/cpa" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A CPA extension that embeds prior knowlegde about the compound structure of drugs (SMILES representations), allowing it to extend CPA to unseen drug perturbations.">
           <td class="details-control"></td>
           <td><a href="https://proceedings.neurips.cc/paper_files/paper/2022/hash/aa933b5abc1be30baece1d230ec575a7-Abstract-Conference.html">ChemCPA</a></td>
           <td>2022</td>

           <td><ul><li>Unseen Perturbation Prediction</li><li>Context Transfer</li></ul></td>

           <td><ul><li>AE framework inspired by CPA</li><li>Chemical representation embeddings</li></ul></td>


           <td class="published">✓</td>
            <td><a href="github.com/theislab/chemCPA" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CODEX uses a Deep Neural Network to map cells from control to perturbed states, learning perturbation effects in respective perturbation-dependent latent spaces. These latent spaces can be arbitrarily combined to infer unseen combinatorial effects, allowing the model to predict the outcomes of complex treatment combinations. Additionally, CODEX can leverage prior information from Gene Ontologies to inform the effects of completely unSeen Perturbation Prediction.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/40/Supplement_1/i91/7700898">CODEX</a></td>
           <td>2024</td>

           <td><ul><li>Combinatorial Effect Prediction</li><li>Unseen Perturbation Prediction</li><li>Context Transfer</li></ul></td>

           <td><ul><li>DNN</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/sschrod/CODEX" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="PrePR-CT is a framework designed to predict transcriptional responses to chemical perturbations in unobserved cell types by utilizing cell-type-specific graphs encoded within Graph Attention Networks (GANs). The approach constructs cell graph priors using metacells which are randomly associated with perturbed cells to transform the problem into a regression task.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.07.24.604816v1.full.pdf">PrePR-CT</a></td>
           <td>2024</td>

           <td><ul><li>Context Transfer</li></ul></td>

           <td><ul><li>Graph attention</li><li>Regression</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/reem12345/Cell-Type-Specific-Graphs" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="PDGrapher builds on graph neural network (GNN) to predict therapeutic perturbations that can reverse disease phenotypes, focusing directly on identifying perturbation targets rather than modeling the perturbation effects. By embedding diseased cell states into gene regulatory networks or protein-protein interaction networks, PDGrapher learns latent representations to infer optimal perturbations that drive diseased states toward desired healthy outcomes. The method utilizes dual GNNs - a response prediction module and a perturbagen discovery module - both employing causal graphs as priors and adjusting edges to model interventions. ">
           <td class="details-control"></td>
           <td><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10802439/">PDGrapher</a></td>
           <td>2025</td>

           <td><ul><li>Combinatorial Effect Prediction</li><li>Unseen Perturbation Prediction</li><li>Context Transfer</li></ul></td>

           <td><ul><li>GNN</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/mims-harvard/PDGrapher" class="github-link">
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
         <tr data-description="GraphVCI employs two parallel inference branches to estimate latent variables from factual and counterfactual inputs. In the factual branch, observed gene expressions, treatments, and covariates are encoded via an MLP combined with a GCN/GAT module that integrates a gene regulatory network; its corresponding decoder then reconstructs the observed expression profile. The sparse gene regulatory network is generated using a prior-informed drop out mechanism, based on ATAC-Seq data.  A parallel branch processes counterfactual treatments to generate alternative expression profiles. Training minimizes three losses: an individual-specific reconstruction loss computed as the negative log likelihood (e.g., under a normal or negative binomial distribution) of the observed expressions; a covariate-specific loss implemented as an adversarial network using a binary cross-entropy loss on the counterfactual outputs; and a KL divergence loss that regularizes and aligns the latent space between the factual and counterfactual branches.">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/pdf?id=ICYasJBlZNs">graphVCI</a></td>
           <td>2023</td>

           <td><ul><li>Context Transfer</li></ul></td>

           <td><ul><li>Dual-branch variational bayes causal inference framework</li><li>PK Representations</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/yulun-rayn/graphVCI" class="github-link">
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
         <tr data-description="Squidiff integrates a diffusion model with a variational autoencoder (VAE) to modulating cellular states and conditions using latent variables. Squidiff can accurately capture and reproduce cellular states, and can be used to generate new single-cell gene expression data over time and in response to stimuli">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.11.16.623974v1">Squidiff</a></td>
           <td>2024</td>

           <td><ul><li>Combinatorial Effect Prediction</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Diffusion Model</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/siyuh/squidiff" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CellOT learns mappings between control and perturbed cell state distributions by solving a dual formulation of the optimal transport problem. The approach learns optimal transport maps as the gradient of a convex potential function, which is approximated using input convex neural networks - (briefly) a specific type of neural network with convex-preserving constraints, such as non-negative weights and a predefined set of activation functions (e.g. ReLU). Instead of relying on regularisation-based OT (e.g. Entropy-regularised Sinkhorn), it jointly optimizes dual potentials (a pair of functions) via a max–min loss.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-023-01969-x">CellOT</a></td>
           <td>2023</td>

           <td><ul><li>Trace Cell Populations</li><li>Perturbation Responsiveness</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Dual (min-max) Formulation OT</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/bunnech/cellot" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CondOT builds on CellOT to learn context-aware optimal transport maps by conditioning on an auxiliary variable. Instead of learning a fixed transport map, it learns a context-dependent transport map that adapts based on this auxiliary information. The OT map is modeled as the gradient of a convex potential using partially input convex neural networks, which ensures mathematical properties required for parametrised optimal transport. The auxiliary variables can be of different types: continuous (like dosage or spatial coordinates), categorical (like treatment groups, represented via one-hot encoding), or learned embeddings learned. Additionally, CondOT includes a separate neural module, a combinator network, for combinatorial predictions.">
           <td class="details-control"></td>
           <td><a href="https://proceedings.neurips.cc/paper_files/paper/2022/file/2d880acd7b31e25d45097455c8e8257f-Paper-Conference.pdf">CondOT</a></td>
           <td>2022</td>

           <td><ul><li>Trace Cell Populations</li><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Conditioned Dual (min-max) Formulation OT</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/bunnech/condot/tree/main" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="The paper extends entropic Gromov-Wasserstein Optimal Transport and Co-Optimal Transport to incorporate perturbation labels for aligning data across different modalities from large-scale perturbation screens. The core innovation involves constraining the learned cross-modality coupling matrix to be &#34;label-compatible&#34;, meaning that the transport plan is informed by the perturbation labels and is only allowed to match between cells that have received the same perturbation label, which is achieved by modifying the Sinkhorn algorithm. This label-compatible alignment is used to train a model to estimate cellular responses to perturbations when measurements are available in only one modality.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/2405.00838">GWOT</a></td>
           <td>2025</td>

           <td><ul><li>Trace Cell Populations</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Optimal Transport</li><li>Multi-modal</li></ul></td>


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

           <td><ul><li>Trace Cell Populations</li><li>Context Transfer</li><li>Seen Perturbation Prediction</li></ul></td>

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

           <td><ul><li>Trace Cell Populations</li><li>Context Transfer</li><li>Seen Perturbation Prediction</li></ul></td>

           <td><ul><li>Flow Matching</li><li>Optimal Transport</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/kksniak/metric-flow-matching.git" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="scDiffusion employs a Latent Diffusion Model for generating single-cell RNA sequencing data, using a three-part framework: a fine-tuned autoencoder for initial data transformation, a skip-connected multilayer perceptron denoising network, and a condition controller for cell-type-specific data generation. ">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/40/9/btae518/7738782">scDiffusion</a></td>
           <td>2024</td>

           <td><ul><li>Context Transfer</li></ul></td>

           <td><ul><li>Diffusion</li><li>VAE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/EperLuo/scDiffusion" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CFGen is a flow-based model for producing multi-modal scRNA-seq data. CFGen builds on CellFlow and explicitly models the discrete, over-dispersed nature of single-cell counts when generating synthetic data.">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/forum?id=3MnMGLctKb">CFGen</a></td>
           <td>2024</td>

           <td><ul><li>Trace Cell Populations</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Optimal Transport</li><li>Multi-modal</li><li>Conditional Flow Matching</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/theislab/CFGen" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="CellFlow learns a vector field to predict time-dependent expression profiles under diverse conditions. The model encodes various covariates (perturbation, dosage, batch, etc.), aggregates the embeddings via attention and deep sets, and uses a conditional flow matching framework to learn the underlying flow of the effect.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2025.04.11.648220v1.full.pdf">cellFlow</a></td>
           <td>2024</td>

           <td><ul><li>Trace Cell Populations</li><li>Context Transfer</li><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>Conditional Flow Matching</li><li>Optimal Transport</li></ul></td>


           <td class="published">✗</td>
            <td>✗</td>
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
         <tr data-description="SubCell is a set of self-supervised vision transformer (ViT) models trained on low-plex single-cell immunofluorescence images from the Human Protein Atlas to learn biologically meaningful representations of protein localisation and tissue morphology. The models are optimised using a multi-task objective that combines masked autoencoding for spatial reconstruction, a cell-specific contrastive loss to enforce consistency across augmented views of the same cell, and a protein-specific contrastive loss to align embeddings of cells stained for the same protein across different cell lines and experiments. An attention pooling module is used to priotise informative subcellular regions. The resulting models, particularly ViT-ProtS-Pool and MAE-CellS-ProtS-Pool, are shown to generalise across datasets, imaging modalities, cell types, and perturbations without fine-tuning.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.12.06.627299v1.abstract">SubCell</a></td>
           <td>2024</td>

           <td><ul><li>Context Transfer</li><li>Nonlinear Gene Programmes</li></ul></td>

           <td><ul><li>A collection of Vision Transformer Models</li><li>Contrastive loss</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/czi-ai/sub-cell-embed" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="VirTues is a multi-modal foundation model based on a vision transformer architecture, trained on multiplex spatial proteomics data from lung, breast, and melanoma tumors. It combines image representations with protein language model (PLM) embeddings of molecular markers and constructs hierarchical summary tokens at the cell, niche, and tissue levels. This PLM-based tokenisation enables the model to predict previously unseen markers. The architecture employs a sparse attention mechanism that factorises attention into spatial and marker components to manage the computational complexity of high-dimensional input. Training is performed using a masked autoencoding objective - i.e. it reconstructs missing subsets of spatial and protein data.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/abs/2501.06039">VirTues</a></td>
           <td>2025</td>

           <td><ul><li>Context Transfer</li></ul></td>

           <td><ul><li>ViT</li><li>Maked Autoencoder</li><li>Multi-modal</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/bunnelab/virtues" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="OmiCLIP is a dual-encoder foundation model that encodes H&amp;E histology patches with a Vision Transformer and “gene-sentence” representations of (10X Visium) spatial transcriptomics (ST) using a Transformer initialised on LAION-5B. It projects both modalities into a shared latent space using symmetric contrastive learning, which pulls matched histology-transcriptome pairs. The model is pretrained on paired H&amp;E images and ST data across diverse organs and disease states. These aligned embeddings underpin downstream tasks such as spatial registration of histology to transcriptomic spots, zero-shot tissue annotation, cell-type deconvolution, retrieval of matching transcriptomic profiles for novel histology inputs, and prediction of spatial gene-expression patterns directly from histology.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-025-02707-1">OmiCLIP</a></td>
           <td>2025</td>

           <td><ul><li>Context Transfer</li></ul></td>

           <td><ul><li>ViT</li><li>Contrastive Learning</li><li>Multi-modal</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/GuangyuWangLab2021/Loki" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Prophet represents each experiment as a set of three axes - cellular state (cell lines), treatments (perturbations), and phenotypic readouts - and projects diverse prior knowledge types (e.g., CCLE bulk RNA-seq for cell lines; chemical fingerprints or transcriptomic/genomic vectors for perturbations; learnable embeddings for readouts) into a shared token space. It is pre-trained on a set of diverse perturbation experiments covering readouts such as cell viability, compound IC50, Cell Painting morphology features, mRNA transcript abundance, and cell type proportions. A transformer-based encoder integrates these tokenised inputs, feeding a regression head that’s trained end-to-end to minimise mean squared error across all outcome types. The model is fine-tuned for assay-specific data applications.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.08.12.607533v2">Prophet</a></td>
           <td>2024</td>

           <td><ul><li>Unseen Perturbation Prediction</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Transformer</li><li>Multi-modal</li><li>Knowledge Informed</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/theislab/prophet" class="github-link">
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
            <td><a href="https://github.com/Bertinus/FLeCS" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="RENGE attempts to infer gene regulatory networks from time-series single-cell CRISPR knockout data. It models changes in gene expression following a knockout by propagating the effects through direct and higher-order (indirect) regulatory paths, where the gene network is represented as a matrix of regulatory strengths between gene pairs.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s42003-023-05594-4">RENGE</a></td>
           <td>2023</td>

           <td><ul><li>Context Transfer</li><li>GRN Inference</li><li>Causal Structure</li></ul></td>

           <td><ul><li>Regression model</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/masastat/RENGE" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="scPRAM is a computational framework for predicting single-cell gene expression changes in response to perturbations. The method integrates three main components: a variational autoencoder (VAE), optimal transport, and an attention mechanism. The VAE encodes the gene expression data into a latent space. Optimal transport is applied in this latent space to match unpaired cells before and after perturbation by finding an optimal coupling between their distributions. For each test cell, the attention mechanism computes a perturbation vector by comparing its latent representation (query) against those of matched training cells (keys and values). The predicted post-perturbation response is generated by adding the perturbation vector to the query and decoding it back to gene expression space using the VAE decoder.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/40/5/btae265/7646141">scPRAM</a></td>
           <td>2024</td>

           <td><ul><li>Context Transfer</li><li>Trace Cell Populations</li></ul></td>

           <td><ul><li>VAE</li><li>OT</li><li>Attention</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/jiang-q19/scPRAM" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="PRESCIENT models cellular differentiation as a stochastic diffusion process, where the drift term is parameterised as the negative gradient of a neural network-learned potential function. The model is trained using time-series single-cell RNA-seq data, and fits the potential function by minimizing the regularised Wasserstein distance between simulated and observed cell populations at each time point, explicitly incorporating cell proliferation by weighting cells according to their expected number of descendants. PRESCIENT can simulate differentiation trajectories for both observed and (in silico) perturbed cell states, enabling the prediction of cell fate outcomes under various genetic interventions.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-021-23518-w">Prescient</a></td>
           <td>2021</td>

           <td><ul><li>Trace Cell Populations</li><li>Seen Perturbation Prediction</li><li>Context Transfer</li></ul></td>

           <td><ul><li>Diffusion</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/gifford-lab/prescient" class="github-link">
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

