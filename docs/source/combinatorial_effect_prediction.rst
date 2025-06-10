Combinatorial Effect Prediction
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
         <tr data-description="MultiCPA extends CPA to predict combinatorial perturbation responses from CITE-seq data by integrating gene and protein modalities using either concatenation or a product of experts. It employs totalVI-inspired decoders and likelihoods to model modality-specific outputs.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2022.07.08.499049v1.abstract">MultiCPA</a></td>
           <td>2022</td>

           <td><ul><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>AE framework inspired by CPA</li><li>totalVI likelihood</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/theislab/multicpa" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="GEARS is uses graph neural networks to learn multidimensional embeddings for genes and their perturbations by respectively leveraging gene co-expression and GO-derived similarity graphs. It first derives refined gene embeddings through a co-expression-based GNN and separately processes perturbation embeddings via a GO graph to incorporate prior biological relationships, with the latter design enabling predictions for unSeen Perturbation Prediction. These embeddings are integrated by adding the aggregated perturbation signal to the gene representations and then decoded using gene-specific layers augmented by a cross-gene context module, ultimately reconstructing the post-perturbation transcriptomic profile. The model is trained end-to-end with a combined autofocus and direction-aware loss, and it can optionally quantify uncertainty through a Gaussian likelihood framework.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41587-023-01905-6#Abs1">GEARS</a></td>
           <td>2023</td>

           <td><ul><li>Combinatorial Effect Prediction</li><li>Unseen Perturbation Prediction</li></ul></td>

           <td><ul><li>GNNs for co-expression and GO relationships</li><li>Label embeddings</li><li>PK Representations</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/snap-stanford/GEARS" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="AttentionPert is a complex generative model that utilizes attention-based mechanisms to reconstruct perturbed cellular profiles from perturbation condition and precomputed Gene2Vec embeddings. It uses two encoders to capture global and local relationships between genes and perturbations (following GEARS). The PertWeight encoder models attention-based interactions between perturbations, while the PertLocal encoder identifies localized perturbation effects using an augmented GO graph.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/40/Supplement_1/i453/7700899">AttentionPert</a></td>
           <td>2024</td>

           <td><ul><li>Combinatorial Effect Prediction</li><li>Unseen Perturbation Prediction</li></ul></td>

           <td><ul><li>Transformer Model</li><li>GNN</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/BaiDing1234/AttentionPert" class="github-link">
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
         <tr data-description="SALT &amp; PEPER represents a straightforward two-step approach. The initial SALT model assumes additive effects of individual perturbations. Building on this foundation, PEPER leverages a neural network to learn a non-linear correction, effectively accounting for non-additive combinatorial effects. Notably, despite its simplicity, this approach has demonstrated impressive performance on standard extrapolation benchmarks.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/abs/2404.16907">SALT&PEPER</a></td>
           <td>2024</td>

           <td><ul><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>Additive Model</li><li>DNN</li></ul></td>


           <td class="published">✓</td>
            <td>✗</td>
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
         <tr data-description="CellFlow learns a vector field to predict time-dependent expression profiles under diverse conditions. The model encodes various covariates (perturbation, dosage, batch, etc.), aggregates the embeddings via attention and deep sets, and uses a conditional flow matching framework to learn the underlying flow of the effect.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2025.04.11.648220v1.full.pdf">cellFlow</a></td>
           <td>2024</td>

           <td><ul><li>Trace Cell Populations</li><li>Context Transfer</li><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>Conditional Flow Matching</li><li>Optimal Transport</li></ul></td>


           <td class="published">✗</td>
            <td>✗</td>
         </tr>
         <tr data-description="scGPT processes each cell as a sequence of gene tokens, expression-value tokens and condition tokens (e.g., batch, perturbation or modality), embedding each and summing before feeding them into stacked transformer blocks whose specialised, masked multi-head attention layers enable autoregressive prediction of masked gene expressions from non-sequential data. scGPT is pretrained using a masked gene expression-prediction objective that jointly optimizes cell and gene embeddings, and can be fine-tuned on smaller datasets with task-specific supervised losses. For gene regulatory network inference, scGPT derives k-nearest neighbor similarity graphs from learned gene embeddings and analyses attention maps to extract context-specific Gene Programmes and gene-gene interactions.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-024-02201-0">scGPT</a></td>
           <td>2024</td>

           <td><ul><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li><li>GRN Inference</li></ul></td>

           <td><ul><li>Foundational Gene expression embeddings (from >33M human cells)</li><li>Self-supervised masked expression prediction</li><li>Customised non-sequential (flash) attention</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/bowang-lab/scGPT" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="C2S-Scale is a family of large language models (LLMs) for single-cell RNA-seq analysis that extends the Cell2Sentence (C2S) framework by converting cell gene-expression profiles into ordered “cell sentences” for natural-language processing.Each C2S-Scale model is initialized from a publicly released Gemma-2 or Pythia checkpoint, i.e. leverages pre-existing language representations, and is then further pre-trained on a multimodal corpus of over a billion tokens. Each cell sentence is paired with the abstract (and, where available, additional free-text annotations) from the same study, allowing the model to learn matched transcriptomic and experimental context. ">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2025.04.14.648850v1">C2S-Scale</a></td>
           <td>2025</td>

           <td><ul><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>Family of LLMs with to 27B parameters</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/vandijklab/cell2sentence" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="LPM is a decoder-only deep neural network designed for large-scale integration and prediction across heterogeneous perturbation datasets. LPM encodes perturbation (P), readout (R), and context (C) as discrete variables, each with its own embedding space implemented via learnable look-up tables. These embeddings are concatenated to and used for inference">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/2503.23535">LPM</a></td>
           <td>2025</td>

           <td><ul><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li><li>GRN Inference</li></ul></td>

           <td><ul><li>DNN Decoder</li></ul></td>


           <td class="published">✗</td>
            <td>✗</td>
         </tr>
         <tr data-description="scGenePT combines CRISPR single‐cell RNA‐seq perturbation data with language‐based gene embeddings. It builds on a pretrained scGPT by adding gene‐level text embeddings from NCBI Gene/UniProt summaries or GO annotations, to the token, count, and perturbation embeddings of the model during fine-tuning on perturbational data.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.10.23.619972v1">scGenePT</a></td>
           <td>2025</td>

           <td><ul><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li><li>GRN Inference</li></ul></td>

           <td><ul><li>scGPT</li><li>ChatGPT prompts</li></ul></td>


           <td class="published">✗</td>
            <td>✗</td>
         </tr>
         <tr data-description="scFoundation uses an asymmetric transformer encoder–decoder: its embedding module converts each continuous gene expression scalar directly into a high-dimensional learnable vector without discretization; the encoder takes as input only nonzero and unmasked embeddings through vanilla transformer blocks to model gene–gene dependencies efficiently. The zero and masked gene embeddings, along with the encoder embeddings, are passed to the decoder, which uses Performer-style attention to reconstruct transcriptome-wide representations, specifically those of masked genes. Specifically, scFoundation is trained using a masked regression objective on both raw and downsampled count vectors, with two total-count tokens concatenated to inputs to account for sequencing depth variance. The decoder-derived gene context embeddings are then used as node features in GEARS for single-cell perturbation response prediction.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-024-02305-7">scFoundation</a></td>
           <td>2024</td>

           <td><ul><li>Nonlinear Gene Programmes</li><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

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

           <td><ul><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li><li>GRN Inference</li></ul></td>

           <td><ul><li>Foundational Gene expression embeddings (from >50M human cells)</li><li>Self-supervised masked regression with down-sampling</li><li>Sparse transformer encoder</li><li>Performer-style attention decoder</li><li>PK-informed</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/xCompass-AI/GeneCompass" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A VAE that disentangles control and pertubed cells into a latent space organized by a causal DAG. The encoder produces a Gaussian latent code z, while an intervention encoder transforms intervention one-hot encodings into two embeddings - a soft assignment vector that targets specific latent dimensions and a scalar capturing the intervention’s magnitude. Multiplying and adding these embeddings to z yields a modified latent vector that simulates a soft intervention, whereas zeroing them recovers the control condition. A causal layer then processes the latent vectors using an upper-triangular matrix G, which enforces an acyclic causal structure and propagates intervention effects among the latent factors. The decoder is applied twice - once to the modified latent code to generate virtual counterfactual outputs that reconstruct interventional outcomes, and once to the unmodified code to recover control samples. This dual decoding forces the model to disentangle intervention-specific effects from the intrinsic data distribution. The training objective combines reconstruction error to reconstruct control samples, a discrepancy loss (e.g., MMD) to align virtual counterfactuals with observed interventional data, KL divergence on the latent space, and an L1 penalty on G to enforce sparsity.">
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
         <tr data-description="Iterpert is an active learning framework for Perturb-seq experiments that uses GEARS to predict gene expression gene expression perturbation effects. The method iteratively retrains GEARS on new data and selects the next batch of perturbations using an enhanced kernel, which is constructed by fusing the GEARS-derived kernel with kernels from six prior information sources (additional Perturb-seq data, optical pooled screens, scRNA-seq atlases, protein structures, protein–protein interaction networks, and literature-derived features). Each prior source is mapped into a kernel matrix, normalized, and combined with the model kernel via a mean fusion operator. The fused kernel is then used with a greedy distance maximization rule to select perturbation batches under budget constraints (a limited set of experiments per round). ">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2023.12.12.571389v1.full.pdf">IterPert</a></td>
           <td>2024</td>

           <td><ul><li>Unseen Perturbation Prediction</li><li>Combinatorial Effect Prediction</li></ul></td>

           <td><ul><li>GEARS</li><li>Active Learning</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/Genentech/iterative-perturb-seq" class="github-link">
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

