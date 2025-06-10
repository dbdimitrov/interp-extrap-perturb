GRN Inference
=============


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
         <tr data-description="Geneformer is a context-aware transformer encoder comprising six layers of full dense self-attention over an input sequence of up to 2,048 genes, producing embeddings for genes and cells. Genes in each single-cell transcriptome are encoded as  rank value vectors - each gene’s expression is ranked within each cell. Pretraining uses a self-supervised masked learning objective (masking 15% of gene tokens and minimizing a prediction loss to recover their identities).">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41586-023-06139-9">Geneformer</a></td>
           <td>2023</td>

           <td><ul><li>GRN Inference</li></ul></td>

           <td><ul><li>Foundational Gene expression embeddings (from ~30M human cells)</li><li>Self-supervised masked regression</li><li>Standard transformer attention</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/jkobject/geneformer" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
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
         <tr data-description="scPRINT is implemented as a bidirectional transformer, focusing on scalable zero-shot applications to new datasets. During pre-training, it optimises a single composite loss that sums: (1) a denoising objective, which up-samples down-sampled transcript counts via a zero-inflated negative-binomial decoder; (2) a bottleneck reconstruction objective, where the model must regenerate full expression profiles from its compressed cell embedding; and (3) a hierarchical label-prediction objective that forces disentanglement of latent factors for cell type, disease, platform and other metadata. Each gene token is the sum of: a learned protein embedding for its gene ID; an MLP encoding of its log-normalized count; and a positional encoding of its genomic locus . Pre-training contexts consist of 2,200 randomly sampled expressed genes per cell. At inference, cell-specific gene networks are derived from the model’s multi-head attention maps by either averaging all heads or selecting a subset post hoc based on correlation with external priors (e.g., protein–protein interaction databases, ChIP-seq, perturbation-ground-truth networks).">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-025-58699-1">scPrint</a></td>
           <td>2025</td>

           <td><ul><li>Gene Programmes</li><li>GRN Inference</li><li>Multi-component Disentanglement</li></ul></td>

           <td><ul><li>Foundational Gene expression embeddings (from >50M human cells)</li><li>BERT-like Bidirectional transformers (with flashattention2)</li><li>Self-supervised masked regression</li><li>A classifier decoder</li><li>ZINB likelihood decoder</li><li>PK Representations</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/cantinilab/scPRINT" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="TODO">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41587-024-02182-7">LINGER</a></td>
           <td>2024</td>

           <td><ul><li>GRN Inference</li></ul></td>

           <td><ul><li>Multi-modal</li><li>Prior Knowledge Informed</li><li>Shapley values</li><li>DNN</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/Durenlab/LINGER" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="TODO">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41592-023-01938-4">SCENIC+</a></td>
           <td>2022</td>

           <td><ul><li>GRN Inference</li></ul></td>

           <td><ul><li>Multi-modal</li><li>Prior Knowledge Informed</li><li>Gradient Boosting</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/aertslab/scenicplus" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="TODO">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41586-022-05688-9">CellOracle</a></td>
           <td>2023</td>

           <td><ul><li>GRN Inference</li></ul></td>

           <td><ul><li>Multi-modal</li><li>Prior Knowledge Informed</li><li>Regularised (Linear) Regression</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/morris-lab/CellOracle" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Dictys integrates scRNA-seq and scATAC-seq data to infer gene regulatory networks (GRNs) and their changes across multiple conditions. By leveraging multiomic data, Dictys infers context-specific networks and dynamic GRNs using steady-state solutions of the Ornstein-Uhlenbeck process to model transcriptional kinetics and account for feedback loops. It reconstructs GRNs by detecting transcription factor (TF) binding sites and refining these networks with single-cell transcriptomic data, capturing regulatory shifts that reflect TF activity beyond expression levels.">
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

