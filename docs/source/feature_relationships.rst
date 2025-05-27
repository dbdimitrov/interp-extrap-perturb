Feature Relationships
=====================


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
         <tr data-description="Celcomen (CCE) disentangles intra- and inter-cellular gene regulation in spatial transcriptomics data by processing gene expression through two parallel interaction functions. One function uses a graph convolution layer (k-hop GNN) to learn a gene-gene interaction matrix that captures cross-cell signaling, while the other applies a linear layer to model regulation within individual cells. During training, Celcomen combines a normalization term—computed via a mean field approximation that decomposes the overall likelihood into a mean contribution and an interaction contribution - with a similarity measure that directly compares each cell’s predicted gene expression (obtained via message passing) to its actual expression, thereby driving the model to adjust its interaction matrices so that the predictions closely match the observed data. Simcomen (SCE) then leverages these fixed, learned matrices to simulate spatial counterfactuals (e.g., gene knockouts) for in-silico experiments.">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/pdf?id=Tqdsruwyac">Celcomen</a></td>
           <td>2025</td>

           <td><ul><li>Unsupervised Disentanglement</li><li>Feature Relationships</li></ul></td>

           <td><ul><li>K-hop Convolution</li><li>Mean field estimation</li><li>Spatially-informed</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/Teichlab/celcomen" class="github-link">
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
         <tr data-description="MISTy extracts intra- and intercellular relationships from spatial omics data by learning multivariate interactions through a multi-view approach, where each view represents a collection of variables (e.g., a modality or an aggragation of a spatial niche). It jointly models spatial and functional aspects of the data, supporting any number of views with arbitrary numbers of variables. Target variables (intrinsic view) are predicted using random forests (by default), either via leave-feature-one-out within the intrinsic view or using the remaining (extrinsic) views.">
           <td class="details-control"></td>
           <td><a href="https://link.springer.com/article/10.1186/s13059-022-02663-5">MISTy</a></td>
           <td>2022</td>

           <td><ul><li>Feature Relationships</li></ul></td>

           <td><ul><li>Spatial Niches</li><li>Random Forrest (or other regression models)</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://saezlab.github.io/mistyR/" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="SpaCeNet aims to untangle the complex relationships between molecular interactions within and between cells by analyzing spatially resolved single-cell data. To achieve this, SpaCeNet leverages an adaptation of probabilistic graphical models (PGMs) to enable spatially resolved conditional independence testing. This approach allows for the identification of direct and indirect dependencies, as well as the removal of spurious gene association patterns. Additionally, SpaCeNet incorporates explicit cell-cell distance information to differentiate between short- and long-range interactions, thereby distinguishing between baseline cellular variability and interactions influenced by a cell&#39;s microenvironment.">
           <td class="details-control"></td>
           <td><a href="https://genome.cshlp.org/content/34/9/1371">SpaCeNet</a></td>
           <td>2024</td>

           <td><ul><li>Feature Relationships</li></ul></td>

           <td><ul><li>Generalised Gaussian Graphical Model</li><li>Spatially-informed</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/sschrod/SpaCeNet" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="Kasumi extends MISTy by focusing on identifying localized relationship patterns that are persistent across tissue samples. Instead of modeling global relationships, it uses a sliding-window approach to learn representations of local tissue patches (neighborhoods), characterized by multivariate, potentially non-linear relationships across views. These window-specific relationship signatures are clustered (using graph-based community detection) into spatial patterns, which are retained based on a persistence criterion—i.e., being consistently observed across multiple samples. This enables Kasumi to represent each sample as a distribution over interpretable, shared local patterns, facilitating tasks like patient stratification while maintaining model explainability.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-025-59448-0">Kasumi</a></td>
           <td>2025</td>

           <td><ul><li>Feature Relationships</li></ul></td>

           <td><ul><li>Spatially-informed</li><li>Random Forrest (or other regression models)</li><li>Convolution Operations</li></ul></td>


           <td class="published">✓</td>
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

