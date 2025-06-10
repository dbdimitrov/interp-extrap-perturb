Contrastive Disentanglement
===========================


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
         <tr data-description="A modified version of PCA, where the covariance matrix (COV) is the difference between COV(case/perturbed) and αCOV(control/background). The hyperparameter α is used to balance having a high case variance and a low control variance. To provide some intuition, when α is 0, the model reduces to classic PCA on the case data. Optimal alphas (equal to k clusters) are identified using spectral clustering over a range of cPCA runs with different alphas, with selection based on the similarity of cPCA outputs.">
           <td class="details-control"></td>
           <td><a href="https://www.nature.com/articles/s41467-018-04608-8#Sec7">cPCA</a></td>
           <td>2018</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>Modified PCA</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/abidlabs/contrastive" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A non-negative matrix factorisation that decomposes gene expression matrices into common and condition-specific patterns. For each condition, the observed expression matrix is approximated as the sum of a common component - represented by a common feature matrix with condition-specific coefficient matrices - and a specific component unique to each condition, represented by its own feature matrix  and coefficients. The model uses an alternating approach to minimize the combined reconstruction error (squared Frobenius norm) across common and shared components.">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/nar/article/47/13/6606/5512984">CSMF</a></td>
           <td>2019</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>NMF</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://www.zhanglab-amss.org/homepage/software.html" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A family of contrastive latent variable models (cLVMs), where case data are modeled as the sum of background and salient latent embeddings, while control data are reconstructed solely from background embeddings: - cLVM with Gaussian likelihoods and priors - Sparse cLVM with horseshoe prior used to regularize the weights - Robust cLVM with a Student&#39;s t distribution - cLVM with automatic relevance determination to regularise the columns of the weight matrix - contrastive VAE, as a non-linear extension of the framework The shared concept across these models is that each model learns a shared set of latent variables for the background and target data, while salient latent variables are learnt solely for the target data.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/abs/1811.06094">cLVM</a></td>
           <td>2019</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>Factor Models</li><li>Contastive VAE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/kseverso/contrastive-LVM" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="VAE with two sets of latent variables (two encoders): salient and background, each learned using amortised inference from both case and control observations, respectively. The latent variables are concatenated and then decoded simultaneously via a shared decoder. During the generative process (decoding), the control observations are reconstructed solely from the background latent space, with salient latent variables being set to 0, while the case observations are generated from both sets of latent variables. Optionally, the two sets of latent variables can be further disentagled by minimizing their total correlation, in practice done by training a discriminator to distinguish real from permuted latent samples.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/1902.04601">cVAE</a></td>
           <td>2019</td>

           <td><ul><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>Contrastive VAE</li></ul></td>


           <td class="published">✗</td>
            <td><a href="https://github.com/abidlabs/contrastive_vae" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A sparse version of contrastive PCA that enhances interpretability in high-dimensional settings by integrating l1 regularization into an iterative procedure to estimate sparse loadings and principal components">
           <td class="details-control"></td>
           <td><a href="https://academic.oup.com/bioinformatics/article/36/11/3422/5807607">scPCA</a></td>
           <td>2020</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>Modified PCA</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/PhilBoileau/EHDBDscPCA" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A probabilistic model that builds on cPCA, additionally proposing a case-control-ratio-adjusted α as a more interpretable alternative to the same parameter in cPCA (see comment above).">
           <td class="details-control"></td>
           <td><a href="https://projecteuclid.org/journals/annals-of-applied-statistics/volume-18/issue-3/Probabilistic-contrastive-dimension-reduction-for-case-control-study-data/10.1214/24-AOAS1877.short">PCPCA</a></td>
           <td>2024</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>modified PCA</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/andrewcharlesjones/pcpca" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
         <tr data-description="A family of contrastive Poisson latent variable models (CPLVMs), based on a Gamma-Poisson hierarchical generative process: - CPLVM: The variational posterior is approximated using log-normal distributions, preserving non-negativity in the latent factors. - CGLVM: Extends CPLVM by allowing latent factors to take negative values, replacing Gamma priors with Gaussian priors and using a log-link function for the Poisson rates. Variational posteriors are modeled as multivariate Gaussians. The authors also propose a hypothesis testing framework, in which log-(ELBO)-Bayes is calculated between a Null model, omitting the salient latent space, and the full contrastive model. This framework is used to quantify global (across all genes) and joint expression changes in subsets of genes (akin to gene set enrichment analysis).">
           <td class="details-control"></td>
           <td><a href="https://projecteuclid.org/journals/annals-of-applied-statistics/volume-16/issue-3/Contrastive-latent-variable-modeling-with-application-to-case-control-sequencing/10.1214/21-AOAS1534.short">CPLVMs</a></td>
           <td>2022</td>

           <td><ul><li>Linear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>NB likelihood</li><li>Factor Models</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/andrewcharlesjones/cplvm" class="github-link">
                  <i class="fab fa-github" aria-hidden="true"></i>
                  <span class="sr-only">GitHub</span>
                </a></td>
         </tr>
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
         <tr data-description="A Contrastive VAE framework, similar to cVAE, which additionally incorporates a maximum mean discrepancy (MMD) loss to enforce salient latent variables in the control data to approach zero, while also using it to align the background latent variables between case and control conditions.">
           <td class="details-control"></td>
           <td><a href="https://arxiv.org/pdf/2202.10560">mmVAE</a></td>
           <td>2022</td>

           <td><ul><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>Contrastive VAE</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/suinleelab/MM-cVAE" class="github-link">
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
         <tr data-description="A VAE that disentangles disease (case) from healthy (control) cells by learning invariant background and salient space representations. The background and salient representations are summed to reconstruct the count data, with an (optional) interaction term capturing the interplay between cell type and disease. As done in contrastive methods, the salient representation for control cells is set to 0 during the generative (data reconstruction) process. The invariance of the background latent variables is enforced through two GAN-style neural networks: one encouraging the prediction of cell types from the background space, while the other penalises the prediction of disease labels, ensuring that disease-specific information is isolated in the salient space.">
           <td class="details-control"></td>
           <td><a href="https://openreview.net/pdf?id=fkoqMdTlEg">scDSA</a></td>
           <td>2023</td>

           <td><ul><li>Nonlinear Gene Programmes</li><li>Contrastive Disentanglement</li></ul></td>

           <td><ul><li>NB likelihood</li><li>Domain-Adversarial NNs</li><li>VAE</li><li>Addative Shift</li></ul></td>


           <td class="published">✓</td>
            <td>✗</td>
         </tr>
         <tr data-description="A VAE that combines the contrastiveVI/cVAE architecture with a classifier that learns the pairing of perturbation labels to cells. As in ContrastiveVI, unperturbed cells are drawn solely from background latent space, while cells classified as perturbed are reconstructed from both the background and salient sapces. Additionally, Hilbert-Schmidt Independence Criterion (HSIC) is used to disentagle the background and salient latent spaces.">
           <td class="details-control"></td>
           <td><a href="https://www.biorxiv.org/content/10.1101/2024.01.05.574421v1.full">SC-VAE</a></td>
           <td>2024</td>

           <td><ul><li>Contrastive Disentanglement</li><li>Perturbation Responsiveness</li></ul></td>

           <td><ul><li>VAE</li><li>NB likelihood</li></ul></td>


           <td class="published">✓</td>
            <td>✗</td>
         </tr>
         <tr data-description="An extension of ContrastiveVI that incorporates an auxiliary classifier to estimate the effects of perturbations, where the classifier operates on the salient variables and is sampled from a relaxed straight-through Bernoulli distribution. The output from the classifier also directly informs the salient latent space, indicating whether a cell expressing a gRNA successfully underwent a corresponding genetic perturbation. Additionally, Wasserstein distance is replaced by KL divergence, encouraging non-perturbed cells to map to the null region of the salient space. For datasets with a larger number of perturbations, the method also re-introduces and minimizes the Maximum Mean Discrepancy between the salient and background latent variables. This discourages the leakage of perturbation-induced information into the background latent variables, ensuring a clearer separation of perturbation effects.">
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
         <tr data-description="A non-negative matrix factorization method that decomposes single-cell gene expression data into common and condition-specific gene modulees. Each sample’s expression matrix is modeled as the sum of a shared component (W₂V) and condition-specific components (W₁Hⱼ), plus residual noise. The approach minimizes a loss function combining reconstruction error (Frobenius norm) with regularization terms that control module scale and inter-condition similarity.">
           <td class="details-control"></td>
           <td><a href="https://link.springer.com/article/10.1186/s13059-022-02649-3#Sec11">scINSIGHT</a></td>
           <td>2022</td>

           <td><ul><li>Contrastive Disentanglement</li><li>Linear Gene Programmes</li></ul></td>

           <td><ul><li>NMF</li></ul></td>


           <td class="published">✓</td>
            <td><a href="https://github.com/Vivianstats/scINSIGHT" class="github-link">
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

