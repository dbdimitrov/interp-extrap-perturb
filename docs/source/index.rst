🧬 Interpretation, Extrapolation, and Perturbation of Single Cells
=====================================================================================

.. raw:: html

   <div align="center">

.. |stars-badge| image:: https://img.shields.io/github/stars/dbdimitrov/interp-extrap-perturb?style=social
   :target: https://github.com/dbdimitrov/interp-extrap-perturb/stargazers
   :alt: GitHub stars

.. |license-badge| image:: https://img.shields.io/github/license/dbdimitrov/interp-extrap-perturb
   :target: https://github.com/dbdimitrov/interp-extrap-perturb/blob/main/LICENSE
   :alt: License

.. |repo-badge| image:: https://img.shields.io/badge/GitHub-repo-black?logo=github&logoColor=white
   :target: https://github.com/dbdimitrov/interp-extrap-perturb
   :alt: Project repository

.. |methods-badge| image:: https://img.shields.io/badge/methods-139-blue.svg
   :target: methods.html
   :alt: Methods count

.. |prs-badge| image:: https://img.shields.io/badge/PRs-welcome-brightgreen.svg
   :target: contribute.html
   :alt: PRs welcome

|repo-badge| |stars-badge| |license-badge| |methods-badge| |prs-badge|

.. raw:: html

   </div>

---

📚 **A Living Catalogue of Single-Cell Perturbation Methods**

Single-cell genomics is moving beyond cell atlases toward models that **interpret and extrapolate cellular responses to perturbations**. 
This webpage lists technical details for more than **139 of these methods**. Moreover, as part of a perspective *(under review)*, 
it aims to help researchers pick the right tool and highlight open challenges for future method advances.

.. raw:: html

   <div class="highlight-info">
   <p><strong>🎯 Mission</strong>: Helping researchers navigate the rapidly evolving landscape of single-cell perturbation analysis</p>
   </div>

.. figure:: _static/overview.png
   :scale: 40%
   :align: center
   :alt: Goals of Emerging Causal and Mechanistic Single-Cell Methods

   🔬 **Goals of Emerging Causal and Mechanistic Single-Cell Methods**

---

🚀 **Key Features**
---------------------

.. raw:: html

   <div class="grid-container">
   <div class="grid-item">
   <h4>🔍 <strong>Comprehensive Coverage</strong></h4>
   <p> More than 100 curated methods from peer-reviewed papers and preprints</p>
   </div>
   
   <div class="grid-item">
   <h4>🏷️ <strong>Task Classification</strong></h4>
   <p>Methods organized by computational tasks and applications</p>
   </div>
   
   <div class="grid-item">
   <h4>📱 <strong>Interactive Interface</strong></h4>
   <p>Searchable, sortable tables with expandable descriptions</p>
   </div>
   
   <div class="grid-item">
   <h4>🔗 <strong>Direct Access</strong></h4>
   <p>Quick links to publications and code repositories</p>
   </div>
   
   <div class="grid-item">
   <h4>🤝 <strong>Community-Driven</strong></h4>
   <p>Easy contribution system for adding new methods</p>
   </div>
   
   <div class="grid-item">
   <h4>⚡ <strong>Always Current</strong></h4>
   <p>Automated updates and continuous integration</p>
   </div>
   </div>

   <style>
   .grid-container {
     display: grid;
     grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
     gap: 1.5rem;
     margin: 2rem 0;
   }
   .grid-item {
     padding: 1.5rem;
     border: 1px solid #ddd;
     border-radius: 8px;
     background: #f9f9f9;
   }
   .grid-item h4 {
     margin-top: 0;
     color: #8B0000;
   }
   </style>

---

📖 **Documentation Contents**
-------------------------------

.. toctree::
   :maxdepth: 2
   :caption: 📚 Explore the Catalog

   methods
   contribute

---

.. raw:: html

   <div align="center">
   <h3>🌟 <strong>Ready to Explore?</strong> 🌟</h3>
   <p>
   <a href="methods.html" class="btn-primary">🔍 Browse All Methods</a>
   &nbsp;&nbsp;
   <a href="contribute.html" class="btn-secondary">➕ Add Your Method</a>
   </p>
   
   <p><em>🤝 Join the community of researchers advancing single-cell perturbation analysis!</em></p>
   </div>

   <style>
   .btn-primary, .btn-secondary {
     display: inline-block;
     padding: 12px 24px;
     margin: 8px;
     border-radius: 6px;
     text-decoration: none;
     font-weight: bold;
     transition: all 0.3s ease;
   }
   .btn-primary {
     background: #8B0000;
     color: white;
   }
   .btn-secondary {
     background: #2E8B57;
     color: white;
   }
   .btn-primary:hover, .btn-secondary:hover {
     transform: translateY(-2px);
     box-shadow: 0 4px 8px rgba(0,0,0,0.2);
     color: white;
     text-decoration: none;
   }
   </style>