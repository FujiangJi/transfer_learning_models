<div align="center">

<h1>Leveraging transfer learning and leaf spectroscopy for leaf trait prediction with broad spatial, species, and temporal applicability</h1>


[Fujiang Ji](https://fujiangji.github.io/) <sup>a</sup>, [Fa Li](https://scholar.google.com/citations?user=lOAXHLwAAAAJ&hl=en) <sup>a</sup>, [Hamid Dashti](https://hamiddashti.github.io) <sup>a</sup>, [Dalei Hao](https://scholar.google.com/citations?user=LapapmUAAAAJ&hl=en) <sup>b</sup>, [Philip A. Townsend](https://forestandwildlifeecology.wisc.edu/people/faculty-and-staff/philip-townsend/) <sup>a</sup>, [Ting Zheng](https://www.researchgate.net/profile/Ting-Zheng-12) <sup>a</sup>, [Hangkai You](https://scholar.google.com/citations?hl=en&user=zOLpygsAAAAJ) <sup>a</sup>, [Min Chen](https://globalchange.cals.wisc.edu/staff/chen-min/) <sup>a, c</sup>

<sup>a</sup> Department of Forest and Wildlife Ecology, University of Wisconsin- Madison, Madison, WI, USA;  
<sup>b</sup> Atmospheric, Climate, & Earth Sciences Division, PaciﬁcNorthwest National Laboratory, Richland, WA, USA;  
<sup>c</sup> Data Science Institute, University of Wisconsin- Madison, Madison, WI, USA.

</div>

<p align='center'>
  <a href="https://github.com/FujiangJi/transfer_learning_models"><img alt="Pape" src="https://img.shields.io/badge/TPAMI-Paper-6D4AFF?style=for-the-badge" /></a>
</p>

## Summary
* Accurate and reliable prediction of leaf traits is crucial for understanding plant adaptations to environmental variation, monitoring terrestrial ecosystems, and enhancing comprehension of functional diversity and ecosystem functioning.
* Various approaches (e.g., statistical, physical models) have been developed to estimate leaf traits through hyperspectral remote sensing and leaf spectroscopy. However, the absence of high-performing, transferable, and stable models across various domains of space, plant functional types (PFTs) and seasons hinder our ability to quantify and comprehend spatiotemporal variations in leaf traits. 
* This study proposes robust and highly transferable models for better predicting leaf traits with hyperspectral reflectance. Three datasets were assembled, pairing common leaf traits — chlorophyll (Chla+b, µg/cm<sup>2</sup>), carotenoids (Ccar, µg/cm<sup>2</sup>), leaf mass per area (LAM, g/m<sup>2</sup>), equivalent water thickness (EWT, g/m<sup>2</sup>) — with leaf spectra measurements collected across diverse geographic locations in the U.S. and Europe, PFTs, and seasons. 
* Through comparison with other state-of-the-art statistical models, including partial-least squares regression (PLSR) and Gaussian Process Regression (GPR), as well as pure physical models, we found that the proposed transfer learning models achieved better predictive performance and higher transferability. 

* **Objectives:**
  * **_(1) Do our proposed transfer learning models for predicting leaf traits have better performance than other state-of-the-art statistical models like PLSR and GPR, and the pure RTMs?_**
  * **_(2) Are the transfer learning models more transferable across different geographic locations, PFTs, and seasons than other models?_** 
  * **_(3) How the inconsistency and the quantity of real observations used for fine-tuning influence the performance of transfer learning models?_**

* **Conclusions:**
  * Numerous models have been developed to predict leaf traits based on leaf spectroscopy, each of which has its limitations. The absence of universally high-performing, transferable, and stable models across different domains hinder our ability to quantify and comprehend spatiotemporal variations in leaf traits and their responses to environmental changes and biodiversity in terrestrial ecosystems. 
  * In this study, we ensembled three types of datasets, with significant variability in leaf traits and leaf spectra across different locations, PFTs, and seasons. Our proposed transfer learning models, incorporating domain knowledge from RTMs and limited observational data, achieved better predictive performance compared to other statistical models and pure RTMs.
  * The transfer learning models exhibited higher transferability than statistical models. Our study underscores that transfer learning models can harness the advantages of both RTMs and statistical models and represent a promising approach for effectively predicting leaf traits. 

## Three datasets
<div align="center">
  <table>
    <colgroup>
      <col style="width: 10%;">
      <col style="width: 10%;">
      <col style="width: 20%;">
      <col style="width: 20%;">
      <col style="width: 40%;">
    </colgroup>
    <thead>
      <tr>
        <th>Leaf traits/ datasets</th>
        <th>Spectro-radiometers</th>
        <th>Spatial dataset</th>
        <th>PFT dataset</th>
        <th>Temporal dataset</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>Chla+b (µg/cm<sup>2</sup>)</b></td>
        <td>ASD FieldSpec 3/4/Pro</td>
        <td>&bull; <b>1600</b> samples. 8 sites (200 samples for each site).<br>&bull; Foreoptic type: integrating sphere, leaf clip, and contact probe.</td>
        <td>&bull; <b>3000</b> samples. DBF, CPR, GRA (1000 samples each).<br>&bull; Foreoptic type: integrating sphere, leaf clip, contact probe.</td>
        <td>&bull; <b>608</b> samples.<br>&bull; Early: 71;<br>&bull; Peak: 278;<br>&bull; Post-peak: 259<br>&bull; Foreoptic: integrating sphere.</td>
      </tr>
      <tr>
        <td><b>Ccar (µg/cm<sup>2</sup>)</b></td>
        <td>ASD FieldSpec 3/4/Pro</td>
        <td>&bull; <b>1000</b> samples. 5 sites.<br>&bull; Foreoptic: integrating sphere, leaf clip.</td>
        <td>&bull; <b>2100</b> samples. DBF, CPR, GRA (700 each).<br>&bull; Foreoptic: integrating sphere, leaf clip, contact probe.</td>
        <td>&bull; <b>634</b> samples.<br>&bull; Early: 71;<br>&bull; Peak: 278;<br>&bull; Post-peak: 285;<br>&bull; Foreoptic: integrating sphere.</td>
      </tr>
      <tr>
        <td><b>EWT (g/m<sup>2</sup>)</b></td>
        <td>SVC HR-1024i</td>
        <td>&bull; <b>400</b> samples. 4 sites.<br>&bull; Foreoptic: integrating sphere, leaf clip.</td>
        <td>&bull; <b>540</b> samples. DBF, GRA, CPR.<br>&bull; Foreoptic: integrating sphere, leaf clip.</td>
        <td>N/A</td>
      </tr>
      <tr>
        <td><b>LMA (g/m<sup>2</sup>)</b></td>
        <td>ASD FieldSpec 3/4/Pro</td>
        <td>&bull; <b>4800</b> samples. 12 sites.<br>&bull; Foreoptic: integrating sphere, leaf clip, contact probe.</td>
        <td>&bull; <b>1400</b> samples. 7 PFTs (200 each).<br>&bull; Foreoptic: integrating sphere, leaf clip, contact probe.</td>
        <td>&bull; <b>626</b> samples.<br>&bull; Early: 278;<br>&bull; Peak: 303;<br>&bull; Post-peak: 45;<br>&bull; Foreoptic: integrating sphere.</td>
      </tr>
    </tbody>
  </table>
</div>

<img src="figs/spatial sites distribution.png" title="" alt="" data-align="center">
<p align="center"><b>Fig.1.</b>Spatial distribution of leaf trait samples in the spatial dataset.</p>

## Modeling approaches
<img src="figs/workflows.png" title="" alt="" data-align="center">
<p align="center"><b>Fig.2.</b>Overall workflow for estimating leaf traits based on various models.</p>


## Requirements
* Python 3.7.13 and more in **[environment.yml](environment.yml)**

### Usage

* **Clone this repository and set environment**
  ```
  git clone https://github.com/FujiangJi/transfer_learning_models.git
  conda env create -f environment.yml
  conda activate py37
  ```
* **Leaf trait estimation**  
_Navigate to the directory **[src_code](src_code)** and execute the code in the following steps after updating the input/output paths:_
  * **_Construct Look-up Table:_** Based on PROSPECT and Leaf-SIP models to construct paired RTM synthetic reflectance and leaf traits.
    * Runing on the local PC:
      ```
      python 1_LUT_construction.py
      ```
    * Runing on the high-performance computing (HPC) cluster (Linux OS): go to directory **[scripts](scripts)**
      ```
      sbatch 1_LUT_bash.sh
      ```
  *  **_Pre-train DNN models:_** Pre-train the RTMs synthetic reflectance and leaf traits using DNN models.
      * Runing on the local PC:
        ```
        python 2_pretrain_DNN.py
        ```
      * Runing on the high-performance computing (HPC) cluster (Linux OS): go to directory **[scripts](scripts)**
        ```
        sbatch 2_pretrain_bash.sh
        ```
  *  **_Train statistical models (GPR, PLSR):_** Train the statistical models -- GPR, PLSR -- using different portion of observation data (10% - 80%).
      * Runing on the local PC:
        ```
        python 3_partial_obs_GPR_PLSR.py
        ```
      * Runing on the high-performance computing (HPC) cluster (Linux OS): go to directory **[scripts](scripts)**
        ```
        sbatch 3_partial_obs_GPR_PLSR_bash.sh
        ```
  *  **_Fine-tune pre-trained DNN model:_** Fine-tune the pre-trined DNN models using different portion of observation data (10% - 80%).
      * Runing on the local PC:
        ```
        python 4_fine_tune_DNN.py
        ```
      * Runing on the high-performance computing (HPC) cluster (Linux OS): go to directory **[scripts](scripts)**
        ```
        sbatch 4_fine_tune_DNN_bash.sh
        ```
  *  **_Leave one site out:_** Test the spatial transferability of statistical models and transfer learning models.
      * Runing on the local PC:
        ```
        python 5_GPR_PLSR_spatial_CV.py
        python 6_spatial_fine_tune.py
        ```
      * Runing on the high-performance computing (HPC) cluster (Linux OS): go to directory **[scripts](scripts)**
        ```
        sbatch 5_GPR_PLSR_spatial_CV_bash.sh
        sbatch 6_spatial_fine_tune_bash.sh
        ```
  *  **_Leave one PFT out:_** Test the PFTs transferability of statistical models and transfer learning models.
      * Runing on the local PC:
        ```
        python 7_GPR_PLSR_PFTs_CV.py
        python 8_PFTs_fine_tune.py
        ```
      * Runing on the high-performance computing (HPC) cluster (Linux OS): go to directory **[scripts](scripts)**
        ```
        sbatch 7_GPR_PLSR_PFTs_CV_bash.sh
        sbatch 8_PFTs_fine_tune_bash.sh
        ```
  *  **_Leave one season out:_** Test the temporal transferability of statistical models and transfer learning models.
      * Runing on the local PC:
        ```
        python 9_GPR_PLSR_temporal_CV.py
        python 10_temporal_fine_tune.py
        ```
      * Runing on the high-performance computing (HPC) cluster (Linux OS): go to directory **[scripts](scripts)**
        ```
        sbatch 9_GPR_PLSR_temporal_CV_bash.sh
        sbatch 10_temporal_fine_tune_bash.sh
        ```
  *  **_Pure radiative transfer models inversion:_** Estimating leaf traits using pure RTMs.
      * Runing on the local PC:
        ```
        python 11_pure_PROSPECT_estimation.py
        python 12_pure_LeafSIP_estimation.py
        ```
      * Runing on the high-performance computing (HPC) cluster (Linux OS): go to directory **[scripts](scripts)**
        ```
        sbatch 11_pure_PROSPECT_estimation_bash.sh
        sbatch 12_pure_LeafSIP_estimation_bash.sh
        ```
* **Description of files**
  * **[datasets](datasets)** directory: include the three type of datasets used in this study.
  * **[src_code](src_code)** directory:
    * [Models.py](src_code/Models.py): contains functions of statistical models and transfer learning models.
    * [prospect_d.py](src_code/prospect_d.py): source code of PROSPECT model.
    * [LeafSIP.py](src_code/LeafSIP.py): source code of Leaf-SIP model.
    * [spectral_library.py](src_code/spectral_library.py): absorption coefficients used in PROSPECT model.
    * [dataSpec_PDB.csv](src_code/dataSpec_PDB.csv): absorption coefficients used in Leaf-SIP model.
  * **[saved_ML_model](saved_ML_model)** directory: saved for the pre-trained DNN models, statistical models and the transfer learning models.
  
## Reference
In case you use our framework and code in your research, Please cite our paper:
* If you have any questions, please feel free to reach me at fujiang.ji@wisc.edu.
  ```
  Ji,F.; Li, F.; Dashti, H.; Hao, D.; Townsend, P. A.; Zheng, T.; You, H.; Chen, M. 
  Leveraging transfer learning and leaf spectroscopy for leaf trait prediction with broad 
  spatial, species, and temporal applicability. 2025. (Manuscript submitted, DOI forthcoming).
  ```

## Contact
```
fujiang.ji@wisc.edu
min.chen@wisc.edu
```
## Credits
* The datasets used in this study were subsets of the compiled dataset from previous study **_(Ji et al, New Phytologist, 2024)_**[**[<u>Link</u>]**](https://doi.org/10.1111/nph.19807), and the original data was from the EcoSIS Spectral Library, available at https://ecosis.org/.
* This study is supported by the National Aeronautics and Space Administration (NASA) through Remote Sensing Theory, Commercial SmallSat Data Scientific Analysis (CSDSA) and Terrestrial Ecology programs.
* We acknowledge high-performance computing support from the UW-Madison Center for High Throughput Computing (CHTC) in the Department of Computer Sciences. 
