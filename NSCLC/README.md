# ProteoForge Application on the NSCLC Data by Tomin et al. 2025

This folder contains the data and the results of the ProteoForge application on the data from the publication by Tomin et al. 2025, titled **Increased antioxidative defense and reduced advanced glycation end-product formation by metabolic adaptation in non-small-cell-lung-cancer patients** [manuscript](https://www.nature.com/articles/s41467-025-60326-y). The input data is published part of manuscript and hosted on PRIDE with the following codes [PXD052340](https://www.ebi.ac.uk/pride/archive/projects/PXD052340), [PXD062503](https://www.ebi.ac.uk/pride/archive/projects/PXD062503) and [PXD061610](https://www.ebi.ac.uk/pride/archive/projects/PXD061610).

> - `./data/20230316_HumanCr_20806.fasta` (The FASTA file used in the search and the analysis)

There have conducted various analyses with abundance of datasets, which I will be picking some of them to demonstrate the ProteoForge application. The analyses are as follows:

## 1. The Effects of Hypoxia on lung Cancer Cells

- The data is from PRIDE with [PXD062503](https://www.ebi.ac.uk/pride/archive/projects/PXD062503) code.
- The following is the description of the data take from the PRIDE project page:
  > 300,000 of H358 cells were seeded in 6-well plates in quadruplicates per plate and placed either at 1 % or 21 % oxygen for either 48 or 72h, respectively. Upon the indicated time, cells were harvested in 500 µl of harvesting solution (80 % methanol in 50 mM ammonium acetate, supplemented with 2.5 mM N-ethylmaleimide (NEM) and heavy glutathione internal standard), sonicated for 5 s at 10 % amplitude and processed by for one-pot redox metabolite and protein thiol analysis approach. This project is in vitro validation of findings connected to the PRIDE project PXD052340.
- The zip file (`DIANNoutput.zip`) from the PRIDE project page is downloaded and extracted to the `data/input/hypoxia/` folder. Only the relevant files are kept in the folder:
    - `report.pdf` (The report of the analysis DIANN)
    - `report.parquet` (The report of the analysis DIANN in parquet format (precursors))
    - `report.gg_matrix` (The gene group matrix)
    - `report.pg_matrix` (The protein group matrix)
    - `report.pr_matrix` (The protein report)
    - `report.protein_description` (The protein description file, with sequence and other information)
    - `report.stats` (The sample statistics file)
- The design of the experiment is as follows:
    - 2 conditions: 1% and 21% oxygen
    - 2 time points: 48h and 72h
    - 4 replicates per condition and time point

- Proposed way to utilize the ProteoForge to find potential differential Proteoforms (dPFs):
    - Use the `report.parquet` file as the input for the ProteoForge application.
    - Set the conditions and time points as the groups in the ProteoForge application.
    - Run the analysis to find potential dPFs between the conditions and time points.
