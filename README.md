# Repository for PIRATE Paper Code 

![pirate_image](images/pirate.png)

## Purpose

This repository contains the code needed to reproduce key results from the manuscript "PIRATE: Plundering AlphaFold Predictions to Automate Protein Engineering." Additionally, the core functionality for PIRATE disorder prediction and directed evolution is included in the /PIRATE directory. 

## Installation

Set up your environment by creating a Python 3.9 environment. We recommend installing the dependencies listed in requirements.txt with pip:

```pip install -r requirements.txt```

Please download the AFSM1, AFSM2, AFSM3, and PIRATE models from (google drive link). These should be placed in the /models directory.

## Downloading DR-BERT 

For reproducing Figures 4 and 5, you'll need a local copy of the DR-BERT checkpoint file. Please follow the instructions in the GitHub repository [here.](https://github.com/maslov-group/DR-BERT)

## Reproducing Figures 3-6

You can run the code needed to reproduce these figures by executing the Jupyter notebooks located in the Figure 3, Figure 4, and Figure 5 directories. Figure 6 results can be reproduced by navigating the working directory to the Figure 6 folder and running the Python script using:

```python fig6_pde.py```

## Running PIRATE

To run the PIRATE application, please navigate your working directory to the /PIRATE folder.

The streamlit app can be run using the command: 

```streamlit run pirate_app.py```

The user is presented with two modes of use:
- Basic sequence analysis: in this mode, you enter your sequence and the app will return a plot of predicted disorder (per residue).
- PIRATE directed evolution: In this mode, you enter your sequence, determine the maximum number of residues to alter, specify any residues that are not to be changed, specify the number of mutations to test at each site and the number of sites to test per round of evolution, and select a final scoring criteria. The output is a CSV containing the sequences, mutation information, and scoring data for each sequence from the Monte Carlo simulation. 

## License [![License: CC-BY-NC-ND-4.0](https://img.shields.io/badge/license-CC_BY_NC_ND_4.0-green)](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en)

This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 deed. You may copy and redistribute the material in any medium or format. This license does not permit commercial use of this product or distribution of this product if it has been modified. 

For inquiries regarding commercial use, please email info@evotec.com for more information.