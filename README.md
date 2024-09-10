# Repository for PIRATE Paper Code 

## Purpose

This repository contains the code needed to reproduce key results from the manuscript "PIRATE: Plundering AlphaFold Predictions to Automate Protein Engineering." Additionally, the core functionality for PIRATE disorder prediction and directed evolution is included in the /PIRATE directory. 

## Installation

Set up your environment by installing Python 3.9 and the dependencies listed in requirements.txt. Please download the AFSM1, AFSM2, AFSM3, and PIRATE models from (google drive link). These should be placed in the /models directory.


## Running PIRATE

To run the PIRATE application, please navigate your working directory to the /PIRATE folder.

The streamlit app can be run using the command: 

```streamlit run pirate_app.py```

The user is presented with two modes of use:
- Basic sequence analysis: in this mode, you enter your sequence and the app will return a plot of predicted disorder (per residue).
- PIRATE directed evolution: In this mode, you enter your sequence, determine the maximum number of residues to alter, specify any residues that are not to be changed, specify the number of mutations to test at each site and the number of sites to test per round of evolution, and select a final scoring criteria. The output is a CSV containing the sequences, mutation information, and scoring data for each sequence from the Monte Carlo simulation. 

## License

This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 deed. You may copy and redistribute the material in any medium or format. This license does not permit commercial use of this product or distribution of this product if it has been modified. 