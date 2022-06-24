# Outlier DetectIon for Networks (ODIN) Python Implementation

This repository contains our Python implementation of the algorithm for detecting outliers in structural connectivity binary adjacency matrices using ODIN (Outlier DetectIon for Networks), which we developed in our paper, [Outlier Detection for Multi-Network Data](https://arxiv.org/abs/2205.06398). 

The repository contains the directory data which contains a toy dataset for fitting with ODIN and the lobe and hemisphere locations of every ROI. It also contains the following python files built as a local python project.

- ODIN: The main functionality of this implementation
  - ODIN.py :  all relevant functions for fitting the model and calculating the influence.
  - helper_functions.py : contains some tools to construct some of the matrices needed to fit and visualize the data.
  - simulation_generation_functions.py file is for simulating data from our model.
- simulation_runtime.py does the runtime simulations shown in our paper
- simulation_ownmodel_boxplots.py generates the boxplots in our paper
- simulation_ownmodel_sen_spe.py can be used to generate the sensitivity/specificity table shown for our model in the paper.
- main.py shows the step by step fitting of our model

There is also an R implementation of this project [here](https://github.com/ODIN-r).
