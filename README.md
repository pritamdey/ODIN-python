# Outlier DetectIon for Networks (ODIN) Python Implementation

This repository contains our Python implementation of the algorithm for detecting outliers in structural connectivity binary adjacency matrices using ODIN (Outlier DetectIon for Networks), which we developed in our paper, [Outlier Detection for Multi-Network Data](https://arxiv.org/abs/2205.06398). 

The repository contains the directory data which contains a toy dataset for fitting with ODIN and the lobe and hemisphere locations of every ROI. It also contains the following python files built as a local python project.

- ODIN: This package contains the implementation. It has the following modules. The main functionality is bundled in
  the class ODIN. The method fit_and_detect_outliers of the ODIN class fits the model, calculates the influence measures and calculates the thresholds to classify outliers.
- main.py shows the step by step fitting of our model
- main.ipynb is a jupyter notebook with the same content as main.py
- simulation_runtime.py does the runtime simulations shown in our paper
- simulation_ownmodel_boxplots.py generates the boxplots in our paper
- simulation_ownmodel_sen_spe.py can be used to generate the sensitivity/specificity table shown for our model in the paper.

There is also an R implementation of this project [here](https://github.com/pritamdey/ODIN-r).
