Reproducible material for **Correlating seismic gathers with subsurface models with CLIP - Mohammad Taufik and Randy Harsuko**

# Project structure
This repository is organized as follows:

* :open_file_folder: **src**: python library containing routines for the `seismiclip` source codes;
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details);
* :open_file_folder: **scripts**: set of python scripts used to run multiple experiments;

## Notebooks
The following notebooks are provided:

- :orange_book: ``Example-1-clip-training.ipynb``: notebook performing training using the elastic OpenFWI dataset;
- :orange_book: ``Example-2-linear-probing.ipynb``: notebook performing analysis to the trained CLIP model using linear probing applications;


## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate seismiclip
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX A100 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.
