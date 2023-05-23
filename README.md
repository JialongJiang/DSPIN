# DSPIN
Code and examples of the D-SPIN framework for the preprint "D-SPIN constructs gene regulatory network models from multiplexed scRNA-seq data revealing organizing principles of cellular perturbation response" ([bioRxiv](https://www.biorxiv.org/content/10.1101/2023.04.19.537364))

![alternativetext](/figure/readme/Figure1_20230309_Inna.png)

## Demos

Two demos of D-SPIN are available on Google Colab. 

The first demo reconstructs the regulatory network of simulated hematopoietic stem cell (HSC) differentiation network with perturbations using the BEELINE framework (Pratapa, Aditya, et al. Nature methods, 2020). 

[Demo1](https://colab.research.google.com/drive/1YdvjNiCkyGx-azXzXz7gqjGGE9RXrDbL?usp=sharing)

The second demo reconstructs regulatory network and response vector in a single-cell dataset collected in the ThomsonLab.In the dataset, human peripheral blood mononuclear cells (PBMCs) were treated with various signaling molecules with different dosages. 

[Demo2](https://colab.research.google.com/drive/1zrWFZWtaHQAzG88jgtovCPzt3wiXdlwf?usp=sharing)

## General workflow of D-SPIN

For detailed decription of the framework and hyperparameter choice in the model, please refer to the preprint.

D-SPIN takes single-cell sequencing data of multiple perturbation conditions. In the second demo, PBMCs are treated with different signaling molecules such as CD3 antibody, LPS, IL1B, and TGFB1

![alternativetext](/figure/thomsonlab_signaling/example_conditions.png)

D-SPIN identifies a set of gene programs that coexpress in the data, and represent each cell as a combination of gene program expression states. 

![alternativetext](/figure/thomsonlab_signaling/gene_program_example.png)

D-SPIN uses cross-correlation and mean of each perturbation condition to inferred a unified regulatory network and the response vector of each perturbation condition. The inference can be parallelized across perturbation conditions. The inference code is in Matlab using "parfor", while for demo purpose Python code (without parallelization) is provided.

The inferred regulatory network and perturbations can be jointly analyzed to reveal how perturbations act in the context of the regulatory network.

![alternativetext](/figure/thomsonlab_signaling/regulatory_network_modules.png)

![alternativetext](/figure/thomsonlab_signaling/joint_network_perturbation.png)

## References

1. Pratapa, Aditya, et al. "Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data." Nature methods 17.2 (2020): 147-154.