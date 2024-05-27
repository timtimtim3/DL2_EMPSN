# Fast SE(n) Equivariant Simplicial Message Passing Networks

## Introduction
This repository is part of an ongoing project aimed at developing message passing network that leverages the expressivity of simplicial message passing for fast SE(n) equivariant convolutions. By incorporating simplicial complexes into graph representations, EMPSN (Eijkelboom et al., 2023) enhances the model's capability to capture a broader range of structures beyond mere edge endpoints, facilitating a more detailed understanding of the data topology. CSMPN (Liu et al., 2024) allows for efficiently sharing the parameters of message passing across different simplex orders. Lifting data to spherical harmonic representation and applying SE(n) equivariant convolution afterwards allows PONITA (Bekkers et al., 2023) to perform position and orientation equivariant learning very fast. Our aim is to combine simplicial message passing with spherical harmonic lifting in order to still leverage the extra information from simplices but also make the system fast.

## Code Organization
The main components of the repository that are essential for our experiments:
* `./`: job scripts for experiments, outputs.
* `src/`: all the main code.
  * `empsn/`: EMPSN code which is currently not used in the experiments and will maybe be removed when we finish the project.
  * `ponita/ponita/`: components that are currently being used. Directly in the folder - main_X.py wrappers for each potential dataset. For the current experiments `main_qm9.py` is used.
    * `csmpn`: CSMPN repo, from which the simplicial transform is used.
    * `geometry`: geometrical transformations for P(O)NITA.
    * `models`: classes for PONITA (uses information about both position and orientation) and PNITA (only position information).
    * `nn`: classes for hidden layers of the network and their components.
    * `transforms`: data transformation into spherical harmonic embeddings for P(O)NITA, but also code to lift data to simplicial complexes as defined in EMPSN (currently we use the variant from `csmpn` for this).

## Experiments
The experiments are currently done on QM9 dataset and on the point-cloud version of PONITA - PNITA. To run training and testing of the network on QM9, run `./src/ponita/main_qm9.py` or directly `./runPnitaSim.job` with appropriate `sbatch` settings. 

To run the first experiment, use `--num_ori 0` to select PNITA and `--simplicial` to use simplicial structures:

```
cd ./src/ponita/
python3 -u main_qm9.py --num_workers 36 --num_ori 0 --simplicial
```

For the second experiment, also use `--preserve_edges` to include all the original edges as 1-simplices:

```
cd ./src/ponita/
python3 -u main_qm9.py --num_workers 36 --num_ori 0 --simplicial --preserve_edges
```

For the third experiment, use `./src/ponita/main_qm9_debug.py` with `--initial_edges` to include all the original edges as 1-simplices and to exclude 2-simplices with isolated nodes:

```
cd ./src/ponita/
python3 -u main_qm9_debug.py --num_workers 36 --num_ori 0 --simplicial --initial_edges

## Requirements and Conda Environments
In order to run the code in this repository install the following conda environment:

```
conda create --yes --name empsn_ponita python=3.10 numpy scipy matplotlib
conda activate empsn_ponita
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install pyg==2.3.1 -c pyg -y
pip3 install wandb
pip3 install pytorch_lightning==1.8.6
pip3 install pyg_lib==0.3.1 torch_scatter==2.1.0 torch_sparse==0.6.17 torch_cluster==1.6.0 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
conda install gudhi networkx
```

## References
Bekkers, EJ., Vadgama, S., Hesselink, RD., van der Linden, PA. Romero, DW. (2023). Fast, Expressive SE (n) Equivariant Networks through Weight-Sharing in Position-Orientation Space. arXiv preprint arXiv:2310.02970.

Eijkelboom, F., Hesselink, R., & Bekkers, E. J. (2023, July). E $(n) $ Equivariant Message Passing Simplicial Networks. In International Conference on Machine Learning (pp. 9071-9081). PMLR.

Liu, C., Ruhe, D., Eijkelboom, F., & Forré, P. (2024). Clifford group equivariant simplicial message passing networks. arXiv preprint arXiv:2402.10011.

## License
Bekkers et al. (2023) repository has MIT license, so all code related to PONITA should be considered to be under the MIT license. However, the two other repositories do not specify the license explicitly which means all rights are reserved by the author. As Cong Liu is our TA for this project, we assume we may copy and re-use CSMPN code, however, we probably have to explicitly ask Floor Eijkelboom for permission before the final submission, and specify the license here. 
