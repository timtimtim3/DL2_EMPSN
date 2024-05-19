# README

This repository combines Repo EMPSN https://github.com/flooreijkelboom/equivariant-simplicial-mp 
Repo PONITA https://github.com/ebekkers/ponita 
Repo CSMPN https://github.com/congliuUvA/Clifford-Group-Equivariant-Simplicial-Message-Passing-Networks 
in order to create Fast SE(n) Equivariant Simplicial Message Passing Network. 

## Code Organization
The main components of the repository that are essential for our experiments:
* `src/`: contains all the main code.
  * `empsn/`: EMPSN code which is currently not used in the experiments and will maybe be removed when we finish the project.
  * `ponita/ponita/`: components that are currently being used.
    * `csmpn`: CSMPN repo, from which the simplicial transform is used.
    * `csmpn`: CSMPN repo, from which the simplicial transform is used.


The experiments and their outputs could be found in the folder ./scripts/. Currently the experiments are done on QM9 dataset. 

## References
Bekkers, EJ., Vadgama, S., Hesselink, RD., van der Linden, PA. Romero, DW. 2023. Fast, Expressive SE (n) Equivariant Networks through Weight-Sharing in Position-Orientation Space, arXiv preprint arXiv:2310.02970.

Eijkelboom, F., Hesselink, R. Bekkers, E. 2023. E(n) Equivariant Message Passing Simplicial Networks. E(n) equivariant message passing simplicial networks.
