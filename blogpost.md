# TITLE (<span style="color:red;font-weight:bold;background-color:yellow">TODO</span>)

## 1. Introduction <!-- Vincent -->

Traditional approaches in Graph Neural Networks (GNNs), such as Message Passing Neural Networks (MPNNs), are restricted in their expressiveness as they only allow information flow between immediate neighbors. Recent developments have introduced higher-dimensional simplices (Eijkelboom et al., 2023) and geometric data integrations (Bekkers et al., 2023), increasing expressivity but at a significant computational cost. Our proposed model aims to synergize Eijkelboom et al. (2023)'s simplicial, E(n)-equivariant message passing framework with Bekkers et al. (2023)'s group convolutional methods to enhance computational efficiency without compromising expressivity.

The objective of our research is to construct an EMPSN that maintains high expressivity while significantly reducing computational demands. This approach is anticipated to mitigate the complexity-related challenges present in current models and enable faster, more scalable GNNs suitable for complex datasets. The significance of our work is its potential to improve the efficiency of GNNs, thus facilitating their broader application. We expect that our findings will not only demonstrate enhanced performance and reduced computational load compared to existing models but also provide insights into the integration of complex topological structures in neural networks.


### 1.1 Message Passing Neural Networks
MPNNs are a class of GNNs that operate by iteratively updating node features based on information from neighboring nodes (Gilmer et al., 2017). The message passing framework is defined by a series of functions that update node features based on the features of neighboring nodes. Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be a graph with nodes $\mathcal{V}$ and edges $\mathcal{E}$. Each node $v_i \in \mathcal{V}$ and edge $e_{ij} \in \mathcal{E}$ has associated feature vectors $\mathbf{f}_i$ and $\mathbf{a}_{ij}$, respectively. The message passing framework consists three steps. First, the messages $\mathbf{m}_{ij}$ from a central node $v_j$ to all neighbouring nodes $v_i$ are computed:

$$\mathbf{m}_{ij} = \phi_m(\mathbf{f}_i, \mathbf{f}_j, \mathbf{a}_{ij}),$$

where $\phi_m$ is a multi-layer perceptron (MLP). Second, the messages are aggregated to update the node features:

$$\mathbf{m}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij},$$

where $\mathcal{N}(i)$ is the set of neighboring nodes of $v_i$ and. Finally, the node features are updated:

$$\mathbf{f}_i' = \phi_f(\mathbf{f}_i, \mathbf{m}_i),$$

where $\phi_f$ is another MLP.

Although MPNNs have been successful in various applications, they are limited in their expressivity. Due to their local nature, MPNNs can only capture information from immediate neighbors, which limits their ability to capture complex topological structures in the data. In fact, it has been shown that MPNNs are isomorphic to the Weisfeiler-Lehman graph isomorphism test, and that MPNNs can be considered as a 1-dimensional Weisfeiler-Lehman (1-WL) graph isomorphism test (Xu et al., 2018). This means that MPNNs are limited in capturing higher-dimensional topological structures in the data. Figure 1 illustrates an example where two graphs with different topological structures would yield the same result by 1-WL.

<table align="center">
    <tr align="center">
        <td><img src="figures/gnn-limitations.png" width=350></td>
    </tr>
    <tr align="left">
    <td colspan=2><b>Figure 1.</b> In this example the color of the nodes determine the feature value, hence the graphs on the left and right would yield the same result by 1-WL, despite having different topological structures (Bodnar et al., 2021).</td>
    </tr>
</table>

An extension to MPNNs is the $E(n)$ Equivariant Graph Neural Network (EGNN), which are designed to be equivariant to the symmetries of the data (Satorras, 2021). Equivariant networks are neural networks that respect the symmetries of the data, meaning that if the input data is transformed in a certain way, the output of the network should transform in the same way. Formally, a function $f: X \rightarrow Y$ is equivariant to a group action $G$ if for all $x \in X$ and $g \in G$, we have $f(gx) = gf(x)$. In the context of GNNs, equivariance means that if the input graph is transformed by a permutation of the nodes, the output of the network should be transformed in the same way.

A MPNN is made equivariant by conditioning the message function on $E(n)$ invariant information, such as the relative position of the nodes. The message function is then defined as:

$$\mathbf{m}_{ij} = \phi_m(\mathbf{f}_i, \mathbf{f}_j, \text{Inv}(\mathbf{x}_i, \mathbf{x}_j), \mathbf{a}_{ij})$$
    
where $\text{Inv}(\mathbf{x}_i, \mathbf{x}_j)$ is a function that computes the invariant information between nodes $v_i$ and $v_j$.

Eventhough equivariant networks are more expressive than regular MPNNs, they still have limitations in capturing higher-dimensional topological structures in the data.

### 1.2 Message Passing Simplicial Networks
To address the limitations of MPNNs, Bodnar et al. (2021) introduced the concept of simplicial complexes in GNNs. A simplicial complex is a generalization of a graph that captures higher-dimensional topological structures in the data. A simplicial complex is defined as a set of simplices, where a simplex is a generalization of a node, edge, and triangle to higher dimensions. For example, a 2-simplex is a triangle, a 3-simplex is a tetrahedron, and so on. An abstract simplicial complex (ASC) is a set of simplices that satisfies the property that if a simplex is in the set, all its faces are also in the set. Figure 2 shows an example of a graph lifted to a simplicial complex.

<table align="center">
    <tr align="center">
        <td><img src="figures/simplicial-complex-example.png" width=350></td>
    </tr>
    <tr align="left">
    <td colspan=2><b>Figure 2.</b> Example of graph lifted to simplicial complex (Eijkelboom et al., 2023).</td>
    </tr>
</table>

Bodnar et al. (2021) introduced the concept of message passing simplicial networks (MPSNs), which is a type of MPNN in which four different types of ajdacencies between objects (i.e. simplices, denoted with $\sigma$) within an ASC are considered:
1. Boundary adjacencies $\mathcal{B}(\sigma)$: two simplices are boundary adjacent if they share a face.
2. Co-boundary adjacencies $\mathcal{C}(\sigma)$: two simplices are co-boundary adjacent if they are faces of the same simplex.
3. Lower adjacencies $\mathcal{N}_\downarrow(\sigma)$: two simplices are lower adjacent if one is a face of the other.
4. Upper adjacencies $\mathcal{N}_\uparrow(\sigma)$: two simplices are upper adjacent if one is a co-face of the other.

The message passing framework in MPSNs is similar to regular MPNNs, but the aggregation step is extended to consider all four types of adjacencies. For example, for a type of adjacency $\mathcal{}$, the messages are aggregated as follows:

$$\mathbf{m}_\mathcal{A}(\sigma) = \text{Agg}_{\tau\in\mathcal{A}(\sigma)}(\phi_{\mathcal{A}}(\mathbf{f}_\sigma,\mathbf{f}_\tau))$$

where $\phi_{\mathcal{A}}$ is an MLP and $\text{Agg}$ is an aggregation function. The node features are then updated based on the aggregated messages from all four types of adjacencies:

$$\mathbf{f}_\sigma' = \phi_f(\mathbf{f}_\sigma, \mathbf{m}_\mathcal{B}(\sigma), \mathbf{m}_\mathcal{C}(\sigma), \mathbf{m}_{\mathcal{N}_\downarrow}(\sigma), \mathbf{m}_{\mathcal{N}_\uparrow}(\sigma))$$

MPSNs have been shown to be more expressive than MPNNs, as they can capture higher-dimensional topological structures in the data. However, MPSNs are not equivariant to the symmetries of the data, which limits their performance on tasks where symmetries are important. To address this limitation, Eijkelboom et al. (2023) introduced the concept of $E(n)$ equivariant message passing simplicial networks (EMPSNs), which combine the expressiveness of MPSNs with the $E(n)$ equivariance of EGNNs.


<span style="color:red;font-weight:bold;background-color:yellow">I'm still working on this part, but pushed it so that @Luuk can start on section 1.3</span>




<!-- <table align="center">
    <tr align="center">
        <td><img src="figures/vietoris-rips-complex-example.png" width=350></td>
    </tr>
    <tr align="left">
    <td colspan=2><b>Figure 3.</b> Example of Vietoris Rips complex (Eijkelboom et al., 2023).</td>
    </tr>
</table>

<table align="center">
    <tr align="center">
        <td><img src="figures/different-invariants-upper-adjacent-communcation-example.png" width=200></td>
    </tr>
    <tr align="left">
    <td colspan=2><b>Figure 4.</b> Example of different invariants present in upper adjacent communication between 2-simplices (Eijkelboom et al., 2023).</td>
    </tr>
</table>

<table align="center">
    <tr align="center">
        <td><img src="figures/different-invariants-upper-adjacent-communcation-example.png" width=200></td>
    </tr>
    <tr align="left">
    <td colspan=2><b>Figure 5.</b> Change of invariants after position updates (Eijkelboom et al., 2023).</td>
    </tr>
</table> -->

### 1.3 SE(n) Equivariant Networks through Weight-Sharing in Position-Orientation Space <!-- Luuk -->
(Bekkers et al., 2023)
<span style="color:red;font-weight:bold;background-color:yellow">TODO @Luuk</span>

## 3. Our novel contribution (we should think of a better title/name for our contribution) <!-- Nin & Kristiyan -->
<span style="color:red;font-weight:bold;background-color:yellow">TODO</span> blablablabla we combine the idea of EMPSN with PONITA blablablabla

## 4. Results <!-- Vincent -->
<span style="color:red;font-weight:bold;background-color:yellow">TODO</span>

## 5. Conclusion <!-- Kristiyan -->
<span style="color:red;font-weight:bold;background-color:yellow">TODO</span>

## 6. Authors' Contributions
<span style="color:red;font-weight:bold;background-color:yellow">TODO</span>

## References
Bekkers, EJ., Vadgama, S., Hesselink, RD., van der Linden, PA. Romero, DW. 2023. Fast, Expressive SE (n) Equivariant Networks through Weight-Sharing in Position-Orientation Space, arXiv preprint arXiv:2310.02970.

Bodnar, C., Frasca, F., Wang, Y., Otter, N., Montufar, G. F., Lio, P., & Bronstein, M. (2021, July). Weisfeiler and lehman go topological: Message passing simplicial networks. In International Conference on Machine Learning (pp. 1026-1037). PMLR.

Eijkelboom, F., Hesselink, R. Bekkers, E. 2023. E(n) Equivariant Message Passing Simplicial Networks. E(n) equivariant message passing simplicial networks.

Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O., & Dahl, G.E. (2017). Neural Message Passing for Quantum Chemistry. International Conference on Machine Learning.

Satorras, V. G., Hoogeboom, E., & Welling, M. (2021, July). E (n) equivariant graph neural networks. In International conference on machine learning (pp. 9323-9332). PMLR.

Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2018). How powerful are graph neural networks?. arXiv preprint arXiv:1810.00826.