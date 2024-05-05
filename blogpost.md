# TITLE (<span style="color:red;font-weight:bold;background-color:yellow">TODO</span>)

## 1. Introduction <!-- Vincent -->

Traditional approaches in Graph Neural Networks (GNNs), such as Message Passing Neural Networks (MPNNs), are restricted in their expressiveness as they only allow information flow between immediate neighbors. Recent developments have introduced higher-dimensional simplices (Eijkelboom et al., 2023) and geometric data integrations (Bekkers et al., 2023), increasing expressivity but at a significant computational cost. Our proposed model aims to synergize Eijkelboom et al. (2023)'s simplicial, E(n)-equivariant message passing framework with Bekkers et al. (2023)'s group convolutional methods to enhance computational efficiency without compromising expressivity.

The objective of our research is to construct an EMPSN that maintains high expressivity while significantly reducing computational demands. This approach is anticipated to mitigate the complexity-related challenges present in current models and enable faster, more scalable GNNs suitable for complex datasets. The significance of our work is its potential to improve the efficiency of GNNs, thus facilitating their broader application. We expect that our findings will not only demonstrate enhanced performance and reduced computational load compared to existing models but also provide insights into the integration of complex topological structures in neural networks.


### 1.1 Message Passing Neural Networks
MPNNs are a class of GNNs that operate by iteratively updating node features based on information from neighboring nodes (Gilmer et al., 2017). The message passing framework is defined by a series of functions that update node features based on the features of neighboring nodes. Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be a graph with nodes $\mathcal{V}$ and edges $\mathcal{E}$. Each node $`v_i \in \mathcal{V}`$ and edge $`e_{ij} \in \mathcal{E}`$ has associated feature vectors $`\mathbf{f}_i`$ and $`\mathbf{a}_{ij}`$, respectively. The message passing framework consists three steps. First, the messages $`\mathbf{m}_{ij}`$ from a central node $`v_j`$ to all neighbouring nodes $`v_i`$ are computed:
```math
\mathbf{m}_{ij} = \phi_m(\mathbf{f}_i, \mathbf{f}_j, \mathbf{a}_{ij})
```

where $`\phi_m`$ is a multi-layer perceptron (MLP). Second, the messages are aggregated to update the node features:
```math
\mathbf{m}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}
```

where $\mathcal{N}(i)$ is the set of neighboring nodes of $`v_i`$ and. Finally, the node features are updated:
```math
\mathbf{f}_i' = \phi_f(\mathbf{f}_i, \mathbf{m}_i)
```

where $`\phi_f`$ is another MLP.

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
```math
\mathbf{m}_{ij} = \phi_m(\mathbf{f}_i, \mathbf{f}_j, \text{Inv}(\mathbf{x}_i, \mathbf{x}_j), \mathbf{a}_{ij})
```
    
where $`\text{Inv}(\mathbf{x}_i, \mathbf{x}_j)`$ is a function that computes the invariant information between nodes $`v_i`$ and $`v_j`$.

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
3. Lower adjacencies $`\mathcal{N}_\downarrow(\sigma)`$: two simplices are lower adjacent if one is a face of the other.
4. Upper adjacencies $`\mathcal{N}_\uparrow(\sigma)`$: two simplices are upper adjacent if one is a co-face of the other.

The message passing framework in MPSNs is similar to regular MPNNs, but the aggregation step is extended to consider all four types of adjacencies. For example, for a type of adjacency $\mathcal{A}$, the messages are aggregated as follows:
```math
\mathbf{m}_\mathcal{A}(\sigma) = \text{Agg}_{\tau\in\mathcal{A}(\sigma)}(\phi_{\mathcal{A}}(\mathbf{f}_\sigma,\mathbf{f}_\tau))
```

where $`\phi_{\mathcal{A}}`$ is an MLP and $\text{Agg}$ is an aggregation function. The node features are then updated based on the aggregated messages from all four types of adjacencies:

```math
\mathbf{f}_\sigma' = \phi_f(\mathbf{f}_\sigma, \mathbf{m}_\mathcal{B}(\sigma), \mathbf{m}_\mathcal{C}(\sigma), \mathbf{m}_{\mathcal{N}_\downarrow}(\sigma), \mathbf{m}_{\mathcal{N}_\uparrow}(\sigma))
```

MPSNs have been shown to be more expressive than MPNNs, as they can capture higher-dimensional topological structures in the data. However, MPSNs are not equivariant to the symmetries of the data, which limits their performance on tasks where symmetries are important. To address this limitation, Eijkelboom et al. (2023) introduced the concept of $E(n)$ equivariant message passing simplicial networks (EMPSNs), which combine the expressiveness of MPSNs with the $E(n)$ equivariance of EGNNs by first lifting the graph to a simplicial complex and then conditioning the message function on $E(n)$ invariant geometric information.

Lifting a graph to a simplicial complex can be done either by a graph lift or constructing a Vietoris-Rips complex. A graph lift is a simplicial complex where each node in the graph is a 0-simplex, each edge is a 1-simplex, each triangle is a 2-simplex, and so on. However, this approach can lead to a simplicial complex that is too dense, which can be computationally expensive. To address this issue, a Vietoris-Rips complex can be constructed. A Vietoris-Rips complex is a simplicial complex that is constructed by connecting nodes in the graph that are within a certain distance of each other. Figure 3 shows an example of a Vietoris-Rips complex.

<table align="center">
    <tr align="center">
        <td><img src="figures/vietoris-rips-complex-example.png" width=350></td>
    </tr>
    <tr align="left">
    <td colspan=2><b>Figure 3.</b> Example of Vietoris Rips complex (Eijkelboom et al., 2023).</td>
    </tr>
</table>

Three types of geometric invariants are considered in EMPSNs: volumes, angles, and distances. Let $\mathcal{K}$ be a simplicial complex embedded in $\mathbb{R}^n$ and $`\sigma=\{v_0, \cdot\cdot\cdot ,v_n\}`$ and $\tau$ be two simplices in $\mathcal{K}$. The volume and angle invariants are defined as follows:
```math
\text{Vol}(\sigma) = \frac{1}{n!}\left|\det\left(v_1, \cdot\cdot\cdot, v_n\right)\right|
```
```math
\text{Ang}(\sigma, \tau) = \cos^{-1}\left(\frac{|\mathbf{n}_\sigma\cdot\mathbf{n}_\tau|}{|\mathbf{n}_\sigma||\mathbf{n}_\tau|}\right)
```

where $`\mathbf{n}_\sigma`$ and $`\mathbf{n}_\tau`$ are the normal vectors of the simplices $\sigma$ and $\tau$, respectively. The distance invariant is a 4-dimensional concatenation of four distances between the simplices. Considering two distinct adjacent simplices share all but one vertex, we can distinguish their shared points $`\{p_i\}`$ from their unique points $a$ and $b$ where $a$ is the unique point of $\sigma$ and $b$ is the unique point of $\tau$. Let $\mathbf{x}$ be the position of a point. This is illustrated in Figure 4. The 4-dimensional distance invariant is then defined as the aggregation of the distances between the unique points and the shared points:
```math
\text{Dist}=\begin{bmatrix} 
\text{Agg}_i \|\mathbf{x}_{p_i} - \mathbf{x}_a\|\\
\text{Agg}_i \|\mathbf{x}_{p_i} - \mathbf{x}_b\|\\
\text{Agg}_{i,j} \|\mathbf{x}_{p_i} - \mathbf{x}_{p_j}\|\\
\|\mathbf{x}_{a} - \mathbf{x}_b\|
\end{bmatrix}
```

<table align="center">
    <tr align="center">
        <td><img src="figures/different-invariants-upper-adjacent-communcation-example.png" width=200></td>
    </tr>
    <tr align="left">
    <td colspan=2><b>Figure 4.</b> Example of different invariants present in upper adjacent communication between 2-simplices (Eijkelboom et al., 2023).</td>
    </tr>
</table>

Message passing in EMPSNs is similar to MPSNs, but the message function is conditioned on the geometric invariants. The message function sent to a simplex $\sigma$ over an adjacency $\mathcal{A}$ is defined as:
```math
\mathbf{m}_{\mathcal{A}}(\sigma) = \text{Agg}_{\tau\in\mathcal{A}(\sigma)} \phi_m(\mathbf{f}_\sigma, \mathbf{f}_\tau, \text{Inv}(\sigma, \tau))
```

where $\text{Inv}(\sigma, \tau)$ is a function that computes the geometric invariants between simplices $\sigma$ and $\tau$. The node features are then updated based on the aggregated messages from all four types of adjacencies:

$E(n)$ equivariant message passing simplicial networks have been shown to perform on par with state-of-the-art approaches for learning on graphs. The usage of higher-dimensional emergent simplex learning has been shown to be beneficial without requiring more parameters, leveraging the benefits of topological and geometric methods. Furthermore, the results indicate that using geometric information combats over-smoothing, with this effect being stronger in higher dimensions. However, the computational cost of EMPSNs is still high, which motivates the need for more efficient models that can maintain high expressivity while reducing computational demands.

### 1.3 SE(n) Equivariant Networks through Weight-Sharing in Position-Orientation Space <!-- Luuk -->
(Bekkers et al., 2023)
<span style="color:red;font-weight:bold;background-color:yellow">TODO @Luuk</span>

## 2. Our novel contribution (we should think of a better title/name for our contribution) <!-- Nin & Kristiyan -->
<span style="color:red;font-weight:bold;background-color:yellow">TODO</span> blablablabla we combine the idea of EMPSN with PONITA blablablabla

## 3. Results <!-- Vincent -->
<span style="color:red;font-weight:bold;background-color:yellow">TODO</span>

## 4. Conclusion <!-- Kristiyan -->
<span style="color:red;font-weight:bold;background-color:yellow">TODO</span>

## 5. Authors' Contributions
<span style="color:red;font-weight:bold;background-color:yellow">TODO</span>

## References
Bekkers, EJ., Vadgama, S., Hesselink, RD., van der Linden, PA. Romero, DW. 2023. Fast, Expressive SE (n) Equivariant Networks through Weight-Sharing in Position-Orientation Space, arXiv preprint arXiv:2310.02970.

Bodnar, C., Frasca, F., Wang, Y., Otter, N., Montufar, G. F., Lio, P., & Bronstein, M. (2021, July). Weisfeiler and lehman go topological: Message passing simplicial networks. In International Conference on Machine Learning (pp. 1026-1037). PMLR.

Eijkelboom, F., Hesselink, R. Bekkers, E. 2023. E(n) Equivariant Message Passing Simplicial Networks. E(n) equivariant message passing simplicial networks.

Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O., & Dahl, G.E. (2017). Neural Message Passing for Quantum Chemistry. International Conference on Machine Learning.

Satorras, V. G., Hoogeboom, E., & Welling, M. (2021, July). E (n) equivariant graph neural networks. In International conference on machine learning (pp. 9323-9332). PMLR.

Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2018). How powerful are graph neural networks?. arXiv preprint arXiv:1810.00826.
