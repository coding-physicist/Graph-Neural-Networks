# Graph Neural Networks Implementation

Implementations of Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and GraphSAGE using PyTorch Geometric.

## Prerequisites

- Python 3.7+
- PyTorch
- PyTorch Geometric
- NumPy
- Matplotlib
- Jupyter Notebook

## Setup

```bash
# Clone repository
git clone https://github.com/coding-physicist/Graph-Neural-Networks.git
cd Graph-Neural-Networks

# Install dependencies
pip install torch torch-geometric numpy matplotlib jupyter

# Launch notebook
jupyter notebook
```

## Models

### GCN (Graph Convolutional Network)
- File: `GCN_Implementation.ipynb`
- Node classification using spectral convolutions
- Formula: H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))

### GAT (Graph Attention Network)
- File: `GAT_Implementation.ipynb`
- Multi-head attention mechanism for graphs
- Formula: α_ij = softmax(LeakyReLU(a^T [W h_i || W h_j]))

### GraphSAGE
- File: `GraphSage Implementation.ipynb`
- Inductive learning via sampling and aggregation
- Formula: h_v^(k+1) = σ(W · CONCAT(h_v^k, h_N(v)))

## Usage

1. Open desired notebook
2. Execute cells sequentially
3. Modify hyperparameters as needed
4. Train on custom datasets by replacing data loading section

## Extension

- Models inherit from `torch.nn.Module`
- Add custom layers in `__init__`
- Implement forward pass logic
- Modify training loop for different tasks

## Citation

```bibtex
@misc{graph-neural-networks-implementation,
  author = {coding-physicist},
  title = {Graph Neural Networks Implementation},
  year = {2025},
  url = {https://github.com/coding-physicist/Graph-Neural-Networks}
}
```

### References

1. GCN: Kipf & Welling (2017). Semi-supervised classification with graph convolutional networks. ICLR.
2. GAT: Veličković et al. (2018). Graph attention networks. ICLR.
3. GraphSAGE: Hamilton et al. (2017). Inductive representation learning on large graphs. NIPS.

## License

MIT License
