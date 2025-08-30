# Graph Neural Networks Implementation

A comprehensive implementation of Graph Neural Networks (GNNs) using PyTorch Geometric, featuring GCN, GAT, and GraphSAGE architectures with practical applications and examples.

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
- [Usage Instructions](#usage-instructions)
- [Implemented Models](#implemented-models)
  - [Graph Convolutional Network (GCN)](#graph-convolutional-network-gcn)
  - [Graph Attention Network (GAT)](#graph-attention-network-gat)
  - [GraphSAGE](#graphsage)
- [Example Applications](#example-applications)
- [Running Jupyter Notebooks](#running-jupyter-notebooks)
- [Project Structure](#project-structure)
- [Citing and Acknowledgements](#citing-and-acknowledgements)
- [License](#license)

## Project Description

This repository provides a comprehensive implementation of three fundamental Graph Neural Network architectures: Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and GraphSAGE. The implementations are built using PyTorch Geometric and demonstrate practical applications across various graph-based learning tasks.

Each model is implemented with detailed explanations, visualization capabilities, and real-world examples to help understand the underlying concepts and practical applications of graph neural networks.

## Features

- ✅ **Complete GNN Implementations**: Full implementations of GCN, GAT, and GraphSAGE
- ✅ **PyTorch Geometric Integration**: Leverages the power of PyG for efficient graph operations
- ✅ **Interactive Jupyter Notebooks**: Step-by-step tutorials with explanations
- ✅ **Visualization Tools**: Graph visualization and learning dynamics plotting
- ✅ **Multiple Datasets**: Examples on various graph datasets
- ✅ **Performance Benchmarking**: Comparative analysis of different architectures
- ✅ **Extensible Framework**: Easy to extend for custom applications
- ✅ **Documentation**: Comprehensive documentation and code comments

## Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.7 or higher
- CUDA-compatible GPU (optional, for faster training)
- Git (for cloning the repository)

### Required Python Libraries

The following libraries will be installed during the installation process:

- `torch` (PyTorch)
- `torch-geometric` (PyTorch Geometric)
- `numpy`
- `matplotlib`
- `scikit-learn`
- `jupyter`
- `networkx`
- `seaborn`
- `pandas`

## Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/coding-physicist/Graph-Neural-Networks.git
cd Graph-Neural-Networks
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n gnn-env python=3.8
conda activate gnn-env

# Or using venv
python -m venv gnn-env
# On Windows
gnn-env\Scripts\activate
# On macOS/Linux
source gnn-env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch (choose the appropriate version for your system)
# For CPU-only installation:
pip install torch torchvision torchaudio

# For CUDA installation (example for CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install additional dependencies
pip install numpy matplotlib scikit-learn jupyter networkx seaborn pandas
```

### Step 4: Verify Installation

```python
import torch
import torch_geometric
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Geometric version: {torch_geometric.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Usage Instructions

### Quick Start

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open any of the implementation notebooks**:
   - `GCN_Implementation.ipynb` - Graph Convolutional Networks
   - `GAT_Implementation.ipynb` - Graph Attention Networks
   - `GraphSage Implementation.ipynb` - GraphSAGE implementation

3. **Run the cells step by step** to understand the implementation and see the results.

### Training Custom Models

Each notebook provides examples of how to:
- Load and preprocess graph data
- Initialize the model with custom parameters
- Train the model with your data
- Evaluate performance and visualize results

## Implemented Models

### Graph Convolutional Network (GCN)

**File**: `GCN_Implementation.ipynb`

GCNs perform localized convolutions on graph-structured data by aggregating information from neighboring nodes. This implementation includes:

- **Architecture**: Multi-layer GCN with ReLU activations
- **Key Features**:
  - Spectral-based convolution operations
  - Efficient message passing
  - Node classification and graph-level tasks
- **Applications**: Social network analysis, citation networks, molecular property prediction

**Mathematical Foundation**:
```
H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
```

### Graph Attention Network (GAT)

**File**: `GAT_Implementation.ipynb`

GATs introduce attention mechanisms to graph neural networks, allowing nodes to attend to different neighbors with varying importance.

- **Architecture**: Multi-head attention with learnable attention weights
- **Key Features**:
  - Self-attention mechanism for graphs
  - Multi-head attention for richer representations
  - Inductive learning capabilities
- **Applications**: Knowledge graphs, social influence prediction, protein interaction networks

**Mathematical Foundation**:
```
α_ij = softmax(LeakyReLU(a^T [W h_i || W h_j]))
h_i' = σ(Σ_j α_ij W h_j)
```

### GraphSAGE

**File**: `GraphSage Implementation.ipynb`

GraphSAGE (SAmple and aggreGatE) is designed for inductive representation learning on large graphs by sampling and aggregating from node neighborhoods.

- **Architecture**: Sampling-based aggregation with multiple aggregator functions
- **Key Features**:
  - Inductive learning (works on unseen nodes)
  - Multiple aggregation functions (mean, LSTM, pooling)
  - Scalable to large graphs
- **Applications**: Large-scale social networks, recommendation systems, dynamic graphs

**Mathematical Foundation**:
```
h_N(v) = AGGREGATE({h_u : u ∈ N(v)})
h_v^(k+1) = σ(W · CONCAT(h_v^k, h_N(v)))
```

## Example Applications

### 1. Node Classification
- **Dataset**: Cora, CiteSeer, PubMed
- **Task**: Classify academic papers by topic
- **Models**: All three architectures compared

### 2. Graph Classification
- **Dataset**: MUTAG, PROTEINS
- **Task**: Classify entire graphs (molecular property prediction)
- **Models**: Adapted versions with graph-level pooling

### 3. Link Prediction
- **Dataset**: Facebook social network
- **Task**: Predict future connections in social networks
- **Models**: Modified architectures for edge prediction

### 4. Graph Generation
- **Dataset**: Small molecular graphs
- **Task**: Generate new molecular structures
- **Models**: Variational graph autoencoders

## Running Jupyter Notebooks

### System Requirements
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: 2GB free space for datasets and outputs
- **GPU**: Optional, but recommended for large datasets

### Notebook Execution Guide

1. **Start with GCN Implementation**:
   ```bash
   jupyter notebook GCN_Implementation.ipynb
   ```
   - Begin with the fundamentals of graph convolutions
   - Understand message passing mechanisms
   - Experiment with different layer configurations

2. **Explore GAT Implementation**:
   ```bash
   jupyter notebook GAT_Implementation.ipynb
   ```
   - Learn attention mechanisms in graphs
   - Visualize attention weights
   - Compare single-head vs multi-head attention

3. **Advanced GraphSAGE Techniques**:
   ```bash
   jupyter notebook "GraphSage Implementation.ipynb"
   ```
   - Understand inductive learning
   - Experiment with different aggregators
   - Test on large-scale datasets

### Troubleshooting Common Issues

- **CUDA Out of Memory**: Reduce batch size or use CPU
- **Package Import Errors**: Ensure all dependencies are installed
- **Slow Training**: Enable GPU acceleration or reduce dataset size

## Project Structure

```
Graph-Neural-Networks/
├── GCN_Implementation.ipynb      # Graph Convolutional Network implementation
├── GAT_Implementation.ipynb      # Graph Attention Network implementation
├── GraphSage Implementation.ipynb # GraphSAGE implementation
├── README.md                     # Project documentation (this file)
├── data/                         # Directory for datasets (created during runtime)
├── outputs/                      # Generated plots and results (created during runtime)
└── models/                       # Saved model checkpoints (created during runtime)
```

### Generated Directories

During execution, the notebooks will create additional directories:
- `data/`: Downloaded datasets and preprocessed data
- `outputs/`: Visualization plots, training curves, and results
- `models/`: Trained model parameters and checkpoints

## Citing and Acknowledgements

### Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{graph-neural-networks-implementation,
  author = {coding-physicist},
  title = {Graph Neural Networks Implementation},
  year = {2025},
  url = {https://github.com/coding-physicist/Graph-Neural-Networks},
  note = {Implementation of GCN, GAT, and GraphSAGE using PyTorch Geometric}
}
```

### Original Papers

This implementation is based on the following seminal papers:

1. **GCN**: Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.
2. **GAT**: Veličković, P., et al. (2018). Graph attention networks. ICLR.
3. **GraphSAGE**: Hamilton, W., et al. (2017). Inductive representation learning on large graphs. NIPS.

### Acknowledgements

- **PyTorch Geometric Team**: For the excellent graph deep learning library
- **PyTorch Team**: For the robust deep learning framework
- **Open Source Community**: For datasets and inspiration
- **Academic Researchers**: For the foundational GNN papers and methods

### Special Thanks

- All contributors and users of this repository
- The graph neural network research community
- Stack Overflow and GitHub communities for problem-solving support

## License

This project is licensed under the MIT License - see the details below:

```
MIT License

Copyright (c) 2025 coding-physicist

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHOS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contact and Support

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and community support
- **Email**: For private inquiries (check profile for contact info)

**Happy Learning with Graph Neural Networks! 🚀📊**
