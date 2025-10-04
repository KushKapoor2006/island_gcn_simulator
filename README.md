# Algorithmic Simulator for Islandization-GCN

This project is a Python-based simulation of the core algorithms presented in the research paper: **"I-GCN: A Graph Convolutional Network Accelerator with Runtime Locality Enhancement through Islandization"**.

The goal is to provide a clear, high-level implementation of the "islandization" technique for educational purposes. It demonstrates how a graph can be partitioned into dense, local clusters (islands) to improve the efficiency of Graph Convolutional Network (GCN) computations.

**Note:** This is an *algorithmic simulation*, not a hardware model. The performance benefits described in the paper (e.g., reduced off-chip memory access) are conceptualized here but not physically benchmarked.

## Core Concepts

The simulation is divided into two main stages, mirroring the paper's architecture:

1.  **Island Locator**: This stage finds dense subgraphs or "islands" within the main graph by identifying high-degree "hubs" and performing a size-constrained Breadth-First Search (BFS) from each one.

2.  **Island Consumer**: This stage simulates the GCN computations on the found islands. By processing one island at a time, it mimics the data locality benefits of loading a small, dense cluster into fast on-chip memory.


## Usage

To run the full simulation pipeline, execute the `main.py` script:

```bash
python main.py
```
