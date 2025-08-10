# &Pi;net: Optimizing hard-constrained neural networks with orthogonal projection layers

[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/TODO)
[![GitHub stars](https://img.shields.io/github/stars/antonioterpin/pinet?style=social)](https://github.com/antonioterpin/pinet/stargazers)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://github.com/antonioterpin/pinet/LICENSE)
[![codecov](https://codecov.io/gh/antonioterpin/pinet/graph/badge.svg?token=UQ48NNZSI4)](https://codecov.io/gh/antonioterpin/pinet)
[![Tests](https://github.com/antonioterpin/pinet/actions/workflows/test.yaml/badge.svg)](https://github.com/antonioterpin/pinet/actions/workflows/test.yaml)
[![PyPI version](https://img.shields.io/pypi/v/pinet-hcnn.svg)](https://pypi.org/project/pinet-hcnn)

[![Follow Panos](https://img.shields.io/badge/LinkedIn-Panagiotis%20Grontas-blue?&logo=linkedin)](https://www.linkedin.com/in/panagiotis-grontas-4517b0184)
[![Follow Antonio](https://img.shields.io/twitter/follow/antonio_terpin.svg?style=social)](https://twitter.com/antonio_terpin)

![Cover Image](media/cover.jpg)

This repository contains a [JAX](https://github.com/jax-ml/jax) implementation of &Pi;net, an output layer for neural networks that ensures the satisfaction of specified convex constraints.

> [!NOTE] TL;DR
> &Pi;net leverages operator splitting for rapid and reliable projections in the forward pass, and the implicit function theorem for backpropagation. It offers a *feasible-by-design* optimization proxy for parametric constrained optimization problems to obtain modest-accuracy solutions faster than traditional solvers when solving a single problem, and significantly faster for a batch of problems.

## Index
- [Getting started](#getting-started)
- [Examples](#examples)
- [Contributing](#contributing-☕️)
- [Citation](#citation-🙏)

## Getting started
To install &Pi;net, run:
- CPU-only (Linux/macOS/Windows)
  ```bash
  pip install pinet-hcnn
  ```
- GPU (NVIDIA, CUDA 12)
  ```bash
  pip install "pinet-hcnn[cuda12]"
  ```

> [!WARNING] CUDA dependencies
> .....

We also provide a working [Docker](https://docs.docker.com/) image to reproduce the results of the paper and to build on top.
```bash
docker compose build
docker compose ...
```


### Supported platforms 💻
|        | Linux x86\_64 | Linux aarch64 | Mac aarch64 | Windows x86\_64 | Windows WSL2 x86\_64 |
| -------------- | ------------- | ------------- | ----------- | --------------- | -------------------- |
| **CPU**        | ✅           | ✅           | ✅         | ✅             | ✅                  |
| **NVIDIA GPU** | ✅           | ✅           | n/a         | ❌              | ❌         |


## Examples

### A toy example: Approximating a MPC controller

![Animation of the training values]()
![Animation of the deployed controller]()

> [!TIP] custom
> **TITLE**?<br/>
> Another line

### Works using &Pi;net ⚙️
We collect here applications using &Pi;net. Please feel free to open a pull request to add yours! 🤗

Title | Link
--|--
Multi-vehicle trajectory optimization with non-convex preferences | [![View Repo](https://img.shields.io/badge/GitHub-antonioterpin%2Fglitch-blue?logo=github)](https://github.com/antonioterpin/glitch)




## Contributing ☕️
Contributions are more than welcome! 🙏 Please check out our [contributing page](./CONTRIBUTING.md), and feel free to open an issue for problems and feature requests⚠️.

## Benchmarks 📈
Below, we summarize the performance gains of &Pi;net over state-of-the-art methods. We consider three metrics:
- Relative Suboptimality ($\texttt{RS}$): The suboptimality of a candidate solution $\hat{y}$ compared to the optimal objective $J(y^{\star})$, computed by a high-accuracy solver.
- Constraint Violation ($\texttt{CV}$): Maximum violation ($\infty$-norm) of any constraint (equality and inequality). In practice, any solver achieving a $\texttt{CV}$ below $10^{-5}$ is considered to have high accuracy and there is little benefit to go below that. Instead, when methods have sufficiently low $\texttt{CV}$, having a low $\texttt{RS}$ is better.
- Learning curves: Progress on $\texttt{RS}$ and $\texttt{CV}$ over wall-clock time on the validation set.
- Single inference time: The time required to solve one instance at test time.
- Batch inference time: The time required to solve a batch of $1024$ instances at test time.

We report the results for an optimization problem with optimization variable of dimension $d$, $n_{\mathrm{eq}}$ equality and $n_{\mathrm{ineq}}$ inequality convex constraints and with a  non-convex objective. Here, we use a small and a large (in the parametric optimization sense) datasets $(d, n_{\mathrm{eq}}, n_{\mathrm{ineq}})  \in \{(100, 50, 50), (1000, 500, 500)\}$.

![Non-convex CV and RS](media/nonconvex-cvrs.jpg)
![Non-convex learning curves](media/nonconvex-times.jpg)

Overall, &Pi;net outperforms the state-of-the-art in accuracy and training times.
For more comparisons and ablations, please check out our [paper](TODO).

## Citation 🙏
If you use this code in your research, please cite our paper:
```bash
   @inproceedings{grontas2025pinet,
     title={Pinet: Optimizing hard-constrained neural networks with orthogonal projection layers},
     author={Grontas, Panagiotis and Terpin, Antonio and Balta C., Efe and D'Andrea, Raffaello and Lygeros, John},
     journal={arXiv preprint arXiv:TODO},
     year={2025}
   }
```
