<div align="center">

# Deep Learning With PyTorch
### The Yogesh Bawankar Repository

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**A comprehensive educational suite showcasing generative models, from foundational architectures to cutting-edge designs.**

[Explore Projects](#-highlighted-projects) • [Repository Structure](#-repository-structure) • [Contribution](#-contribution)

</div>

---

## Objective

This repository serves as an **academic and teaching-oriented resource** for understanding, building, and visualizing deep generative models.

It is designed to help students, researchers, and practitioners explore the diversity of generative learning approaches in a modular, structured, and highly readable format.

---

## Repository Structure

Each folder is a self-contained module representing a specific category of neural architecture.

| Directory | Description |
| :--- | :--- |
| **auto-regressive-models** | PixelCNN and related sequential density estimators. |
| **cnn** | Basic CNN models for robust image recognition. |
| **diffusion** | Denoising Diffusion Probabilistic Models (**DDPM**, **DDIM**). |
| **dit-models** | Diffusion Transformers (**DiT**) implementations. |
| **energy-based-models** | EBMs trained utilizing Langevin dynamics. |
| **flow-based-models** | Invertible models including **RealNVP** and **Glow**. |
| **gans** | GAN, DCGAN, WGAN, and various conditional variants. |
| **latent-manifold-ae** | Latent space exploration with **VAEs** and **AEs**. |
| **multi-model** | Cross-modal tasks (Text-to-Image, Image Captioning). |
| **rbm** | Contrastive Divergence and Restricted Boltzmann Machines. |
| **rnn** | Recurrent networks including **LSTM** and **GRU**. |
| **score-based-gen-conv** | Score-matching models with CNN backbones. |
| **score-based-gen-models** | Langevin and NCSN-style samplers. |
| **time-series** | Forecasting models tailored for temporal data. |
| **transformer** | Sequence models (Vanilla, GPT architectures). |
| **variational-auto-encoder** | VAEs and conditional variants. |
| **vision-transformer** | **ViT** implementation for image understanding. |

---

## Highlighted Projects

### 1. Diffusion Models
> **"A Concise Implementation of Denoising Diffusion Probabilistic Models"**
* **Core:** U-Net architecture with Gaussian noise scheduling.
* **Feature:** Reverse sampling with iterative denoising.

### 2. Generative Adversarial Networks (GANs)
> **"Adversarial Image Synthesis on MNIST"**
* **Core:** Generator and Discriminator competitive loop.
* **Feature:** Side-by-side comparison of Real vs. Generated images.

### 3. Variational Autoencoders (VAEs)
> **"Latent Variable Modeling and Image Generation"**
* **Core:** The Reparameterization trick.
* **Feature:** Smooth sampling and latent space interpolation.

### 4. Score-Based Models
> **"Unsupervised Image Synthesis via Score Matching"**
* **Core:** Trainable score networks using Langevin Dynamics.
* **Feature:** MCMC sampling techniques.

### 5. Mini DALL·E (Text-to-Image)
> **"Learning Discrete Visual Representations from Textual Descriptions"**
* **Architecture:** VQ-VAE with a Transformer backbone.
* **Capability:** Generates images based on captions (color, shape, objects).

### 6. Image Captioning
> **"Visual Grounding through Language"**
* **Architecture:** ResNet encoder + LSTM decoder with Soft Attention.
* **Capability:** Accurate caption generation for synthetic scenes.

---

## Usage

All notebooks are written for **clarity** and **modularity**. To get started:

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yogesh-bawankar/Deep-Learning-Pytorch.git](https://github.com/yogesh-bawankar/Deep-Learning-Pytorch.git)
    ```
2.  Navigate to the desired model folder.
3.  Install dependencies (ensure you have `torch` and `torchvision` installed).
4.  Run the notebook or script.

---

## Contribution

Contributions are welcome to extend and enhance this educational repository! You may:

* Add new generative model examples.
* Improve visualizations or evaluation metrics.
* Refactor existing notebooks into reusable scripts.

**Steps to contribute:**
1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## License & Acknowledgements

**License:** Distributed under the MIT License. Free for personal, educational, and research use.

**Acknowledgements:**
This repository is inspired by the open-source community and research from:
* *OpenAI, DeepMind, LucidRain*
* *The PyTorch official community*
* *Original authors of DALL·E, VQ-VAE, and DDPM papers*

---
<div align="center">

**Created by Yogesh Bawankar**

</div>
