# Image Generation with Generative Adversarial Networks (GANs)

This project provides a comprehensive implementation of a Generative Adversarial Network (GAN) for generating realistic handwritten digit images using the MNIST dataset. The implementation is built in TensorFlow and presented as an interactive Jupyter notebook that guides users through the entire process of building, training, and evaluating a GAN.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Training Details](#training-details)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)
- [References](#references)

## Introduction

Generative Adversarial Networks (GANs) are a class of machine learning frameworks introduced by Ian Goodfellow et al. in 2014. GANs consist of two neural networks—the Generator and the Discriminator—that compete against each other in a zero-sum game. The Generator learns to produce fake data that resembles real data, while the Discriminator learns to distinguish between real and fake data. Through this adversarial process, both networks improve until the Generator produces highly realistic outputs.

![GAN Architecture Diagram](https://media.geeksforgeeks.org/wp-content/uploads/20250801145950815286/gan.webp)

This project implements a basic GAN architecture to generate images of handwritten digits (0-9) that mimic the MNIST dataset.

## Features

- Complete GAN implementation with TensorFlow 2.x
- Modular architecture with separate Generator and Discriminator networks
- Training loop with visualization of generated images during training
- Evaluation metrics and testing procedures
- Interactive Jupyter notebook with step-by-step explanations
- Support for both standard GAN loss and Least Squares GAN (LSGAN) variants
- Preprocessing utilities for MNIST dataset
- Visualization tools for generated images

## Architecture

### Generator Network
The Generator takes a random noise vector as input and produces synthetic images:

- **Input**: Random noise vector (96 dimensions)
- **Layer 1**: Dense layer (1024 units) → ReLU activation
- **Layer 2**: Dense layer (1024 units) → ReLU activation
- **Output Layer**: Dense layer (784 units) → Tanh activation
- **Output Shape**: (batch_size, 784) - flattened 28x28 images

### Discriminator Network
The Discriminator evaluates whether an input image is real or fake:

- **Input**: Flattened image (784 dimensions)
- **Layer 1**: Dense layer (256 units) → Leaky ReLU activation (α=0.01)
- **Layer 2**: Dense layer (256 units) → Leaky ReLU activation (α=0.01)
- **Output Layer**: Dense layer (1 unit) - no activation (logits)
- **Output**: Single scalar representing "realness" score

## Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- TensorFlow 2.x
- NumPy
- Matplotlib
- Git (for cloning the repository)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/image-generation-gan.git
   cd image-generation-gan
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv gan_env
   source gan_env/bin/activate  # On Windows: gan_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   ```

## Usage

1. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook:**
   Navigate to `notebooks/generative-adversarial-networks-gans.ipynb` and open it.

3. **Run the notebook cells sequentially:**
   - **Setup**: Import libraries and set up the environment
   - **Dataset**: Load and preprocess MNIST data
   - **Model Definition**: Implement Generator and Discriminator architectures
   - **Loss Functions**: Define training objectives
   - **Training**: Train the GAN with visualization
   - **Evaluation**: Test and analyze results

4. **Key functions to run:**
   - `test_discriminator()` - Verify discriminator architecture
   - `test_generator()` - Verify generator architecture
   - `run_a_gan()` - Execute the training loop

## Training Details

### Hyperparameters
- **Batch Size**: 128
- **Number of Epochs**: 10 (configurable)
- **Noise Dimension**: 96
- **Learning Rate**: 0.001
- **Beta1 (Adam optimizer)**: 0.5
- **Loss Function**: Binary cross-entropy (standard GAN) or Mean squared error (LSGAN variant)

### Training Process
1. Sample real images from MNIST dataset
2. Generate fake images from random noise
3. Train Discriminator on real vs. fake images
4. Train Generator to fool the Discriminator
5. Repeat for multiple epochs with periodic visualization

### Monitoring Training
The training loop provides:
- Loss values for Generator and Discriminator
- Generated image samples every 20 iterations
- Final generated images after training completion

## Results

After training, the GAN should generate realistic-looking handwritten digits. The notebook includes:

- **Visualization**: Grid of generated images
- **Quality Assessment**: Qualitative evaluation of generated samples
- **Training Curves**: Loss progression plots (if implemented)
- **Comparison**: Real vs. generated image comparison

Expected outcomes:
- Generated digits should resemble handwritten numbers 0-9
- Images should be 28x28 grayscale (matching MNIST format)
- Quality improves with more training epochs

## Project Structure

```
image-generation-gan/
│
├── notebooks/
│   └── generative-adversarial-networks-gans.ipynb    # Main implementation
│
├── unit-testing-data/
│   └── gan-checks-tf.npz                            # Test data for validation
│
├── requirements.txt                                  # Python dependencies
│
└── README.md                                         # This file
```

## Dependencies

- `tensorflow>=2.0.0` - Deep learning framework
- `numpy>=1.18.0` - Numerical computing
- `matplotlib>=3.2.0` - Data visualization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [Goodfellow, I., et al. "Generative Adversarial Nets." NIPS, 2014.](https://arxiv.org/abs/1406.2661)
- [Radford, A., et al. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." ICLR, 2016.](https://arxiv.org/abs/1511.06434)
- [Mao, X., et al. "Least Squares Generative Adversarial Networks." ICCV, 2017.](https://arxiv.org/abs/1611.04076)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---

**Note**: This implementation is for educational purposes. For production use, consider more robust architectures and extensive hyperparameter tuning.