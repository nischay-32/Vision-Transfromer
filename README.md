# Vision Transformer (ViT) for CIFAR-10 Classification

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository contains a TensorFlow/Keras implementation of a **Vision Transformer (ViT)** model, designed and trained for image classification on the **CIFAR-10** dataset. The code demonstrates the core concepts of the ViT architecture, including image patching, positional embeddings, and the Transformer encoder block.



***

## 📚 Overview

The Vision Transformer is a model for image classification that applies the standard Transformer architecture, famous for its success in natural language processing, to image data. Instead of processing word tokens, a ViT splits an image into a sequence of fixed-size patches, flattens them, and feeds them into a series of Transformer blocks. This project provides a clear and commented implementation of this architecture from scratch.

### Key Features
- **Custom Keras Layers:** Implements `Patches` and `PatchEncoder` as custom `tf.keras.layers.Layer` for modularity.
- **Data Augmentation:** Utilizes Keras preprocessing layers for on-the-fly data augmentation (flipping, rotation, zoom) to improve model generalization.
- **Transformer Encoder:** Builds the core of the model using standard `MultiHeadAttention` and MLP blocks.
- **Training and Evaluation:** Includes a complete workflow to compile, train, and evaluate the model, saving the best-performing weights.

***

## ⚙️ Model Architecture

The model follows the standard ViT pipeline:

1.  **Image Patching:** The input image (resized to 72x72) is divided into a grid of non-overlapping 6x6 patches.
2.  **Patch & Position Embedding:** Each patch is flattened and linearly projected into a vector. Learnable positional embeddings are added to these vectors to retain spatial information.
3.  **Transformer Encoder:** The sequence of embedded patches is passed through a series of Transformer Encoder blocks. Each block consists of:
    * Layer Normalization
    * Multi-Head Self-Attention
    * Residual Connections
    * Layer Normalization
    * Multi-Layer Perceptron (MLP)
4.  **MLP Head:** The output from the Transformer Encoder is fed into a final MLP head for classification into one of the 10 CIFAR-10 classes.

***

## 📊 Dataset

The model is trained and evaluated on the **CIFAR-10 dataset**. This dataset consists of 60,000 32x32 color images across 10 distinct classes (e.g., airplane, automobile, bird, cat). The dataset is automatically downloaded via the `tf.keras.datasets` API.



***

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

You'll need Python 3.8+ and the following libraries. It's recommended to create a virtual environment to manage dependencies.

```
tensorflow
matplotlib
numpy
```

You can install them by creating a `requirements.txt` file with the content above and running:
```bash
pip install -r requirements.txt
```

### Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/vit-cifar10.git](https://github.com/YOUR_USERNAME/vit-cifar10.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd vit-cifar10
    ```
3.  Install the required packages as described in the prerequisites.

### Usage

To train the model, simply execute the main Python script.

```bash
python train_vit.py
```

The script will perform the following actions:
1.  Load the CIFAR-10 dataset.
2.  Build the Vision Transformer model.
3.  Compile the model with the AdamW optimizer and Sparse Categorical Crossentropy loss.
4.  Train the model for the specified number of epochs, saving the best weights to the `./tmp/` directory.
5.  Evaluate the best model on the test set and print the final accuracy.
6.  Run and display a prediction on a sample test image.

***

## 📈 Results

After training, the model's performance will be evaluated on the test set. The expected output will include the test accuracy and top-5 accuracy.

* **Test Accuracy:** `XX.XX%`
* **Test Top 5 Accuracy:** `XX.XX%`

*(Note: Final accuracy will depend on hyperparameters and training duration. The values in the provided code are set for a comprehensive run.)*

***

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

***

## 🙏 Acknowledgments

This implementation is based on the concepts introduced in the paper:

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.# Vision-Transfromer
