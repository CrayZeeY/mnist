# MNIST Digit Classifier from Scratch in C

A handwritten digit classifier built entirely in C with no machine learning libraries.
Trained on the MNIST dataset, achieves **97.95% test accuracy**, and includes an
interactive drawing app where you draw a digit and the model predicts it in real time.

![Demo](demo.gif)

---

## What this is

Most people learn neural networks by calling `model.fit()` in Keras or PyTorch.
This project builds everything from scratch in C:

- Forward pass — manual matrix multiplication, layer by layer
- Backpropagation — chain rule implemented by hand
- Softmax + cross-entropy loss
- Xavier weight initialization
- Minibatch gradient descent with shuffle
- Binary weight saving and loading
- Interactive drawing GUI using Python + Tkinter

No NumPy. No PyTorch. No TensorFlow. Just C, math, and the MNIST dataset.

---

## Architecture

```
Input       Hidden 1     Hidden 2     Output
784 ──────► 128 ────────► 64 ────────► 10
(28x28)    (ReLU)        (ReLU)       (Softmax)
```

| Parameter       | Value              |
|-----------------|--------------------|
| Input neurons   | 784 (28×28 pixels) |
| Hidden layer 1  | 128 neurons (ReLU) |
| Hidden layer 2  | 64 neurons (ReLU)  |
| Output neurons  | 10 (digits 0–9)    |
| Loss function   | Cross-entropy      |
| Optimizer       | SGD with minibatch |
| Learning rate   | 0.01               |
| Batch size      | 32                 |
| Epochs          | 20                 |
| Test accuracy   | **97.95%**         |

---

## Results

| Epoch | Loss   | Train Acc | Test Acc |
|-------|--------|-----------|----------|
| 1     | 0.2253 | 96.62%    | 96.17%   |
| 5     | 0.0539 | 98.83%    | 97.50%   |
| 10    | 0.0271 | 99.40%    | 97.73%   |
| 16    | 0.0194 | 99.46%    | **97.95%** |
| 20    | 0.0193 | 99.54%    | 97.82%   |

---

## Project structure

```
mnist/
├── src/
│   ├── activations.h / .c   ← sigmoid, relu, softmax and derivatives
│   ├── network.h / .c       ← forward pass, backprop, save, load weights
│   ├── data.h / .c          ← MNIST binary file loader, memory management
│   ├── train.c              ← training loop, minibatch, shuffle
│   └── predict.c            ← load weights and classify test images
├── data/                    ← MNIST dataset files (downloaded separately)
├── weights/                 ← saved weights after training (mnist.dat)
├── draw.py                  ← interactive drawing app
└── Makefile                 ← build system
```

---

## Getting started

### Requirements

```bash
sudo apt install build-essential wget python3-tk python3-numpy -y
```

### Build

```bash
git clone https://github.com/yourusername/mnist-from-scratch-c
cd mnist-from-scratch-c
make
```

### Download MNIST data

```bash
make data
```

This downloads and unzips the 4 MNIST binary files (~55MB total) into `data/`.

### Train

```bash
./train
```

Trains for 20 epochs (~3 minutes on CPU). Weights are saved to `weights/mnist.dat`
after every epoch so training is never lost.

### Predict on test images

```bash
./predict              # first 10 test images
./predict 42           # image at index 42
./predict 42 show      # show ASCII art + confidence scores
```

Example output:
```
Index  Truth    Pred     Result
------ -------- -------- --------
42     7        7        correct
  0: 0.0%
  1: 0.0%
  ...
  7: 99.2%  <- predicted
  ...
```

### Interactive drawing app

```bash
python3 draw.py
```

Opens a window. Draw any digit with your mouse — the model predicts it instantly
with confidence scores for all 10 digits.

---

## How it works

### Forward pass

Each layer computes a weighted sum of its inputs, adds a bias, then applies an
activation function:

```
z = W·x + b
a = activation(z)
```

Hidden layers use ReLU. The output layer uses Softmax to convert raw scores into
probabilities that sum to 1.

### Backpropagation

Gradients flow backwards through the network using the chain rule. For the output
layer, combining Softmax with cross-entropy loss produces a clean gradient:

```
delta_output = predicted - truth
```

Hidden layer gradients are computed by backpropagating through the weight matrices
and multiplying by the ReLU derivative.

### Weight update

```
W = W - learning_rate × gradient
```

Repeated for every weight and bias in the network, after every minibatch of 32 samples.

---

## What I learned

- How neural networks actually work under the hood — not just API calls
- Implementing backpropagation from the chain rule, manually
- Why Xavier initialization matters for deep networks
- How softmax + cross-entropy loss simplifies the output gradient
- Memory management in C — malloc, free, heap vs stack
- Why this architecture plateaus at ~98% and what CNNs solve

---

## Next steps

- [ ] Implement a CNN in C for 99%+ accuracy
- [ ] Add CUDA GPU acceleration
- [ ] Data augmentation (rotations, shifts) to improve handwriting generalization
- [ ] Web demo using WebAssembly

---

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) — 60,000 training images and 10,000 test
images of handwritten digits, collected by Yann LeCun et al.

---

## Author

Biswas — BSc CSIT student, Tribhuvan University