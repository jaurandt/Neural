# Neural Network in C++

A feedforward neural network with backpropagation, implemented from scratch in C++. The network learns to approximate the sine function by training on input/output pairs sampled from `sin(x)`.

## How it works

The network is organized into layers of neurons. Training happens in two phases each pass:

1. **Feed forward** — inputs propagate through the network layer by layer, each neuron computing a weighted sum of its inputs passed through a `tanh` activation function
2. **Backpropagation** — the error between the network's output and the target is propagated backwards through the layers, adjusting each connection's weight using gradient descent

The network topology (number of layers and neurons per layer) is read from the training data file, making it easy to experiment with different architectures without recompiling.

## Project structure

```
Neural/
├── CMakeLists.txt
├── data/
│   └── trainingData.txt    # Training samples for sin(x)
└── src/
    ├── main.cpp            # Training loop
    ├── Net.h / Net.cpp     # Neural network
    ├── Neuron.h / Neuron.cpp   # Neuron and connection weights
    └── TrainingData.h / TrainingData.cpp   # Training data parser
```

## Building and running

**Requirements:** CMake 3.14+, a C++17 compiler (e.g. g++ via [MSYS2](https://www.msys2.org/))

```bash
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build
./build/neural.exe
```

The program prints the inputs, outputs, and targets for each training pass along with the network's recent average error, which decreases as the network learns.

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning rate (η) | 0.15 | Controls the size of each weight update |
| Momentum (α) | 0.3 | Fraction of the previous weight change added to each update |
| Smoothing factor | 10.0 | Window size for the rolling average error |

## Training data format

The training data file specifies the network topology on the first line, followed by alternating input/output pairs:

```
topology: 1 16 8 1
in: 0.0
out: 0.0
in: 0.196
out: 0.195
...
```

The topology `1 16 8 1` means one input neuron, two hidden layers of 16 and 8 neurons, and one output neuron.
