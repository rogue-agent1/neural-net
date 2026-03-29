#!/usr/bin/env python3
"""Neural network (MLP) from scratch with backpropagation."""
import sys, math, random

def sigmoid(x): return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
def sigmoid_d(x): return x * (1 - x)
def relu(x): return max(0, x)
def relu_d(x): return 1 if x > 0 else 0
def tanh_act(x): return math.tanh(x)
def tanh_d(x): return 1 - x * x

class Layer:
    def __init__(self, n_in, n_out, activation="sigmoid"):
        self.weights = [[random.gauss(0, math.sqrt(2.0/n_in)) for _ in range(n_in)] for _ in range(n_out)]
        self.biases = [0.0] * n_out
        self.act_fn = {"sigmoid": sigmoid, "relu": relu, "tanh": tanh_act}[activation]
        self.act_d = {"sigmoid": sigmoid_d, "relu": relu_d, "tanh": tanh_d}[activation]
        self.output = []; self.input = []

    def forward(self, x):
        self.input = x; self.output = []
        for i in range(len(self.biases)):
            s = self.biases[i] + sum(w * xi for w, xi in zip(self.weights[i], x))
            self.output.append(self.act_fn(s))
        return self.output

class Network:
    def __init__(self, sizes, activation="sigmoid", lr=0.1):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Layer(sizes[i], sizes[i+1], activation))
        self.lr = lr

    def forward(self, x):
        for layer in self.layers: x = layer.forward(x)
        return x

    def train(self, x, target):
        output = self.forward(x)
        # Backprop
        deltas = [None] * len(self.layers)
        # Output layer
        L = self.layers[-1]
        deltas[-1] = [(target[i] - output[i]) * L.act_d(output[i]) for i in range(len(output))]
        # Hidden layers
        for l in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[l]; next_layer = self.layers[l + 1]
            deltas[l] = []
            for i in range(len(layer.output)):
                error = sum(deltas[l+1][j] * next_layer.weights[j][i] for j in range(len(next_layer.biases)))
                deltas[l].append(error * layer.act_d(layer.output[i]))
        # Update
        for l, layer in enumerate(self.layers):
            for i in range(len(layer.biases)):
                for j in range(len(layer.weights[i])):
                    layer.weights[i][j] += self.lr * deltas[l][i] * layer.input[j]
                layer.biases[i] += self.lr * deltas[l][i]
        loss = sum((t - o) ** 2 for t, o in zip(target, output)) / len(target)
        return loss

def demo_xor():
    print("=== XOR Neural Network ===")
    net = Network([2, 4, 1], activation="sigmoid", lr=0.5)
    data = [([0,0],[0]), ([0,1],[1]), ([1,0],[1]), ([1,1],[0])]
    for epoch in range(5000):
        loss = sum(net.train(x, y) for x, y in data) / len(data)
        if epoch % 1000 == 0: print(f"Epoch {epoch}: loss={loss:.6f}")
    print("\nResults:")
    for x, y in data:
        pred = net.forward(x)
        print(f"  {x} -> {pred[0]:.4f} (expected {y[0]})")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--iris":
        print("Training on simple classification..."); demo_xor()
    else: demo_xor()

if __name__ == "__main__": main()
