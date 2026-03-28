#!/usr/bin/env python3
"""Minimal neural network with backpropagation. Zero dependencies."""
import sys, math, random, json

def sigmoid(x): return 1 / (1 + math.exp(-max(-500, min(500, x))))
def sigmoid_d(x): return x * (1 - x)

class NeuralNet:
    def __init__(self, layers):
        self.weights = []
        self.biases = []
        for i in range(len(layers)-1):
            w = [[random.gauss(0, 1) / math.sqrt(layers[i]) for _ in range(layers[i])] for _ in range(layers[i+1])]
            b = [0.0] * layers[i+1]
            self.weights.append(w)
            self.biases.append(b)
    def forward(self, x):
        self.activations = [x]
        for w, b in zip(self.weights, self.biases):
            x = [sigmoid(sum(wi*xi for wi,xi in zip(wj, x)) + bj) for wj, bj in zip(w, b)]
            self.activations.append(x)
        return x
    def train(self, x, target, lr=0.1):
        output = self.forward(x)
        deltas = [[(o - t) * sigmoid_d(o) for o, t in zip(output, target)]]
        for l in range(len(self.weights)-1, 0, -1):
            d = []
            for j in range(len(self.weights[l-1])):
                err = sum(deltas[0][k] * self.weights[l][k][j] for k in range(len(deltas[0])))
                d.append(err * sigmoid_d(self.activations[l][j]))
            deltas.insert(0, d)
        for l in range(len(self.weights)):
            for j in range(len(self.weights[l])):
                for k in range(len(self.weights[l][j])):
                    self.weights[l][j][k] -= lr * deltas[l][j] * self.activations[l][k]
                self.biases[l][j] -= lr * deltas[l][j]
        return sum((o-t)**2 for o,t in zip(output, target)) / len(target)
    def save(self, path):
        json.dump({'weights': self.weights, 'biases': self.biases}, open(path, 'w'))
    def load(self, path):
        d = json.load(open(path))
        self.weights, self.biases = d['weights'], d['biases']

if __name__ == '__main__':
    if '--xor' in sys.argv:
        nn = NeuralNet([2, 4, 1])
        data = [([0,0],[0]),([0,1],[1]),([1,0],[1]),([1,1],[0])]
        for epoch in range(5000):
            loss = sum(nn.train(x, y) for x, y in data) / 4
            if epoch % 1000 == 0: print(f"Epoch {epoch}: loss={loss:.6f}")
        print("\nResults:")
        for x, y in data:
            pred = nn.forward(x)
            print(f"  {x} → {pred[0]:.4f} (expected {y[0]})")
    else:
        print("Usage: neural_net.py --xor")
        print("       A minimal neural network demo learning XOR")
