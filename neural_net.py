#!/usr/bin/env python3
"""neural_net - Minimal neural network from scratch."""
import argparse, math, random, sys

class Tensor:
    def __init__(self, data, grad=0.0): self.data = data; self.grad = grad

def sigmoid(x): return 1/(1+math.exp(-max(-500,min(500,x))))
def sigmoid_d(x): s=sigmoid(x); return s*(1-s)
def relu(x): return max(0,x)
def relu_d(x): return 1.0 if x>0 else 0.0

class Layer:
    def __init__(self, nin, nout, act="sigmoid"):
        self.w = [[random.gauss(0,0.5) for _ in range(nin)] for _ in range(nout)]
        self.b = [0.0]*nout; self.act = act
        self.z = [0.0]*nout; self.a_in = []
    def forward(self, x):
        self.a_in = x[:]
        out = []
        for j in range(len(self.b)):
            z = sum(self.w[j][i]*x[i] for i in range(len(x))) + self.b[j]
            self.z[j] = z
            if self.act=="sigmoid": out.append(sigmoid(z))
            elif self.act=="relu": out.append(relu(z))
            else: out.append(z)
        return out
    def backward(self, d_out, lr):
        d_in = [0.0]*len(self.a_in)
        for j in range(len(self.b)):
            if self.act=="sigmoid": dz = d_out[j]*sigmoid_d(self.z[j])
            elif self.act=="relu": dz = d_out[j]*relu_d(self.z[j])
            else: dz = d_out[j]
            for i in range(len(self.a_in)):
                d_in[i] += self.w[j][i]*dz
                self.w[j][i] -= lr*dz*self.a_in[i]
            self.b[j] -= lr*dz
        return d_in

class Network:
    def __init__(self, sizes, act="sigmoid"):
        self.layers = [Layer(sizes[i],sizes[i+1],act) for i in range(len(sizes)-1)]
    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x
    def train(self, X, Y, epochs=1000, lr=0.5):
        for ep in range(epochs):
            loss = 0
            for x, y in zip(X, Y):
                out = self.forward(x)
                d = [out[i]-y[i] for i in range(len(y))]
                loss += sum(e*e for e in d)
                for l in reversed(self.layers): d = l.backward(d, lr)
            if ep % (epochs//10) == 0: print(f"  Epoch {ep}: loss={loss/len(X):.6f}")

def main():
    p = argparse.ArgumentParser(description="Neural network")
    p.add_argument("--demo", choices=["xor","and","or"], default="xor")
    p.add_argument("-e","--epochs", type=int, default=2000)
    p.add_argument("-l","--lr", type=float, default=1.0)
    a = p.parse_args()
    data = {"xor":([[0,0],[0,1],[1,0],[1,1]],[[0],[1],[1],[0]]),
            "and":([[0,0],[0,1],[1,0],[1,1]],[[0],[0],[0],[1]]),
            "or":([[0,0],[0,1],[1,0],[1,1]],[[0],[1],[1],[1]])}
    X, Y = data[a.demo]
    net = Network([2,4,1])
    print(f"Training {a.demo.upper()} gate:")
    net.train(X, Y, a.epochs, a.lr)
    print("\nResults:")
    for x, y in zip(X, Y):
        out = net.forward(x)
        print(f"  {x} -> {out[0]:.4f} (expected {y[0]})")

if __name__ == "__main__": main()
