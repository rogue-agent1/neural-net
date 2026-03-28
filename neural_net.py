#!/usr/bin/env python3
"""Neural network from scratch (feedforward + backprop)."""
import math,random
def sigmoid(x): return 1/(1+math.exp(-max(-500,min(500,x))))
def sigmoid_d(x): s=sigmoid(x);return s*(1-s)
def relu(x): return max(0,x)
def relu_d(x): return 1 if x>0 else 0
class Layer:
    def __init__(self,n_in,n_out,act="sigmoid"):
        self.W=[[random.gauss(0,math.sqrt(2/n_in)) for _ in range(n_in)] for _ in range(n_out)]
        self.b=[0]*n_out;self.act=sigmoid if act=="sigmoid" else relu
        self.act_d=sigmoid_d if act=="sigmoid" else relu_d
        self.z=None;self.a=None;self.input=None
    def forward(self,x):
        self.input=x;self.z=[];self.a=[]
        for j in range(len(self.W)):
            z=sum(self.W[j][i]*x[i] for i in range(len(x)))+self.b[j]
            self.z.append(z);self.a.append(self.act(z))
        return self.a
class NeuralNet:
    def __init__(self,sizes,act="sigmoid"):
        self.layers=[Layer(sizes[i],sizes[i+1],act) for i in range(len(sizes)-1)]
    def forward(self,x):
        for layer in self.layers: x=layer.forward(x)
        return x
    def train(self,X,Y,lr=0.1,epochs=1000):
        for ep in range(epochs):
            total_loss=0
            for x,y in zip(X,Y):
                out=self.forward(x)
                loss=sum((o-t)**2 for o,t in zip(out,y));total_loss+=loss
                deltas=[None]*len(self.layers)
                L=self.layers[-1]
                deltas[-1]=[(out[j]-y[j])*L.act_d(L.z[j]) for j in range(len(out))]
                for l in range(len(self.layers)-2,-1,-1):
                    layer=self.layers[l];next_layer=self.layers[l+1]
                    deltas[l]=[sum(deltas[l+1][j]*next_layer.W[j][i] for j in range(len(deltas[l+1])))*layer.act_d(layer.z[i]) for i in range(len(layer.z))]
                for l in range(len(self.layers)):
                    layer=self.layers[l]
                    for j in range(len(layer.W)):
                        for i in range(len(layer.W[j])):
                            layer.W[j][i]-=lr*deltas[l][j]*layer.input[i]
                        layer.b[j]-=lr*deltas[l][j]
            if ep%200==0: print(f"Epoch {ep}: loss={total_loss:.4f}")
if __name__=="__main__":
    random.seed(42)
    nn=NeuralNet([2,4,1])
    X=[[0,0],[0,1],[1,0],[1,1]];Y=[[0],[1],[1],[0]]
    nn.train(X,Y,lr=1.0,epochs=1000)
    for x,y in zip(X,Y):
        pred=nn.forward(x);print(f"{x} -> {pred[0]:.3f} (expected {y[0]})")
    print("Neural net OK")
