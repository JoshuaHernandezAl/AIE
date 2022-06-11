import numpy as np
from random import random
from file import data, expected_data
from functions import *

class MLP:
    def __init__(self,inputs=3,hiddenLayers=[3,5],outputs=2):
        self.num_inputs=inputs
        self.num_hiddenLayers=hiddenLayers
        self.num_outputs=outputs
        
        layers=[self.num_inputs]+self.num_hiddenLayers+[self.num_outputs]

        #random weights
        self.weights=[]
        for i in range(len(layers)-1):
            w=np.random.rand(layers[i],layers[i+1])
            self.weights.append(w)

        activations=[]
        for i in range(len(layers)):
            a=np.zeros(layers[i])
            activations.append(a)
        self.activations=activations

        derivatives=[]
        for i in range(len(layers)-1):
            d=np.zeros((layers[i],layers[i+1]))
            derivatives.append(d)
        self.derivatives=derivatives


    def forward_propagate(self,inputs):
        activations=inputs
        self.activations[0]=inputs
        for i,w in enumerate(self.weights):
            #calculate inputs
            net_inputs=np.dot(activations,w)
            #caculate activations 
            #! For iris plant may use tanh function
            activations=self._sigmoid(net_inputs)
            self.activations[i+1]=activations
        return activations
    def back_propagate(self,error,flag=False):
        for i in reversed(range(len(self.derivatives))):
            activations=self.activations[i+1]
            #! For iris plant may use tanh function
            delta=error*self._sigmoid_derivative(activations)
            delta_reshaped=delta.reshape(delta.shape[0],-1).T
            current_activation=self.activations[i]
            current_activation_reshaped=current_activation.reshape(current_activation.shape[0],-1)
            self.derivatives[i]=np.dot(current_activation_reshaped,delta_reshaped)
            error=np.dot(delta,self.weights[i].T)            
            if flag:
                print("Derivatives for W{}={}".format(i,self.derivatives[i]))
        

    def gradient_descent(self,learning_rate):
        for i in range(len(self.weights)):
            weights=self.weights[i]
            derivatives=self.derivatives[i]
            weights+=derivatives*learning_rate
    def train(self,inputs,targets,epochs,learning_rate,targetError):
        for i in range(epochs):
            sum_error=0
            for input,target in zip(inputs,targets):
                output=self.forward_propagate(input)
                error=target-output
                self.back_propagate(error)
                self.gradient_descent(learning_rate)
                sum_error+=self._mse(target,output)
            if i%1000==0:
                print("Error: {} at epoch {}".format(sum_error/len(inputs),i))
            if((sum_error/len(inputs))<=targetError):
                return "Por error, en la epoca "+str(i)
            current_epoch=i
        return "Por epocas, en la epoca "+str(current_epoch)

    def _sigmoid_derivative(self,x):
        return x*(1.0-x)

    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def _mse(self,target,output):
        return np.average((target-output)**2)

if __name__ =="__main__":
    mlp=MLP(6,[5],1)

    inputs = data
    targets = expected_data 
    
    
    msg=mlp.train(inputs, targets, 5000, 0.1,0.01)


    # get a prediction
    output = mlp.forward_propagate(inputs)
    
    efficiency=0
    for i in range(len(targets)):
        aux=round(output[i][0])
        if aux==targets[i]:
            efficiency+=1
    efficiency=(efficiency/len(targets))*100

    print(msg)    
    print("Efficiency: {}%".format(efficiency))
