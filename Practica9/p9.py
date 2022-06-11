import numpy as np
from random import random
from file import data, expected_data
from functions import *
import tkinter as tk
from tkinter import *
from tkinter import ttk, messagebox
import os

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
    def _tanh(self,x):
        #return (1-np.exp(-x))/(1+np.exp(-x))
        return np.tanh(x)
    def _tanh_derivate(self,x):
        #return (2*np.exp(x))/((np.exp(x)+1)**2)
        return 1-(np.tanh(x)**2)

#functions
def backpropagation():
    mlp=MLP(6,[7,8],1)

    inputs = data
    targets = expected_data 
    
    
    msg=mlp.train(inputs, targets, 50000, 0.1,0.05)

    # get a prediction
    output = mlp.forward_propagate(inputs)
    
    efficiency=0
    for i in range(len(targets)):
        aux=round(output[i][0])
        if aux==targets[i][0]:
            efficiency+=1
    efficiency=(efficiency/len(targets))*100

    print(msg)    
    print("Efficiency: {}%".format(efficiency))
    paint(143,8,inputs, targets, output)
    lbl_efficiency=tk.Label(mainScreen,text="Eficiencia: {}".format(efficiency), justify=tk.CENTER)
    lbl_efficiency.grid(row=0,column=3)

def cover():
    messagebox.showinfo(title="Elaborado por:",message="Hernandez Almaraz Joshua \n Martínez Martínez Isaac Eduardo ")


if __name__ =="__main__":
    mainScreen=tk.Tk()

    mainScreen.title('Iris Plant')
    mainScreen.geometry('600x600')

    mainScreen.rowconfigure(0,weight=1)
    mainScreen.rowconfigure(1,weight=1)
    mainScreen.rowconfigure(2,weight=1)
    mainScreen.rowconfigure(3,weight=1)
    mainScreen.rowconfigure(4,weight=1)
    mainScreen.columnconfigure(0,weight=1)
    mainScreen.columnconfigure(1,weight=1)
    mainScreen.columnconfigure(2,weight=1)
    mainScreen.columnconfigure(3,weight=1)

    lbl_hidden=tk.Label(mainScreen,text="Capas ocultas", justify=tk.CENTER)
    lbl_hidden.grid(row=0,column=0)
    lbl_hidden_value=tk.Label(mainScreen,text="2", justify=tk.CENTER)
    lbl_hidden_value.grid(row=0,column=1)
    
    lbl_neurons=tk.Label(mainScreen,text="Cantidad neuronas", justify=tk.CENTER)
    lbl_neurons.grid(row=1,column=0)
    lbl_neurons_value=tk.Label(mainScreen,text="7,8", justify=tk.CENTER)
    lbl_neurons_value.grid(row=1,column=1)
    
    lbl_epochs=tk.Label(mainScreen,text="Epocas", justify=tk.CENTER)
    lbl_epochs.grid(row=2,column=0)
    lbl_epochs_value=tk.Label(mainScreen,text="50000", justify=tk.CENTER)
    lbl_epochs_value.grid(row=2,column=1)
    
    
    lbl_learning_rate=tk.Label(mainScreen,text="Razon aprendizaje", justify=tk.CENTER)
    lbl_learning_rate.grid(row=3,column=0)
    lbl_learning_rate_value=tk.Label(mainScreen,text="0.1", justify=tk.CENTER)
    lbl_learning_rate_value.grid(row=3,column=1)
    
    
    lbl_target_error=tk.Label(mainScreen,text="Error", justify=tk.CENTER)
    lbl_target_error.grid(row=4,column=0)
    lbl_target_error_value=tk.Label(mainScreen,text="0.05", justify=tk.CENTER)
    lbl_target_error_value.grid(row=4,column=1)

    btn_backpropagation=ttk.Button(mainScreen, text="Backpropagation",command=backpropagation)
    btn_backpropagation.grid(row=5,column=1,padx=15,pady=15)
    btn_cover=ttk.Button(mainScreen, text="Portada",command=cover)
    btn_cover.grid(row=5,column=0,padx=15,pady=15)
    
    mainScreen.mainloop()
