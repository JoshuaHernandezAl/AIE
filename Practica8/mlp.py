import numpy as np
from random import random
import tkinter as tk
from tkinter import *
from tkinter import ttk, messagebox
import os
from file import data, expected_data
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
            # activations=self._sigmoid(net_inputs)
            activations=self._tanh(net_inputs)
            self.activations[i+1]=activations
        return activations
    def back_propagate(self,error,flag=False):
        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):
            activations=self.activations[i+1]
            #! For iris plant may use tanh function
            delta=error*self._tanh_derivate(activations)
            delta_reshaped=delta.reshape(delta.shape[0],-1).T
            current_activation=self.activations[i]
            current_activation_reshaped=current_activation.reshape(current_activation.shape[0],-1)
            self.derivatives[i]=np.dot(current_activation_reshaped,delta_reshaped)
            error=np.dot(delta,self.weights[i].T)            

    def gradient_descent(self,learning_rate):
        for i in range(len(self.weights)):
            weights=self.weights[i]
            derivatives=self.derivatives[i]
            weights+=derivatives*learning_rate
    def train(self,inputs,targets,epochs,learning_rate,targetError):
        current_epoch=0
        for i in range(epochs):
            sum_error=0
            for input,target in zip(inputs,targets):
                output=self.forward_propagate(input)
                error=target-output
                self.back_propagate(error)
                self.gradient_descent(learning_rate)
                sum_error+=self._mse(target,output)
            if i%100==0:
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

# Functions
def genTable():
    def reset():
        frame.destroy()
        btn_reset.destroy()
    rows=int(entry_hidden.get())
    frame = Frame(mainScreen, width=250, height=300)
    frame.grid(row=1,column=2)
    frame.config(bg="lightblue")
    frame.config(bd=25)
    btn_reset=ttk.Button(mainScreen, text="Reset",command=reset)
    btn_reset.grid(row=2,column=2,padx=15,pady=15)
    global entries 
    entries= []
    for i in range(rows): #Se crea la tabla de pesos para wi
        entries.append(Entry(frame, width=5, font=('Arial',16,'bold'),justify=tk.CENTER))
        entries[i].grid(row=i, column=0) 
    
def backpropagation():
    num_neurons=[]
    for i in range(len(entries)):
        num_neurons.append(int(entries[i].get()))
    epochs =int(entry_epochs.get())
    learning_rate=float(entry_learning_rate.get())
    target_error =float(entry_target_error.get())
    mlp=MLP(4,num_neurons,1)
    inputs=data
    targets=expected_data
    msg=mlp.train(inputs, targets, epochs, learning_rate,target_error)
    # get a prediction
    output = mlp.forward_propagate(inputs)
    print()
    back_paint=tk.Tk()
    back_paint.title('Iris Plant')
    back_paint.geometry('300x300')

    paint(150,6,back_paint,inputs,output,targets)
    messagebox.showinfo(title="Termini por:",message=msg)
    back_paint.mainloop()
def paint(rows,cols,root,inputs,targets,outputs):
        file = open("results.txt", "w")
        head=""
    
        for j in range(cols): #se pintan las cabeceras de la tabla
            if j==cols-1:
                e = Entry(root, width=5, fg='blue',font=('Arial',10,'bold')) 
                e.grid(row=0, column=j+1) 
                e.insert(END, 'Yesp') 
                e.config(state=DISABLED)
                head+="Yesp\t\t|"
            elif j==cols-2:
                e = Entry(root, width=5, fg='blue',font=('Arial',10,'bold')) 
                e.grid(row=0, column=j+1) 
                e.insert(END, 'Y') 
                e.config(state=DISABLED)
                head+="Y\t\t|"
            else:
                e = Entry(root, width=5, fg='blue',font=('Arial',10,'bold')) 
                e.grid(row=0, column=j+1) 
                e.insert(END, "X"+str(j)) 
                e.config(state=DISABLED)
                head+="X"+str(j)+"\t\t|"
        file.write(head + os.linesep)
        content=""
        for i in range(rows): #se pinta el contenido de las tablas
            content=""
            for j in range(cols):
                if j==cols-1:
                    e = Entry(root, width=5, fg='blue',font=('Arial',10,'bold')) 
                    e.grid(row=i+1, column=j+1) 
                    e.insert(END, outputs[i][0]) 
                    e.config(state=DISABLED)
                    content+=str(round(outputs[i][0], 2))+"\t|"
                elif j==cols-2:
                    e = Entry(root, width=5, fg='blue',font=('Arial',10,'bold')) 
                    e.grid(row=i+1, column=j+1) 
                    e.insert(END, targets[i][0]) 
                    e.config(state=DISABLED)
                    content+=str(round(targets[i][0],2))+"\t|"
                else: 
                    e = Entry(root, width=5, fg='blue',font=('Arial',10,'bold')) 
                    e.grid(row=i+1, column=j+1) 
                    e.insert(END, inputs[i][j]) 
                    e.config(state=DISABLED)
                    content+=str(round(inputs[i][j],2))+"\t|"
            file.write(content + os.linesep)
        file.close()
        os.system('code results.txt')
def cover():
    messagebox.showinfo(title="Elaborado por:",message="Hernandez Almaraz Joshua \n Martínez Martínez Isaac Eduardo ")
# Main program
if __name__ =="__main__":
    mainScreen=tk.Tk()

    mainScreen.title('Iris Plant')
    mainScreen.geometry('700x500')

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
    entry_hidden_value=tk.StringVar(value="2")
    entry_hidden=ttk.Entry(mainScreen,font = "Helvetica 22 bold",width=2,justify=tk.CENTER,textvariable=entry_hidden_value)
    entry_hidden.grid(row=0, column=1)
    btn_hidden=ttk.Button(mainScreen, text="Generar ocultas",command=genTable)
    btn_hidden.grid(row=0,column=2,padx=15,pady=15)
    
    lbl_epochs=tk.Label(mainScreen,text="Epocas", justify=tk.CENTER)
    lbl_epochs.grid(row=1,column=0)
    entry_epochs_value=tk.StringVar(value="1000")
    entry_epochs=ttk.Entry(mainScreen,font = "Helvetica 22 bold",width=10,justify=tk.CENTER,textvariable=entry_epochs_value)
    entry_epochs.grid(row=1, column=1)
    
    lbl_learning_rate=tk.Label(mainScreen,text="Razon aprendizaje", justify=tk.CENTER)
    lbl_learning_rate.grid(row=2,column=0)
    entry_learning_rate_value=tk.StringVar(value="0.5")
    entry_learning_rate=ttk.Entry(mainScreen,font = "Helvetica 22 bold",width=10,justify=tk.CENTER,textvariable=entry_learning_rate_value)
    entry_learning_rate.grid(row=2, column=1)
    
    lbl_target_error=tk.Label(mainScreen,text="Error", justify=tk.CENTER)
    lbl_target_error.grid(row=3,column=0)
    entry_target_error_value=tk.StringVar(value="0.01")
    entry_target_error=ttk.Entry(mainScreen,font = "Helvetica 22 bold",width=10,justify=tk.CENTER,textvariable=entry_target_error_value)
    entry_target_error.grid(row=3, column=1)
    
    btn_backpropagation=ttk.Button(mainScreen, text="Backpropagation",command=backpropagation)
    btn_backpropagation.grid(row=4,column=1,padx=15,pady=15)
    btn_cover=ttk.Button(mainScreen, text="Portada",command=cover)
    btn_cover.grid(row=4,column=0,padx=15,pady=15)
    mainScreen.mainloop()
    