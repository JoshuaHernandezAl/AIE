import numpy as np
from random import random
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
            if((sum_error/len(inputs))<targetError):
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
        return (1-np.exp(-x))/(1+np.exp(-x))
    def _tanh_derivate(self,x):
        return (2*np.exp(x))/((np.exp(x)+1)**2)
if __name__ =="__main__":
    # num_neurons=[]
    # hidden =int(input("Capas ocultas:"))
    # for i in range(hidden):
    #     layer_neurons = int(input("Cantidad de neuronas: "))
    #     num_neurons.append(layer_neurons)
    # epochs =int(input("Epocas:"))
    # learning_rate=float(input("Razon de aprendizaje:"))
    # targetError =float(input("Error:"))
    mlp=MLP(4,[6,7],1)
    inputs=data
    targets=expected_data
    
    msg=mlp.train(inputs, targets, 10000, 0.5,0.05)


    # create dummy data
    #5.1,3.5,1.4,0.2,Iris-setosa
    # 7.0,3.2,4.7,1.4,Iris-versicolor
    # 6.4,3.1,5.5,1.8,Iris-virginica
    # input = np.array([0.51,0.35,0.14,0.02])
    # input = np.array([0.7,0.32,0.47,0.14])
    input = np.array([0.64,0.31,0.55,0.18])
    # target = np.array([1])
    # target = np.array([0])
    target = np.array([-1])

    # get a prediction
    output = mlp.forward_propagate(input)
    print()
    # print("El programa termino por: {}".format(msg))
    if round(output[0])==1:
        out="Iris-setosa"
    elif round(output[0])==0:
        out="Iris-versicolor"
    elif round(output[0])==-1:
        out="Iris-virginica"
    print("Our network believes that {} , {}, {} , {} is equal to {}/{} ".format(input[0], input[1],input[2],input[3], output[0],out))
