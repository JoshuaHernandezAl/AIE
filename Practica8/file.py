import numpy as np


data=[]
expected_data=[]
with open("iris.data") as file:
    for line in file:
        aux=line.split(",")
        aux[-1]=aux[-1].rstrip()
        data.append(aux[:-1])
        expected_data.append([aux[-1]])
file.close()
for i,element in enumerate(expected_data):
    # Setosa 0
    # Versicolor 0.5
    # virginica 1
    if element[0]=="Iris-setosa":
        expected_data[i]=[1]
    elif element[0]=="Iris-versicolor":
        expected_data[i]=[0]
    elif element[0]=="Iris-virginica":
        expected_data[i]=[-1]
for i in range(len(data)):
    aux=data[i]
    aux[0]=float(aux[0])/10
    aux[1]=float(aux[1])/10
    aux[2]=float(aux[2])/10
    aux[3]=float(aux[3])/10
    data[i]=aux
data=np.array(data)
#! here is the mistake, size of the matrix are not equal eachother
expected_data=np.array(expected_data, ndmin=2)

