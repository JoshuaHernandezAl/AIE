import numpy as np


data=[]
expected_data=[]
with open("lenses.data") as file:
    for line in file:
        aux=line.split(",")
        aux[-1]=aux[-1].rstrip()
        data.append(aux[:-1])
        expected_data.append([aux[-1]])
file.close()

data=np.array(data)
data=data.astype(int)
# for i in range(len(data)):
#     aux=data[i]
#     aux[0]=float(aux[0])/10
#     aux[1]=float(aux[1])/10
#     aux[2]=float(aux[2])/10
#     aux[3]=float(aux[3])/10
#     data[i]=aux

expected_data=np.array(expected_data, ndmin=2)
expected_data=expected_data.astype(int)
