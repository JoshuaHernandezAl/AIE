import os
def paint(rows,cols,inputs,targets,outputs):
        file = open("results.txt", "w")
        head=""
    
        for j in range(cols): #se pintan las cabeceras de la tabla
            if j==cols-1:
                head+="Y|"
            elif j==cols-2:
                head+="Yesp\t|"
            else:
                if j==0:
                    head+="X"+str(j)+"    \t|"
                else:
                    head+="X"+str(j)+"  \t\t|"
        file.write(head + os.linesep)
        content=""
        for i in range(rows): #se pinta el contenido de las tablas
            content=""
            for j in range(cols):
                if j==cols-1:
                    content+=str(round(round(outputs[i][0]), 2))+"|\t"
                elif j==cols-2:
                    content+=str(round(targets[i][0],2))+"\t\t|"
                else: 
                    content+=str(round(inputs[i][j],2))+"\t\t|"
            file.write(content + os.linesep)
        file.close()
        # os.system('code results.txt')