
def variations(n1, n2):
    return n1 ** n2

def create_table(n1, n2, variations):
    listx = []
    variables = n2
    integers = []
    permutations = n1 ** n2
#---------------------------------------------
    permutations2 = permutations
    while permutations2 > 1:   
        permutations2 //= n1
        integers.append(int(permutations2))
#---------------------------------------------
    for B_1 in range(0, permutations):
        listx.append([])
#-----------------------------------------------------------------

    I_integers = iter(integers)
    integers_reverse = reversed(integers)
    I_integers_reverse = iter(list(integers_reverse))
        
    for i in range(variables):
        
        A = next(I_integers)
        B = next(I_integers_reverse) 

        X = "".join(list(map(lambda num: num * A , variations)))
        X2 = X * B
        I_X2 = iter(X2)
        Y = list(map(lambda listx: listx.append(next(I_X2)), listx))
    
#-------------------------------------------------------------------        
    return listx
# variations = "01"  
# print(create_table(2,3,variations))
