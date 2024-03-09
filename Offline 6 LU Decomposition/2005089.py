import numpy as np

def LU_decomposition(a,b):
    dim = len(b)
    l = np.zeros([dim,dim],dtype=float)
    u = np.zeros([dim,dim],dtype=float)

    for i in range(0,dim):
        for j in range(0,dim):
            if (a[i][j] == 0):
                a[i][j] = 0.0000000000000000000001
            u[i][j] = a[i][j]

    for i in range(0,dim):
        i1 = i+1
        while(i1 < dim):
            divisor = u[i1][i]/ u[i][i]
            for i2 in range(i,dim):
                u[i1][i2] = u[i1][i2] - divisor * u[i][i2]
                l[i1][i2] = divisor
            i1 = i1+1

    for i in range(0, dim):
        for j in range(0, dim):
            if(j == i):
                l[i][i] = 1
            elif(j>i):
                l[i][j] = 0

    print("\nL matrix:\n", l)
    print("\nU matrix:\n", u)

    tmp = np.zeros([dim],dtype=float)
    for i in range(dim):
        for it in range(dim):
            tmp[it] = u[i][it]
        if(np.all(tmp == 0)):
            print("Cannot solve, infinite solutions.")
            exit()
        for it in range(dim):
            tmp[it] = l[i][it]
        if (np.all(tmp == 0)):
            print("Cannot solve, infinite solutions.")
            exit()

    z = np.zeros([dim,1],float)
    b_tmp = np.zeros([dim,1],float)
    for i in range(0,dim):
        b_tmp[i][0] = b[i][0]

    for i in range(0,dim):
        z[i][0] = b_tmp[i][0]
        for j in range(i+1,dim):
            b_tmp[j][0] = b_tmp[j][0] - z[i][0] * l[j][i]
    print("\nZ vector:\n",z)
    x = np.zeros([dim, 1], dtype=float)
    temp = dim - 1
    while temp > -1:
        x[temp][0] = z[temp][0] / u[temp][temp]
        for it in range(temp - 1, -1, -1):
            z[it][0] = z[it][0] - x[temp][0] * u[it][temp]
        temp = temp - 1
    return x

print("Please input the number of equations: ")
num = int(input())
if(num <= 0):
    print("No input's given.")
    exit()
a = np.zeros([num, num], dtype=float)
b = np.zeros([num, 1], dtype=float)
B = np.zeros([num],dtype=float)
print("Please input the coefficients in a row separated by a Space:")
for i in range(0, num):
    string = input()
    tmp_arr = string.split(" ")
    if(len(tmp_arr) != num):
        print("Invalid number of inputs.")
        exit()
    for k in range(0, num):
        a[i][k] = tmp_arr[k]
print("Please input the constants in a column separated by an Enter:")
for i in range(0, num):
    b[i][0] = float(input())
    B[i] = b[i][0]
if(np.all(B==0)):
    print("NO SOLUTIONS!")
    exit()
x = np.array([num],dtype=float)
x = LU_decomposition(a,b)
print("\n- - - SOLUTION - - - \n")
for i in range (0,num):
    print("x",i+1,"=",np.round(x[i][0],4))


