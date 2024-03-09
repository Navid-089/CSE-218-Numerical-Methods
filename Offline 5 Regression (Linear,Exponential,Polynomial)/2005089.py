import math
import matplotlib.pyplot as plt
import numpy as np

# function is xy = c + ax + bx^2

def getValue(x, constants):
    sum = 0
    length = len(constants)
    for i in range(0,length):
        sum = sum + constants[i] * (x ** i)
    return sum

def GaussianElimination(A, B, pivot, showall):
    dim = len(B)
    C = np.hstack((A, B))

    for i in range(0, dim):

        i1 = i + 1

        if (pivot == True):
            max = abs(C[i][i])
            maxr = i
            for j in range(i + 1, dim):
                if ((abs(C[j][i])) > max):
                    max = abs(C[j][i])
                    maxr = j
            if(i!=maxr):
                C[[i, maxr]] = C[[maxr, i]]

        while (i1 < dim):
            divisor = float(-C[i1][i] / C[i][i])
            for i2 in range(i, dim + 1):
                C[i1][i2] = C[i1][i2] + divisor * C[i][i2]


            if (showall == True):
                E = C[0:dim , 0:dim ]
                F = C[0:dim , dim:dim+1]

                print("Coefficient matrix is:")
                print(np.round(E,4))
                print("Constant Matrix is:")
                print(np.round(F,4))
            i1 = i1 + 1

    D = np.zeros([dim, dim], dtype=float)
    if (showall == True):
        for it1 in range(0, dim):
            for it2 in range(0, dim):
                D[it1][it2] = C[it1][it2]
        sum = 1
        for it3 in range(0,dim):
            sum = sum * D[it3][it3]

        print()
        #print("**The determinant of the Co-efficient matrix is : ", ("%.4f" % float(np.linalg.det(D))))
        print("##The determinant of the Co-efficient matrix is : ", ("%.4f" % float(sum)))

    G = np.zeros([dim],dtype=float)
    FF = C[0:dim, 0:dim]
    for it4 in range(0, dim):
        for it5 in range(0,dim):
            G[it5] = FF[it4][it5]
        for it in range(0,dim):
                if(abs(G[it]) < 0.0000000000001):
                    G[it] = 0
        if(np.all((G == 0))):

            print("Infinite Solutions.")
            exit()
    solve =np.zeros([dim,1],dtype=float)
    temp = dim - 1
    while temp > -1:
        solve[temp][0] = C[temp][dim] / C[temp][temp]
        for it in range(temp - 1, -1, -1):
            C[it][dim] = C[it][dim] - solve[temp][0] * C[it][temp]
        temp = temp - 1

    for i in range(0,dim):
        if(abs(solve[i][0]) < 0.0000000000001):
            solve[i][0] = 0
    return solve

def polynomialRegression(x_arr,y_arr2,order):
    dim = len(y_arr2)
    y_arr = np.zeros([dim],dtype=np.float64)
    for i in range(dim):
        y_arr[i] = x_arr[i] * y_arr2[i]

    a = np.zeros([order+1,order+1],dtype=np.float64)
    c = np.zeros([order+1,1],dtype=np.float64)
    for i in range(order + 1):
        tmp_sum = 0
        for j in range(dim):
            tmp_sum = tmp_sum + y_arr[j] * (pow(x_arr[j],i))
        c[i][0] = tmp_sum

    # print(c)

    for i in range(0,order+1):
        for j in range(0,order+1):
            tmp_sum = 0
            for k in range(0,dim):
                tmp_sum = tmp_sum + pow(x_arr[k],(i+j))

            a[i][j] = tmp_sum
    # print(a)
    tmp_constants = GaussianElimination(a,c,True,False)
    return tmp_constants

##MAIN##
## IF FILE I/O IS NEEDED
# x1 = list()
# y1 = list()
# with open("gene.txt") as f:
#     lines =f.readlines()
#     for line in lines:
#         words = line.split()
#         x1.append(float(words[0]))
#         y1.append(float(words[1]))
#     f.close()
# x = np.array(x1)
# y = np.array(y1)

# data = np.genfromtxt("dissolveO2.csv", delimiter=",", skip_header=1)
# x = data[:, 0]
# y= data[:, 1]

x = np.array([1.0,2.0,3.0,4.0,5.0],dtype=np.float64)
y = np.array([2.2,2.8,3.6,4.5,5.5],dtype=np.float64)
print("Please input the order of Polynomial Regression: ")
order = 2

consts = (polynomialRegression(x,y,order))
print("--- Constants --- ")
print("c = ",consts[0])
print("a = ",consts[1])
print("b = ",consts[2])

print("----------")
divisions = 100
x_start = min(x)-0.5
x_end = 6
x_graph = np.linspace(x_start,x_end,divisions)
y_graph2 = np.zeros([divisions],dtype=float)
for i in range(divisions):
    y_graph2[i] = getValue(x_graph[i],consts)

y_graph = np.zeros([divisions],dtype=float)
for i in range(divisions):
    y_graph[i] = y_graph2[i]/x_graph[i]

x_find = np.array([2.5,5.5])
y_find= np.zeros([2],dtype=float)
y_find_final= np.zeros([2],dtype=float)
y_find[0] = getValue(x_find[0],consts)
y_find[1] = getValue(x_find[1],consts)

for i in range(2):
    y_find_final[i] = y_find[i]/x_find[i]

for i in range(2):
    print("For ",x_find[i],", y= ",y_find_final[i])

plt.plot(x_graph,y_graph,label ='$f(x)$', color='r',linewidth= 2)
plt.plot(x,y,'go')
plt.plot(x_find,y_find_final,'b*')
plt.legend(loc="best")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()






