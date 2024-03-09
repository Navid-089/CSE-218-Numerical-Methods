import numpy as np
import matplotlib.pyplot as plt
import math

def getLinearValue(a0,a1,x):
    return a0+a1*x

def linearRegression(x_arr,y_arr):
    dim = len(x_arr)
    x_sum = 0
    y_sum = 0
    xy_sum = 0
    xs_sum = 0
    for i in range(0,dim):
        x_sum = x_sum + x_arr[i]
        y_sum = y_sum + y_arr[i]
        xy_sum = xy_sum + x_arr[i] * y_arr[i]
        xs_sum = xs_sum + (x_arr[i] ** 2)

    a1 = (dim * xy_sum - x_sum * y_sum) / (dim * xs_sum - x_sum ** 2)
    a0 = (y_sum - a1 * x_sum) / dim

    return (a0,a1)

def getExponentialValue(a0,a1,x):
    return a0 * math.exp(a1*x)

def exponentialRegression(x_arr,y_arr2):
    dim = len(x_arr)
    y_arr = np.zeros([dim],dtype=float)
    for i in range(dim):
        y_arr[i] = np.log(y_arr2[i])
    x_sum = 0
    y_sum = 0
    xy_sum = 0
    xs_sum = 0
    for i in range(0,dim):
        x_sum = x_sum + x_arr[i]
        y_sum = y_sum + y_arr[i]
        xy_sum = xy_sum + x_arr[i] * y_arr[i]
        xs_sum = xs_sum + (x_arr[i] ** 2)

    a1 = (dim * xy_sum - x_sum * y_sum) / (dim * xs_sum - x_sum ** 2)
    a0_tmp = (y_sum - a1 * x_sum) / dim
    a0 = math.exp(a0_tmp)
    return (a0,a1)

def getPolynomialValue(x, constants):
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

def polynomialRegression(x_arr,y_arr,order):
    dim = len(y_arr)

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

def getPowerValue(a0,a1,x):
    return a0 * (x ** a1)

def powerModelRegression(x_arr2,y_arr2):
    dim = len(x_arr2)
    y_arr = np.zeros([dim],dtype=float)
    x_arr = np.zeros([dim], dtype=float)
    for i in range(dim):
        y_arr[i] = np.log(y_arr2[i])
        x_arr[i] = np.log(x_arr2[i])
    x_sum = 0
    y_sum = 0
    xy_sum = 0
    xs_sum = 0
    for i in range(0,dim):
        x_sum = x_sum + x_arr[i]
        y_sum = y_sum + y_arr[i]
        xy_sum = xy_sum + x_arr[i] * y_arr[i]
        xs_sum = xs_sum + (x_arr[i] ** 2)

    a1 = (dim * xy_sum - x_sum * y_sum) / (dim * xs_sum - x_sum ** 2)
    a0_tmp = (y_sum - a1 * x_sum) / dim
    a0 = math.exp(a0_tmp)
    return (a0,a1)

###MAIN##
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
# x_ini = np.array(x1)
# y_ini = np.array(y1)

# data = np.genfromtxt("dissolveO2.csv", delimiter=",", skip_header=1)
# x_ini = data[:, 0]
# y_ini= data[:, 1]


#POWER MODEL EXCEPTION
x_ini = np.array([1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000],dtype=np.float64)
y_ini = np.array([10.3,13.5,13.9,14.2,11.6,10.3,9.7,9.6,14.1,19.8,31.1],dtype=np.float64)
x = x_ini[x_ini>0]
y = np.zeros([len(x)],dtype=float)
for i in range(len(x)):
    for j in range(len(x_ini)):
        if(x_ini[j] == x[i]):
            y[i] = y_ini[j]
print("Please input your order of Polynomial Regression:")
order= int(input())
a0_linear,a1_linear = (linearRegression(x,y))
print("--- LINEAR REGRESSION ---")
print("a0 = ",a0_linear,", a1 = ",a1_linear)
print("\n--- EXPONENTIAL REGRESSION ---")
a0_exponential,a1_exponential = exponentialRegression(x,y)
print("a = ",a0_exponential,", b = ",a1_exponential)
print("\n---POWER MODEL ---")
a0_power,a1_power= powerModelRegression(x,y)
print("a = ",a0_power,", b = ",a1_power)
print("\n---POLYNOMIAL MODEL --- ")
consts = (polynomialRegression(x,y,order))
print("--- Constants --- ")
for i in range(order+1):
    print("a",i,"=",consts[i])
print("----------")


divisions = 100
x_start = min(x)
x_end = max(x)
x_graph = np.linspace(x_start,x_end,divisions)
y_graph_linear = np.zeros([divisions],dtype=float)
y_graph_exponential = np.zeros([divisions],dtype=float)
y_graph_polynomial = np.zeros([divisions],dtype=float)
y_graph_power = np.zeros([divisions],dtype=float)
for i in range(divisions):
    y_graph_linear[i] = getLinearValue(a0_linear,a1_linear,x_graph[i])
    y_graph_power[i] = getPowerValue(a0_power,a1_power,x_graph[i])
    y_graph_exponential[i] = getExponentialValue(a0_exponential,a1_exponential,x_graph[i])
    y_graph_polynomial[i] = getPolynomialValue(x_graph[i],consts)

plt.plot(x_graph,y_graph_linear,label ='$linear$', color='g',linewidth= 1)
plt.plot(x_graph,y_graph_power,label ='$power$', color='y',linewidth= 1)
plt.plot(x_graph,y_graph_exponential,label ='$exp$', color='b',linewidth= 1)
plt.plot(x_graph,y_graph_polynomial,label ='$poly$', color='m',linewidth= 1)
plt.plot(x,y,'ro')
plt.legend(loc="best")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()






