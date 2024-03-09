import math
import numpy as np
import matplotlib.pyplot as plt

def getValue(a0,a1,x):
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

x_ini = np.array([1,2,3,4,5,6,7,8,9,10])
y_ini = np.array([2,4.5,8,12.5,18,24.5,32,40.5,50,60.5])
x = x_ini[x_ini>0]
y = np.zeros([len(x)],dtype=float)
for i in range(len(x)):
    for j in range(len(x_ini)):
        if(x_ini[j] == x[i]):
            y[i] = y_ini[j]

a0,a1 = (powerModelRegression(x,y))
print("a = ",a0," b = ",a1)
divisions = 100
x_start = min(x)
x_end = max(x)
x_graph = np.linspace(x_start,x_end,divisions)
y_graph = np.zeros([divisions],dtype=float)
for i in range(divisions):
    y_graph[i] = getValue(a0,a1,x_graph[i])
plt.close()
plt.plot(x_graph,y_graph,label ='$f(x)$', color='r',linewidth= 2)
plt.plot(x,y,'go')
plt.legend(loc="best")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()








