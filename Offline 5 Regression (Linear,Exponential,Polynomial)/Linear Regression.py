import numpy as np
import matplotlib.pyplot as plt

def getValue(a0,a1,x):
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

x = np.array([2,3,5,7,8,10,12,13,15,16])
y = np.array([4.7,6.9,9.8,12.6,14.3,16.9,19.6,21.2,23.8,25.4])
a0,a1 = (linearRegression(x,y))
print("a0 = ",a0,", a1 = ",a1)
# DON'T FORGET TO FIND OUT THE VALUES OF A AND B
divisions = 100
x_start = min(x)
x_end = max(x)
x_graph = np.linspace(x_start,x_end,divisions)
y_graph = np.zeros([divisions],dtype=float)
#DON'T FORGET TO PUT THE VALUES OF A AND B HERE INSTEAD OF A0,A1 IF CHANGED
for i in range(divisions):
    y_graph[i] = getValue(a0,a1,x_graph[i])
plt.close()
plt.plot(x_graph,y_graph,label ='$f(x)$', color='r',linewidth= 2)
plt.plot(x,y,'go')
plt.legend(loc="best")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()








