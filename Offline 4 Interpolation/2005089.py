import numpy as np
import matplotlib.pyplot as plt

def array_reform(x_values, y_values):
    length = len(y_values)
    for i in range(1,length):
        for j in range(0,length-i):
            y_values[j][i] = (y_values[j+1][i-1] - y_values[j][i-1]) / (x_values[j+i] - x_values[j])

def get_closest_values(x_values, y_values, guess, order):
    length = len(x_values)
    diff = np.zeros([length],dtype=float)
    x_prime = np.zeros([length],dtype = float)

    for i in range(0,length):
        diff[i] = abs(guess-x_values[i])
        x_prime[i] = x_values[i]




    for i in range(0, length):
        for j in range(i + 1, length):
            if ((diff[j]) < (diff[i])):
                tmp = x_prime[j]
                x_prime[j] = x_prime[i]
                x_prime[i] = tmp
                tmp = diff[j]
                diff[j] = diff[i]
                diff[i] = tmp

    return x_prime[0:order+1]

def n_variable_interpolation(x_values, y_values, guess,order) :
    sum = 0
    it = 0
    while it < order+1:
        tmp = y_values[0][it]
        for it2 in range(0,it):
            tmp = tmp* (guess- x_values[it2])
        print("coefficient ",it,"=",tmp)
        sum = sum + tmp
        it = it+1
    return sum

def l_variable_interpolation(x_values, y_values, guess, order):
    l = np.ones([order+1],dtype=float)
    for i in range(0,order+1):
        numerator = 1
        denominator = 1
        j = 0
        while(j < order+1):
            if(j !=i):
                numerator = numerator * (guess - x_values[j])
                denominator = denominator * (x_values[i] - x_values[j])
            j=j+1

        l[i] = float(numerator/denominator)
    sum = 0
    for i in range(order+1):

        sum = sum + (y_values[i][0] * l[i])
    return sum

# def l_linear_interpolation(x_values, y_values, guess):
#     place = -1
#     length = len(x_values)
#     for i in range(0, length - 1):
#         if (x_values[i] < guess and x_values[i + 1] > guess):
#             place = i
#     l0 = (guess-x_values[place+1])/(x_values[place]-x_values[place+1])
#     l1 = (guess-x_values[place])/(x_values[place+1]-x_values[place])
#     return (l0*y_values[place][0]+l1*y_values[place+1][0])


##MAIN##

# IF FILE I/O IS NEEDED from a TXT file
x1 = list()
y1 = list()
with open("datapoints.txt") as f:
    lines =f.readlines()
    for line_no,line in enumerate(lines):
        if(line_no == 0):
            pass
        else:
            words = line.split()
            x1.append(float(words[0]))
            y1.append(float(words[1]))
    f.close()
x_ini = np.array(x1)
y_ini = np.array(y1)


# # IF FILE I/O IS NEEDED from a CSV file
# data = np.genfromtxt("input.csv", delimiter=",", skip_header=0)
#
# x1 = data[:, 0]
# y1 = data[:, 1]
# x_ini = np.array(x1)
# y_ini = np.array(y1)

# x_ini = np.array([15,10,20,22.5,0,30])
# y_ini = np.array([362.78,227.04,517.35,602.97,0,901.67])
length = len(x_ini)
for i in range(0,length):
    for j in range (i+1,length):
        if((x_ini[j]) < (x_ini[i])):
            tmp = x_ini[j]
            x_ini[j] = x_ini[i]
            x_ini[i] = tmp
            tmp= y_ini[i]
            y_ini[i] = y_ini[j]
            y_ini[j] = tmp

print("Enter x: ")
guess = float(input())
for i in range(0, length):
    if (guess == x_ini[i]):
        print(y_ini[i])
        exit()
if(guess < x_ini[0] or guess > x_ini[length-1]):
    print("The point cannot be interpolated.")
    exit()

#print("\nInput the number of cases of interpolation you need : (for example, 1 if only cubic, 2 if cubic and quadric both)" )
flag = 5
orders = np.zeros([flag],dtype=int)
results = np.zeros([flag],dtype=float)
for it in range(flag):  ## if values of orders are given, put them on an array and read them ## put the errors in an array of erros then find the relative errors

    order = 2+it
    if(order < 0 or order > length):
        print("Interpolation not possible.")
        exit()
    orders[it] = order

    x_new = np.zeros([order+1],dtype=float)
    y_new = np.zeros([order+1],dtype=float)
    x_new = get_closest_values(x_ini,y_ini,guess,order)
    for i in range (order+1):
        tmp = x_new[i]
        for j in range(0,length):
            if(tmp == x_ini[j]):
                y_new[i] = y_ini[j]
    for i in range(0, order+1):
        for j in range(i + 1, order+1):
            if((x_new[j]) < (x_new[i])):
                tmp = x_new[j]
                x_new[j] = x_new[i]
                x_new[i] = tmp
                tmp= y_new[i]
                y_new[i] = y_new[j]
                y_new[j] = tmp

    y = np.zeros([order+1,order+1],dtype=float)
    for i in range(order+1):
        y[i][0] = y_new[i]

    array_reform(x_new,y)



    results[it] = n_variable_interpolation(x_new, y, guess, order)
    print("\nBy newtonian interpolation of",order,"th order, y=",n_variable_interpolation(x_new,y,guess,order))

print("\n- - - - Relative errors - - - - ")
for it in range(0,flag-1):
    relative_error = abs((results[it+1] - results[it]) * 100 / (results[it]))
    print("Relative error between orders ",orders[it],"and",orders[it+1],"is",relative_error)

# for it in range(0,length):
#     if(guess > x_ini[it] and guess < x_ini[it+1]):
#         place= it
# place = place+1
# x_ini = np.insert(x_ini,place,guess)
#
#
# y_ini= np.insert(y_ini,place,results[0])
# leng = len(y_ini)
#
# y_ini = np.delete(y_ini,place)
#
#
# y_ini = np.insert(y_ini,place,results[1])
#
# plt.plot(x_ini,y_ini,label ='$f(x)$',  color= 'r', linewidth = 2, linestyle = '-' )
# y_ini = np.delete(y_ini,place)
#
#
# y_ini = np.insert(y_ini,place,results[2])
# plt.plot(x_ini,y_ini,label ='$f(x)$',  color= 'b', linewidth = 2, linestyle = '-' )
# y_ini = np.delete(y_ini,place)
#
# y_ini = np.insert(y_ini,place,results[3])
# plt.plot(x_ini,y_ini,label ='$f(x)$',  color= 'g', linewidth = 2, linestyle = '-' )
# y_ini = np.delete(y_ini,place)
#
# y_ini = np.insert(y_ini,place,results[4])
# plt.plot(x_ini,y_ini,label ='$f(x)$',  color= 'm', linewidth = 2, linestyle = '--' )
# y_ini = np.delete(y_ini,place)

# x = np.zeros([120],dtype=float)
# y_final = np.zeros([120],dtype=float)
# for i in range(120):
#     x[i] = i+0.5
# for i in range(0,120):
#     y_final[i] = n_variable_interpolation()


#
# plt.legend(loc='best')
# plt.grid()
# plt.show()





















