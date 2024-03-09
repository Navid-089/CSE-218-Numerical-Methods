import numpy as np
import matplotlib.pyplot as plt

def value(x) :
    c = 5 * pow(10,-4)
    num = -(6.73 * x + 6.725 * pow(10, -8) + 7.26 * pow(10,-4) * c)
    den = 3.62 * pow(10,-12) * x + 3.908 * pow(10,-8) * x * c
    if(den == 0):
        den = 0.0000000000000000001
    return (num/den)

def t_integration(first,last,n):

    h = (last-first)/ n
    sum = value(first) + value(last)

    for i in range(1,n):
        sum = sum + 2 * value(first+h*i)
    sum = sum * (last-first) / (2*n)
    return sum

def s_integration(first,last,n):
    h = (last-first) / (2*n)
    sum = 0
    for i in range(0,n):
        x2 = first + 2* h*(i+1)
        x1 = first +  2* h*(i)
        sum = sum + (x2-x1) * ((value(x1)+4*value((x1+x2)/2)+value(x2))/6)
    return sum

def s_multiple_integration(first,last,n):
    h = (last - first) / (2 * n)
    sum = value(first) + value(last)

    for i in range(1,2*n):
        if (i%2) == 0:
             sum = sum + 2*value(first+i*h)
        else:
             sum = sum + 4 * value(first+i*h)
    sum = sum * (h/3)
    return sum
print("Please input the value of segments(n) : ")
segments = int(input())
t_values = np.zeros([segments],dtype=float)
s_values = np.zeros([segments],dtype=float)
for i in range(1,segments+1):
    t_values[i-1] = t_integration(0.75 * 1.22 * (pow(10,-4)), 0.25 * 1.22 * (pow(10,-4)), i)
    s_values[i-1] = s_integration(0.75 * 1.22 * (pow(10,-4)), 0.25 * 1.22 * (pow(10,-4)), i)

print("---TRAPEZOID RULE---\n")
print("INTEGRAL VALUES:")
for i in range(0,segments):
    print("Time required with",i+1,"segment(s) is",t_values[i])
print("\nRELATIVE ERRORS:")
for i in range(1,segments):
    rel_error = abs((t_values[i] - t_values[i-1]) / t_values[i-1]) * 100
    print("Relative error between segments ",i," and ",i+1,"is ",rel_error,"%")

print("\n\n---SIMPSON'S 1/3RD RULE---\n")
print("INTEGRAL VALUES:")
for i in range(0,segments):
    print("Time required with",2*(i+1),"segments is",s_values[i])
print("\nRELATIVE ERRORS:")
for i in range(1,segments):
    rel_error = abs((s_values[i] - s_values[i-1]) / s_values[i-1]) * 100
    print("Relative error between segments ",2*i," and ",2*i+2,"is ",rel_error,"%")

x = np.array([1.22,1.20,1.0,0.8,0.6,0.4,0.2])
x= x * pow(10,-4)
y = np.zeros([7],dtype=float)
for i in range(1,7):
    y[i] = s_multiple_integration(x[0],x[i],10)


plt.plot(x,y, label ='$f(x)$',  color= 'g', linewidth = 2, )
plt.plot(x,y, 'ro')
plt.xlabel("Oxygen Concentration (mol/cm^3)")
plt.ylabel("Time(seconds)")
plt.legend(loc='best')
plt.grid()
plt.show()





























