import numpy as np
import matplotlib.pyplot as plt

def f1(x) :
    return (x ** 3) - 0.18 * (x ** 2) + 0.0004752

def f2(x):
    return 3*(x ** 2) - (0.36 * x)

def by_bisection(xu,xh,error,max_it):
    e = 1
    it = 0

    while it < max_it :
        xm = (xu+xh) / 2
        if (f1(xm) * f1(xu) < 0):
            xh = xm
        elif (f1(xm) * f1(xu) > 0):
            xu = xm
        else:
            return xm
        if (it > 0):
            e = abs( ((xm - a) / xm) * 100 )

            print("After", it+1,"th iteration, error is ",e)
        a = xm
        if e <=error:
            break
        #i++ doesn't work???
        it = it+1

    return xm

def by_newraph(param, error, max_it):
    x = param
    e = 1
    it = 0

    while it< max_it:
        y = f1(x)
        y_der = f2(x)

        if(y_der == 0):
            y_der = y_der + 0.00001
        x2 = x-(y/y_der)

        if(it >0):
            e = ((x2-x)/x2)*100
            if(e<0):
                e=-e
            print("After",it+1,"th iteration, error is ",e)
        x = x2
        if(e<= error):
            break
        it = it+1
    return x


### main function
x = np.linspace(0,0.15,200)
y = f1(x)
plt.plot(x,y,label ='$f1(x)$', color= 'y', linewidth = 2 )
plt.legend(loc='best')
plt.grid()
plt.show()

print("Bisection Method -----")
print("Erros : ")
val  = by_bisection(0.06, 0.09, 0.005,100)
print("Value of x from Bisection Method: ",val,"\n\n")

print("Newton-Raphson Method -----")
print("Erros : ")
val2  = by_newraph(0.07,0.5,30)
print("Value of x from Newton-Raphson Method: ",val2,"\n\n")









