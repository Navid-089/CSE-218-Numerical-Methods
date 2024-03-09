import numpy as np

def GaussianElimination(A, B, pivot, showall):
    dim = len(B)
    C = np.hstack((A, B))

    for i in range(0, dim):

        i1 = i + 1

        if (pivot == True):
            max = abs(C[i][i])
            maxr = i

            # if(showall == True):
            # print(C)

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

    D = np.zeros([num, num], dtype=float)
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
            print("SOLUTION: ")
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

print("Please input the number of equations: ")
num = int(input())
if(num <= 0):
    print("No input's given.")
    exit()
a = np.zeros([num, num], dtype=float)
b = np.zeros([num, 1], dtype=float)
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
final_solve = GaussianElimination(a, b, True, True)
print("SOLUTION : ")
for i in range(0, num):
        print('x', (i + 1), "=", ("%.4f" % float(final_solve[i][0])))
