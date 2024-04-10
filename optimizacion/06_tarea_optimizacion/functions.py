import numpy as np
import matplotlib.pyplot as plt

##############################
#En este módulo se encuentran definidas las funciones:
#Himmelblau
#Beale
#Rosenbrock
#Hartman
#Así como sus gradientes y Hessianas
##############################
def Himmelblau(x):
    x1=x[0]
    x2=x[1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


def D_Himmelblau(x):
    x1=x[0]
    x2=x[1]
    gx = 4*x1*(x1**2 + x2 - 11) + 2*(x1 + x2**2 - 7)
    gy = 2*(x1**2 + x2 - 11) + 4*x2*(x1 + x2**2 - 7)
    return np.array([gx, gy])

def H_Himmelblau(x):
    x1=x[0]
    x2=x[1]
    hessiana = np.array([[12*(x1**2) + 4*x2 - 42, 4*x1 + 4*x2],
                         [4*x1 + 4*x2, 12*(x2**2) + 4*x1 - 26]])
    return hessiana


def Beale(x):
    x1=x[0]
    x2=x[1]
    return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*(x2**2))**2 + (2.625 - x1 + x1*(x2**3))**2

def D_Beale(x):
    x1=x[0]
    x2=x[1]    
    dx1 = 2*(1.5 - x1 + x1*x2)*(-1 + x2) + 2*(2.25 - x1 + x1*(x2**2))*(-1 + x2**2) + 2*(2.625 - x1 + x1*(x2**3))*(-1 + x2**3)
    dx2 = 2*(1.5 - (x[0]) + (x[0])*(x[1]))*((x[0])) + 2*(2.25 - (x[0]) + (x[0])*(x[1])**2)*(2*(x[0])*(x[1])) + 2*(2.625 - (x[0]) + (x[0])*(x[1])**3)*(3*(x[0])*(x[1])**2)
    return np.array([dx1, dx2])

def H_Beale(x):
    x1=x[0]
    x2=x[1] 
    d_x1x1 = 2 * (x2**6 + x2**4 - 2*x2**3 - x2**2 - 2*x2 + 3)
    d_x2x2 = x1 * (31.5*x2 + x1*(30*x2**4 + 3*x2**2 - 2*x2 - 2) + 9)
    d_x1x2 = 15.75*x2**2 + 9*x2 + 4*x1*(3*x2**5 + 2*x2**3 - 3*x2**2 - x2 - 1) + 3
    return [[d_x1x1, d_x1x2], [d_x1x2, d_x2x2]]



def Rosenbrock(x):
    n = len(x)
    suma = 0
    for i in range(n-1):
        suma += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return suma

def D_Rosenbrock(x):
    n = len(x)
    gradient = np.zeros(n)
    for i in range(n-1):
        gradient[i] += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
        gradient[i+1] += 200 * (x[i+1] - x[i]**2)
    return gradient


def H_Rosenbrock(x):
    n = len(x)
    hessian = np.zeros((n, n))
    for i in range(n - 1):
        hessian[i, i] += 1200 * x[i]**2 - 400 * x[i + 1] + 2
        hessian[i, i + 1] += -400 * x[i]
        hessian[i + 1, i] += -400 * x[i]
        hessian[i + 1, i + 1] += 200
    return hessian


#definimos los parámetros de la funcion de hartman
alpha = np.array([1.0, 1.2, 3.0, 3.2])

A = np.array([[10, 3, 17, 3.5, 1.7, 8],
              [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14]])


P = 10**(-4) * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                          [2329, 4135, 8307, 3736, 1004, 9991],
                          [2348, 1451, 3522, 2883, 3047, 6650],
                          [4047, 8828, 8732, 5743, 1091, 381]])

def Hartman(x):
    
    sum1=0
    for k in range(4):
        sum2=0
        for j in range(6):
            sum2+=A[k][j]*((x[j]-P[k][j])**2)
        sum1+= alpha[k]*np.exp(-sum2)/1.94
    
    return -(1/1.94)*(2.58 + sum1)

def D_Hartman(x):
    
    gradient=np.zeros(6)
    for i in range(6):
        sum1=0
        for k in range(4):
            sum2=0
            for j in range(6):
                sum2+=A[k][j]*((x[j]-P[k][j])**2)
            sum1+= 2*alpha[k]*A[k][i]*(x[i] - P[k][i])*np.exp(-sum2)/1.94
        gradient[i]=sum1
  
    return gradient

def H_Hartman(x):
    
    hessian=np.zeros([6,6])
    for i in range(6):
        for l in range(i, 6):
            sum1=0
            for k in range(4):
                sum2=0
                for j in range(6):
                    sum2+=A[k][j]*((x[j]-P[k][j])**2)
                sum1+=2*alpha[k]*A[k][i]*np.exp(-sum2)*(1-2*A[k][l]*(x[l]-P[k][l])*(x[i]-P[k][i]))/1.94
            hessian[i][l]=hessian[l][i]=sum1
        
    return hessian


def cuadratica(A, b, x):
    """
    Evalúa la función f(x) = 0.5 * x^T * A * x - b^T * x.

    Parámetros: Matriz A, vectores b y x (numpy.ndarray)

    Return: Valor de la función evaluada en x.
    """
    return 0.5 * np.dot(x.T, np.dot(A, x)) - np.dot(b.T, x)

def D_cuadratica(A, b, x):
    """
    Calcula el gradiente de la función cuadrática f(x) = 1/2 * x^T * A * x - b^T * x.
    
    Parámetros: Matriz A, vectores b y x (numpy.ndarray)
    
    Returns: Gradiente de f(x) (numpy.ndarray) evaluada en x.
    """
    return 0.5 * (np.dot(A.T, x) + np.dot(A, x)) - b