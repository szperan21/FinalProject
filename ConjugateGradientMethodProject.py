#Import Packages
import numpy as np
import scipy
import datetime
import math

#Implement the method of steepest descent
def SteepestDescent(A,b,max_steps=10000,start=None, relative_tolerance=.00001, absolute_tolerance=0):
    
    #Select an appropriate dimension vector of zeros as the starting point if none is specified
    if(start==None):
        start = np.zeros(int(math.sqrt(A.size)))
    
    #Initialize vars
    current_approx = start
    current_residual = b - np.matmul(A,current_approx)
    alpha = 0
    
    for iterations in range(max_steps):
        
        #Calculate scalar to move the optimal distance in direction of steepest descent 
        alpha = (np.matmul(current_residual.transpose(), current_residual)) / (np.matmul(current_residual.transpose(), np.matmul(A,current_residual)))
        
        #Take the step and update approximation
        current_approx = current_approx + alpha*current_residual
        
        #Calculate new residual
        current_residual = b - np.matmul(A,current_approx)
        
        #print(np.linalg.norm(current_residual))
        
        #End the loop if length of residual is below desired accuracy tolerance. Default values match scipy.linalg.cg
        if(np.linalg.norm(current_residual) < relative_tolerance*np.linalg.norm(b) or np.linalg.norm(current_residual) < absolute_tolerance):
            break
            
    
    return current_approx

#Compare performance between method of steepest descent, full conjugate-gradient method, and scipy.linalg.solve function

#np.random.seed(2123456782)

size = 2

#Generate a random sparse matrix. The conjugate-gradient method is known to be more advantageous for sparse matrices than dense ones
MStart = datetime.datetime.now()
M = scipy.sparse.random_array((size,size), density = .05)
MEnd = datetime.datetime.now()
print("matrix creation time = ", (MEnd-MStart).total_seconds())

M=M.todense()

#Use the random matrix to create a symmetric, positive-definite matrix
A = (M + M.transpose())/2

#Generate a random target vector b of appropriate dimension
b = array = np.random.random(size).astype(np.float32)

#testA = np.array([[3,2],[2,6]])
#testb = np.array([2, -8])

#print(A)

#Approximate with Steepest Descent
SDStart = datetime.datetime.now()
SDApprox = SteepestDescent(A,b,max_steps=100)
SDEnd = datetime.datetime.now()

print("Method of Steepest Descent:", SDApprox, "time = ", (SDEnd-SDStart).total_seconds())

#Approximate with Conjugate Gradient
CGStart = datetime.datetime.now()
CGApprox = scipy.sparse.linalg.cg(A, b)
CGEnd = datetime.datetime.now()

CGTime = (CGEnd-CGStart).total_seconds()

print("Conjugate-Gradient Method:",CGApprox, "time = ", CGTime)

#Solve with linalg.solve
solveStart = datetime.datetime.now()
sol = scipy.linalg.solve(A,b) #,assume_a='positive definite'
solveEnd = datetime.datetime.now()

print("scipy.linalg.solve:", sol, "time = ", (solveEnd-solveStart).total_seconds())




