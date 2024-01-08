import numpy as np

def Gradient_Descent (X,y):
    curr_m=0 #intial val for m
    curr_b=0 #intial val for b
    iters=10  #no of iterations 
    n=len(X)
    lr=.001  #Learning rate
    

    for i in range(iters):
        y_pred=(curr_m*X)+curr_b  #Y=mx+b
        cost=(1/n)*sum([val**2 for val in (y-y_pred)]) #J(θ)= 2m​∑ i=1m​(h θ​(x (i) )−y (i) ) 2

        m_d=-(2/n)*sum(y-y_pred)
        b_d=-(2/n)*sum(y-y_pred)

        curr_m=curr_m-(lr*m_d)
        curr_b=curr_b-(lr*b_d)

        print(f"m {0}, cost {1},iteration{2}",(curr_m,curr_b,i))
    

X=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])
Gradient_Descent(X,y)

