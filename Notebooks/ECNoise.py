## ECNoise: (f,x_b,M,mult) -> (sigma_hat, fvals)
## Inputs: noisy black-box function f; base point x_b, Px1 col vec;
##         M number of samples to use, M must be >2; a handle for whether we have add/mult noise.
##	   We default to additive noise, so we set mult=false. user can set mult=true if needed.
## Outputs: an estimator to the variance in noise of f sigma_hat;
##          and the M function values stored in fvals

## Cost: M evals of f

import numpy as np

def ECNoise(f, x_b, M, mult=False):
	#determine length scale from max norm of x_b
	#h=1E-2*np.linalg.norm(x_b,ord=np.inf)
	h=0.01
	# Throw error for M too small
	if M<=2:
		return print('Please choose M>2.')

	# Grab dim(domain(f))=:P
	P=np.shape(x_b)[0]	

	# Hard-coded normalized direction to sample in
	if x_b.ndim > 1:
		#p_u=np.ones((P,1))
		p_u=np.random.rand(P,1)
	else:
		#p_u=np.ones(P)
		p_u=np.random.rand(P)
	
	p=p_u/np.linalg.norm(p_u)

	print(p.shape,x_b.shape)

	# Form difference table T
	xvals=np.copy(x_b)	
	T=np.zeros((M,M))
	for i in range(0,M):
		intx = x_b + (i*h)*p
		if i>=1:
			xvals=np.hstack((xvals,intx))
		else:
			xvals=xvals
        	
		T[i,0] = f(intx)
	for j in range(0,M-1):
    		for i in range(0,M-j-1):
        		T[i,j+1] = T[i+1,j] - T[i,j]
	
	fvals=T[:,0] # store the f evals

	# Make a row vector to store the k-level estimators (sigma_k^2) 
	# ie: Initialize empty vector for storage, row vector for readability
	S = np.zeros((1,M))

	# Build S according to paper; each k-th component of S is the k-th level estimator for the variance in our noise
	# which is computed using a scaled average of the k-th level difference values, from the difference table T
	for i in range(1,M):
    		S[0,i] = ((np.math.factorial(i)**2.)/np.math.factorial(2*i))*(1./(M-i))*np.sum(T[:,i]**2,axis=0)
	
	S=S[:,1:] # Don't need the first column (because first col. of T just holds function values!)

	# Form our estimator which averages the values that are usually good estimates
	sigma_hat=np.sum(S[:,1:M-1], axis=1)/(M-2)

	
	if mult is False:
		return [sigma_hat, xvals, fvals]
	else:
		return [sigma_hat/fvals[0]**2, xvals, fvals]
