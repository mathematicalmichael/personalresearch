import numpy as np

def get_mu_star(var,L_1,N):
    return ((8*var*N)/(L_1**2*(N+6)**3))**0.25

def get_h(L1,N):
    return 1/(4*L1*(N+4))

def STARS(x_init,F,mu_star,h,active=None):
    '''    
    x_init: initial x value, size Px1
    F: function we wish to minimize
    note that lower f's are evaluations of F(*)
    mu_star: smoothing parameter
    h: step length 
    active: Should be size Pxk
    
    '''
    
    # Evaluate noisy F(x_init)
    f0 = F(x_init)
    
    # Draw a random vector of same size as x_init
    if active is None:
        u = np.random.normal(0,1,(np.size(x_init),1))
        
    else:
        act_dim=active.shape[1]
        lam = np.random.normal(0,1,(act_dim,1))
        u = active@lam
    
    
    # Form vector y, which is a random walk away from x_init
    y = x_init + (mu_star)*u
    
    # Evaluate noisy F(y)
    g = F(y)
    
    # Form finite-difference "gradient oracle"
    s = ((g - f0)/mu_star)*u 
    
    # Take descent step in direction of -s smooth by h to get next iterate, x_1
    x = x_init - (h)*s
   
    # Evaluate noisy F(x_1)
    f1=F(x)
    
    # Form upper bound for L_1
    d1=(f0-g)*(1/mu_star)
    d2=(g-f1)*(1/h)
    #avg_step=.5*mu_star**2+.5*h**2
    L_1_B=abs(d2-d1)/(h+mu_star)
    
        
    return [x, f1, y, g, x_init, f0, L_1_B]

