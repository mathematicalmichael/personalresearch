import numpy as np


def STARS(x_init, F, mu_star, h, active=None):
    '''    
    x_init: initial x value
    F: function we wish to minimize
    note that lower f's are evaluations of F(*)
    mu_star: smoothing parameter
    h: step length 
    '''

    # Evaluate noisy F(x_init)
    f = F(x_init)  # noise now in F+ k*(2*np.random.rand(1)-1)

    # Draw a random vector of same size as x_init
    if active is None:
        u = np.random.normal(0, 1, np.size(x_init))
    else:
        act_dim = active.shape[0]
        lam = np.random.normal(0, 1, act_dim)
        u = np.dot(lam, active)
        # print(u) #debug

    # Form vector y, which is a random walk away from x_init
    y = x_init + (mu_star)*u

    # Evaluate noisy F(y)
    g = F(y)  # noise now in F + k*(2*np.random.rand(1) -1)

    # Form finite-difference "gradient oracle"
    s = ((g - f)/mu_star)*u

    # Take descent step in direction of -s smooth by h to get next iterate, x_1
    x = x_init - (h)*s

    # Evaluate noisy F(x_1)
    f = F(x)  # noise now in F+ k*(2*np.random.rand(1) -1)

    return [x, f, y, g]


def STARS_RV(x_init, F, u, mu_star, h):
    '''   
    # x_init: initial x value
    # F: function we wish to minimize
    # u: Choice of random vector; probably comes from active subspace
    # note that lower f's are evaluations of F(*)

    # mu_star: smoothing parameter
    # h: step length
    '''

    # Evaluate noisy F(x_init)
    f = F(x_init)  # + k*(2*np.random.rand(1)-1)

    # Form vector y, which is a random walk away from x_init
    # in the direction of u, which the user defines
    sclr = np.random.normal(0, 1)
    u = sclr*u
    y = x_init + (mu_star)*u
    # Evaluate noisy F(y)
    g = F(y)  # + k*(2*np.random.rand(1) -1)

    # Form finite-difference "gradient oracle"
    # print((g-f)/mu_star)
    s = ((g - f)/mu_star)*u
    # print(s)
    # Take descent step in direction of -s smooth by h to get next iterate, x_1
    x_maybe = x_init - (h)*s

    # Evaluate noisy F(x_1)
    f_maybe = F(x_maybe)  # + k*(2*np.random.rand(1) -1)

    # Did we drop the value of f?

    # if f_maybe<=f:
    f = f_maybe
    x = x_maybe
    # else:
    #   f=f
    #   x=x_init

    return [x, f, y, g]


def get_mu_star(var, L_1, N):
    return ((8*var*N)/(L_1**2*(N+6)**3))**0.25


def get_h(L1, N):
    return 1/(4*L1*(N+4))
