import numpy as np


def get_mu_star(var, L_1, N):

    return ((8*var*N)/(L_1**2*(N+6)**3))**0.25


def get_h(L1, N):

    return 1/(4*L1*(N+4))


def get_mult_mu_star(var, L_1, N):
    return ((16*var*N)/((L_1**2)*(1+3*var)*(N+6)**3))**0.25


def STARS(x_init, F, mu_star, h, active=None, mult=False, wts=None):
    '''     
    x_init: initial x value, size Px1

    F: function we wish to minimize
; note that lower f's are evaluations of F(*)

    mu_star: smoothing parameter

    h: step length 

    active: Should be size Pxk
; default is none
    mult: Set to true if mult noise; default is add noise
    wts: Should be size Px1, filled with the weights
         Default is None.
         If using active vars, user should provide wts=ss.eigenvals    

    '''

    # Evaluate noisy F(x_init)

    f0 = F(x_init)

    # Compute mult_mu_star_k (if needed)
    if mult is False:
        mu_star = mu_star
    else:
        mu_star = mu_star*abs(f0)**0.5

    # Draw a random vector of same size as x_init

    if active is None:

        u = np.random.normal(0, 1, (np.size(x_init), 1))

    else:

        act_dim = active.shape[1]

        if wts is None:
            lam = np.random.normal(0, 1, (act_dim, 1))

        else:
            lam = np.zeros((act_dim, 1))
            for i in range(act_dim):
                lam[i] = np.random.normal(0, wts[0][0]/wts[i][0])

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

    f1 = F(x)

    # Form upper bound for L_1

    d1 = (f0-g)*(1/mu_star)

    d2 = (g-f1)*(1/h)

    avg_step = .5*mu_star**2+.5*h**2

    L_1_B = abs(d2-d1)/avg_step

    return [x, f1, y, g, x_init, f0, L_1_B]

    # if f1<=f0:
    # return [x, f1, y, g, x_init, f0, L_1_B]

    # else:
    # return [x_init, f0, y, g, x_init, f1, L_1_B]
