import numpy as np
import copy
import matplotlib.pyplot as plt

# USER DEFINED FUNCTIONS

def Fcost(x, vargs):
    data_fidelity = 0.5*np.linalg.norm(vargs["y"] - vargs["A"] @ x)**2
    reg = vargs["lam"]*np.linalg.norm(x, ord=1)
    return data_fidelity+reg

def f_grad(x, vargs):
    return -1 * vargs["A"].T @ (vargs["y"] - vargs["A"] @ x)

def soft_threshold(M, lam):
    return np.sign(M) * np.maximum(np.abs(M) - lam, 0)

def g_prox(M, c, vargs):
    return soft_threshold(M, c * vargs["lam"])

# OPTIMAL GRADIENT METHOD FUNCTIONS

def gr_restart(Fgrad, ynew_yold, restart_cutoff):
    """
    Determines whether to restart based on a gradient check.
    
    Parameters:
        Fgrad (np.ndarray): The gradient at the current iteration.
        ynew_yold (np.ndarray): The difference between new and old values.
        restart_cutoff (float): The cutoff value for restarting.
        
    Returns:
        bool: True if the sum condition is met for a restart, False otherwise.
    """
    return np.sum(np.real(-Fgrad * ynew_yold)) <= restart_cutoff * np.linalg.norm(Fgrad) * np.linalg.norm(ynew_yold)


def pogm(x0, vargs, Fcost, f_grad, f_L, g_prox, f_mu=0, mom="pogm", restart="gr", restart_cutoff=0, bsig=1, tol=1e-4):
    """
    x, out = pogm_restart(x0, Fcost, f_grad, f_L ;
    f_mu=0, mom=:pogm, restart=:gr, restart_cutoff=0.,
    bsig=1, niter=10, g_prox=(z,c)->z, fun=...)

    Iterative proximal algorithms (PGM=ISTA, FPGM=FISTA, POGM) with restart.

    # in
    - `x0` initial guess
    - `Fcost` function for computing the cost function value ``F(x)``
        - (needed only if `restart === :fr`)
    - `f_grad` function for computing the gradient of ``f(x)``
    - `f_L` Lipschitz constant of the gradient of ``f(x)``

    # option
    - `f_mu` strong convexity parameter of ``f(x)``; default 0.
        - if `f_mu > 0`, ``(\\alpha, \\beta_k, \\gamma_k)`` is chosen by Table 1 in [KF18]
    - `g_prox` function `g_prox(z,c)` for the proximal operator for ``g(x)``
        - `g_prox(z,c)` computes ``argmin_x 1/2 \\|z-x\\|^2 + c \\, g(x)``
    - `mom` momentum option
        - `:pogm` POGM (fastest); default!
        - `:fpgm` (FISTA), ``\\gamma_k = 0``
        - `:pgm` PGM (ISTA), ``\\beta_k = \\gamma_k = 0``
    - `restart` restart option
        - `:gr` gradient restart; default!
        - `:fr` function restart
        - `:none` no restart
    - `restart_cutoff` for `:gr` restart if cos(angle) < this; default 0.
    - `bsig` gradient "gamma" decrease option (value within [0 1]); default 1
        - see ``\\bar{\\sigma}`` in [KF18]
    - `niter` number of iterations; default 10

    # out
    - `x` final iterate
        - for PGM (ISTA): ``x_N = y_N``
        - for FPGM (FISTA): primary iterate ``y_N``
        - for POGM: secondary iterate ``x_N``, see [KF18]

    Optimization Problem: Nonsmooth Composite Convex Minimization
    * ``argmin_x F(x),  F(x) := f(x) + g(x))``
        - ``f(x)`` smooth convex function
        - ``g(x)`` convex function, possibly nonsmooth and "proximal-friendly" [CP11]

    # Optimization Algorithms:

    Accelerated First-order Algorithms when ``g(x) = 0`` [KF18]
    iterate as below for given coefficients ``(\\alpha, \\beta_k, \\gamma_k)``
    * For k = 0,1,...
        - ``y_{k+1} = x_k - \\alpha  f'(x_k)`` : gradient update
        - ``x_{k+1} = y_{k+1} + \\beta_k  (y_{k+1} - y_k) + \\gamma_k  (y_{k+1} - x_k)`` : momentum update

    Proximal versions of the above for ``g(x) \\neq 0`` are in the below references,
    and use the proximal operator
    ``prox_g(z) = argmin_x {1/2\\|z-x\\|^2 + g(x)}``.

    - Proximal Gradient method (PGM or ISTA) - ``\\beta_k = \\gamma_k = 0``. [BT09]
    - Fast Proximal Gradient Method (FPGM or FISTA) - ``\\gamma_k = 0``. [BT09]
    - Proximal Optimized Gradient Method (POGM) - [THG15]
    - FPGM(FISTA) with Restart - [OC15]
    - POGM with Restart - [KF18]

    # references

    - [CP11] P. L. Combettes, J. C. Pesquet,
    "Proximal splitting methods in signal processing,"
    Fixed-Point Algorithms for Inverse Problems in Science and Engineering,
    Springer, Optimization and Its Applications, 2011.
    - [KF18] D. Kim, J.A. Fessler,
    "Adaptive restart of the optimized gradient method for convex optimization," 2018
    Arxiv:1703.04641,
    [http://doi.org/10.1007/s10957-018-1287-4]
    - [BT09] A. Beck, M. Teboulle:
    "A fast iterative shrinkage-thresholding algorithm for linear inverse problems,"
    SIAM J. Imaging Sci., 2009.
    - [THG15] A.B. Taylor, J.M. Hendrickx, F. Glineur,
    "Exact worst-case performance of first-order algorithms
    for composite convex optimization," Arxiv:1512.07516, 2015,
    SIAM J. Opt. 2017
    [http://doi.org/10.1137/16m108104x]

    Copyright 2017-3-31, Donghwan Kim and Jeff Fessler, University of Michigan
    2018-08-13 Julia 0.7.0
    2019-02-24 interface redesign
    2025-06-21 javier redesign for python
    """


    L = f_L
    mu = f_mu
    q = mu/L

    told = 1
    sig = 1
    zetaold = 1

    xold = x0
    yold = x0
    uold = x0
    zold = x0
    Fcostold = Fcost(x0, vargs)
    Fgradold = np.zeros(x0.shape)
    cost = []
    cost.append(Fcostold)
    error = 1000
    xnew = []
    ynew = []
    iter = 1
    while error > tol:
        # PGM update
        if mom == "pgm" and mu != 0:
            alpha = 2/(L+mu)
        else:
            alpha = 1/L
        
        fgrad = f_grad(xold, vargs)

        is_restart = False

        if mom == "pgm" or mom == "fpgm":
            ynew = g_prox(xold - alpha * fgrad, alpha, vargs)
            Fgrad = -(1/alpha)*(ynew - xold)
            Fcostnew = Fcost(ynew, vargs)

            if restart != "none":
                if ((restart == "fr" and Fcostnew > Fcostold) or (restart == "gr" and gr_restart(Fgrad, ynew-yold, restart_cutoff))):
                    told = 1
                    is_restart = True

                Fcostold = Fcostnew
        elif mom == "pogm":
            unew = xold - alpha * fgrad
        else:
            print("mom type not recognized")
            return None
        
        # momentum coefficient "beta"
        if mom == "fpgm" and mu != 0:
            beta = (1 - np.sqrt(q)) / (1 + np.sqrt(q))
        elif mom == "pogm" and mu !=0:
            beta = (2 + q - np.sqrt(q**2 + 8*q))**2 / 4. / (1-q)
        elif mom != "pgm":
            if mom == "pogm" and iter == -1:
                tnew = 0.5 * (1 + np.sqrt(1 + 8 * told**2))
            else:
                tnew = 0.5 * (1 + np.sqrt(1 + 4 * told**2))
            
            beta = (told - 1)/tnew

        # momentum update
        if mom == "pgm":
            xnew = ynew
        elif mom == "fpgm":
            xnew = ynew + beta * (ynew - yold)
        elif mom == "pogm": # see [KF18]
            # momentum coefficient "gamma"
            if mu !=0:
                gamma = (2 + q - np.sqrt(q**2 + 8*q)) / 2.
            else:
                gamma = sig * told / tnew

            znew = (unew + beta * (unew - uold) + gamma * (unew - xold) - beta * alpha / zetaold * (xold - zold))
            zetanew = alpha * (1 + beta + gamma)
            xnew = g_prox(znew, zetanew, vargs) # non-standard PG update for POGM

            # non-standard composite gradient mapping for POGM
            Fgrad = fgrad - 1/zetanew * (xnew - znew)
            ynew = xold - alpha * Fgrad
            Fcostnew = Fcost(xnew, vargs)

            # restart + gamma decrease conditions for POGM
            if restart != "none":
                if ((restart == "fr" and Fcostnew > Fcostold) or (restart == "gr" and gr_restart(Fgrad, ynew-yold, restart_cutoff))):
                    tnew = 1
                    sig = 1
                    is_restart = True
                elif np.sum(np.real(Fgrad * Fgradold)) < 0:
                    sig = bsig * sig
                
                Fcostold = Fcostnew
                Fgradold = Fgrad

            uold = unew
            zold = xnew
            zetaold = zetanew

        if mom == "pogm":
            cost.append( Fcost(xnew, vargs) )
            error = np.linalg.norm(xnew-xold)/np.linalg.norm(xold)
        else:
            cost.append( Fcost(ynew, vargs) )
            error = np.linalg.norm(ynew-yold)/np.linalg.norm(yold)

        

        xold = xnew
        yold = ynew

        if mom != "pgm" and mu == 0:
            told = tnew

        
        iter = iter + 1
        # end of while loop

    if mom == "pogm":
        return xnew, cost
    else:
        return ynew, cost




if __name__ == "__main__":
    np.random.seed(42)
    D = 1000
    N = 3
    noise = 0.1
    lam = 1.0

    A = np.random.rand(D,N)
    f_L = np.linalg.norm(A, ord=2)**2
    x_true = np.array([1.5, 0, -2])
    y = A @ x_true + noise * np.random.randn(D)

    x0 = np.random.randn(N)
    vargs = {"A": A, "y": y, "lam": lam}
    true_cost = Fcost(x_true, vargs)
    
    x_pogm, cost_pogm = pogm(x0, vargs, Fcost, f_grad, f_L, g_prox, f_mu=0, mom="pogm", restart="gr")
    x_fpgm, cost_fpgm = pogm(x0, vargs, Fcost, f_grad, f_L, g_prox, f_mu=0, mom="fpgm", restart="gr")
    x_pgm, cost_pgm = pogm(x0, vargs, Fcost, f_grad, f_L, g_prox, f_mu=0, mom="pgm", restart="gr")
    
    print("pogm x estimate: "+str(x_pogm))
    
    plt.plot(np.log(np.abs(cost_pogm -true_cost)), label="pogm", marker="D")
    plt.plot(np.log(np.abs(cost_fpgm - true_cost)), label="fpgm", marker="x")
    plt.plot(np.log(np.abs(cost_pgm - true_cost)), label="pgm", marker="o")
    plt.legend()
    plt.title("Optimal Gradient Method Comparisons")
    plt.xlabel("Iterations")
    plt.ylabel("$ \\text{log} | \\text{Fcost} - \\text{Ftrue} | $")
    plt.tight_layout()
    plt.savefig("/home/javier/Desktop/ProximalOptimizedGradientMethod/plot.png", dpi = 1200, bbox_inches='tight')