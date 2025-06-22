# OptimalGradientDescent

## Usage

```
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

    if mom == "pogm":
        return xnew, cost
    else:
        return ynew, cost

```