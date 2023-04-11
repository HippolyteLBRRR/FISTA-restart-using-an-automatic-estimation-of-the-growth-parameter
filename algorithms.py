#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import numpy.linalg as npl
import time


def ForwardBackward_step(x,s,proxh,Df):
    """Computes a Forward-Backward step for a composite function F=f+h where
    f is differentiable and the proximal operator of h can be computed.
    
    Parameters
    ----------
        
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        proxh : operator
            Proximal operator of the non differentiable function h.
        Df : operator
            Derivative of the differentiable function f.
    
    Returns
    -------
        
        xp: array_like
            Final vector.
    """
    return proxh(x-s*Df(x),s)



def ForwardBackward(x,s,Niter,epsilon,Df,proxh,F=None,exit_crit=None,
                    extra_function=None,track_ctime=False):
    """Forward-Backward method applied to a composite function F=f+h where f
    is differentiable and the proximal operator of h can be computed.
    
    Parameters
    ----------
        
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        Niter : integer
            Maximum number of iterations.
        epsilon: float
            Expected accuracy for the given exit criteria.
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        F : operator, optional
            Function to minimize. If the user gives F, the function will
            compute the value of F at each iterate.
        exit_crit : operator, optional
            Exit criteria, function of x_k and x_{k-1}. The default is the
            norm of (x_k - x_{k-1}).
        extra_function : operator, optional
            Additional function to compute at each iterate.
        track_ctime : boolean, optional
            Parameter which states if the function returns an array containing
            the computation time at each iteration. The default value is False.

    Returns
    -------
        
        x : array_like
            Last iterate given by the method.
        cost : array_like, optional
            Array containing the value of F at each iterate. It is returned if
            F is given by the user.
        ctime : array_like, optional
            Array containing the computation time required at each iterate. 
            The last coefficient is the total computation time. It is returned
            if track_ctime is True.
        extra_cost : array_like, optional
            Array containing the value of extra_function at each iterate if a 
            function is given."""
    if F is not None:cost = [F(x)]
    if extra_function is not None:extra_cost = [extra_function(x)]
    if track_ctime:
        extratime = 0
        ctime = [0]
        t0 = time.perf_counter()
    if exit_crit is None:exit_crit = lambda x,xm:npl.norm(x-xm,2)
    i = 0
    out = False
    while i < Niter and out == False:
        xm = np.copy(x)
        x = ForwardBackward_step(x,s,proxh,Df)
        out = exit_crit(x,xm) < epsilon
        if track_ctime:
            t_temp = time.perf_counter()
            ctime += [t_temp - extratime - t0]
        if F is not None:cost += [F(x)]
        if extra_function is not None:extra_cost += [extra_function(x)]
        if track_ctime:extratime += time.perf_counter()-t_temp
        i += 1
    output = (x,)
    if F is not None:output+=(np.array(cost),)
    if track_ctime:output+=(np.array(ctime),)
    if extra_function is not None:output+=(np.array(extra_cost))
    if len(output) == 1:output = x
    return output

def FISTA(x,s,Niter,epsilon,Df,proxh,alpha=3,F=None,exit_crit=None
          ,restarted=False, extra_function=None,track_ctime=False):
    """FISTA applied to a composite function F=f+h where f is differentiable
    and the proximal operator of h can be computed.
    
    Parameters
    ----------
    
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        Niter : integer
            Maximum number of iterations.
        epsilon: float
            Expected accuracy for the given exit criteria.
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        alpha : float, optional
            Inertial parameter involved in the step 
            y_k=x_k+k/(k+alpha)*(x_k-x_{k-1}). The default value is 3.
        F : operator, optional
            Function to minimize. If the user gives F, the function will 
            compute the value of F at each iterate.
        exit_crit : operator, optional
            Exit criteria, function of x_k and y_k. The default is the norm 
            of (x_k - y_k).
        restarted : boolean, optional
            Parameter which specifies if FISTA will be restarted. If it is
            True the function FISTA returns an additional boolean "out" which
            reports if the exit condition was satisfied . The purpose is to
            avoid to check the exit condition again after running FISTA. The
            default value is False.
        extra_function : operator, optional
            Additional function to compute at each iterate.
        track_ctime : boolean, optional
            Parameter which states if the function returns an array containing
            the computation time at each iteration. The default value is False.
    Returns
    -------
        
        x : array_like
            Last iterate given by the method.
        cost : array_like, optional
            Array containing the value of F at each iterate. It is returned if
            F is given by the user.
        ctime : array_like, optional
            Array containing the computation time required at each iterate. 
            The last coefficient is the total computation time. It is returned
            if track_ctime is True.
        extra_cost : array_like, optional
            Array containing the value of extra_function at each iterate if a 
            function is given.
        out : boolean, optional
            Parameter reporting if the exit condition is satisfied at the last
            iterate. It is returned if restarted is True which is not the case
            by default."""
    if F is not None and not restarted:cost = [F(x)]
    if F is not None and restarted:cost = []
    if extra_function is not None:extra_cost = [extra_function(x)]
    if track_ctime:
        extratime = 0
        ctime = [0]
        t0 = time.perf_counter()
    if exit_crit is None:exit_crit = lambda x,y:npl.norm(x-y,2)
    y = np.copy(x)
    i = 0
    out = False
    while i < Niter and out == False:
        i += 1
        xm = np.copy(x)
        x = ForwardBackward_step(y,s,proxh,Df)
        out = exit_crit(x,y)<epsilon
        y = x+(i-1)/(i+alpha-1)*(x-xm)
        if track_ctime:
            t_temp = time.perf_counter()
            ctime += [t_temp - extratime - t0]
        if F is not None:cost += [F(x)]
        if extra_function is not None:extra_cost += [extra_function(x)]
        if track_ctime:extratime += time.perf_counter()-t_temp
    output = (x,)
    if F is not None:output += (np.array(cost),)
    if track_ctime:output += (np.array(ctime),)
    if extra_function is not None:output += (np.array(extra_cost),)
    if restarted:output += (out,)
    if len(output) == 1:output = x
    return output


    
def FISTA_fixed_restart(x,s,n_r,Niter,epsilon,Df,proxh,alpha=3,F=None,
                        exit_crit=None,extra_function=None,track_ctime=False):
    """Restarted version of FISTA (a restart occurs every n_r iterations).
    
    Parameters
    ----------
    
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        n_r : integer
            Number of iterations between each restart.
        Niter : integer
            Maximum number of iterations.
        epsilon: float
            Expected accuracy for the given exit criteria.
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        alpha : float, optional
            Inertial parameter involved in the step
            y_k=x_k+k/(k+alpha)*(x_k-x_{k-1}). The default value is 3.
        F : operator, optional
            Function to minimize. If the user gives F, the function will
            compute the value of F at each iterate.
        exit_crit : operator, optional
            Exit criteria, function of x_k and y_k. The default is the norm of
            (x_k - y_k).
        extra_function : operator, optional
            Additional function to compute at each iterate.
        track_ctime : boolean, optional
            Parameter which states if the function returns an array containing
            the computation time at each iteration. The default value is False.
    Returns
    -------
        
        x : array_like
            Last iterate given by the method.
        cost : array_like, optional
            Array containing the value of F at each iterate. It is returned if
            F is given by the user.
        ctime : array_like, optional
            Array containing the computation time required at each iterate. 
            The last coefficient is the total computation time. It is returned
            if track_ctime is True.
        extra_cost : array_like, optional
            Array containing the value of extra_function at each iterate if a 
            function is given."""
    if F is not None:cost = [F(x)]
    if extra_function is not None:extra_cost = [extra_function(x)]
    if exit_crit is None:exit_crit = lambda x,y:npl.norm(x-y,2)
    if track_ctime:ctime = [0]
    i = 0
    out = False
    while i < Niter and out == False:
        outputs = FISTA(x,s,np.minimum(Niter-i,n_r),epsilon,Df,proxh,alpha,F,
                        exit_crit,restarted=True,extra_function=extra_function
                        ,track_ctime=track_ctime)
        x = outputs[0]
        j = 1
        if F is not None:
            cost_temp = outputs[j]
            cost = np.concatenate((cost,cost_temp))
            j+=1
        if track_ctime:
            ctime_temp = outputs[j]
            ctime = np.concatenate((ctime,ctime_temp[1:]+ctime[-1]))
            j+=1
        if extra_function is not None:
            extra_cost_temp = outputs[j]
            extra_cost = np.concatenate((extra_cost,extra_cost_temp[1:]))
            j+=1
        out = outputs[j]        
        i += np.minimum(Niter-i,n_r)
    output = (x,)
    if F is not None:output += (np.array(cost),)
    if track_ctime:output += (np.array(ctime),)
    if extra_function is not None:output += (np.array(extra_cost),)
    if len(output) == 1:output = x
    return output

def FISTA_grad_restart(x,s,Niter,epsilon,Df,proxh,alpha=3,F=None,exit_crit=None,
                       sp=None,extra_function=None,track_ctime=False):
    """FISTA empirical restart based on the gradient (see "Adaptive Restart for
    Accelerated Gradient Schemes" by O'Donghue and Candès).
    
    Parameters
    ----------
    
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        Niter : integer
            Maximum number of iterations.
        epsilon: float
            Expected accuracy for the given exit criteria.
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        alpha : float, optional
            Inertial parameter involved in the step 
            y_k=x_k+k/(k+alpha)*(x_k-x_{k-1}). The default value is 3.
        F : operator, optional
            Function to minimize. If the user gives F, the function will 
            compute the value of F at each iterate.
        exit_crit : operator, optional
            Exit criteria, function of x_k and y_k. The default is the norm 
            of (x_k - y_k).
        sp : operator, optional
            Scalar product of reference.
        extra_function : operator, optional
            Additional function to compute at each iterate.
        track_ctime : boolean, optional
            Parameter which states if the function returns an array containing
            the computation time at each iteration. The default value is False.
    Returns
    -------
        
        x : array_like
            Last iterate given by the method.
        cost : array_like, optional
            Array containing the value of F at each iterate. It is returned if
            F is given by the user.
        ctime : array_like, optional
            Array containing the computation time required at each iterate. 
            The last coefficient is the total computation time. It is returned
            if track_ctime is True.
        extra_cost : array_like, optional
            Array containing the value of extra_function at each iterate if a 
            function is given."""
    if sp is None:sp=lambda x,y:np.dot(x,y)
    if F is not None:cost = [F(x)]
    if extra_function is not None:extra_cost = [extra_function(x)]
    if track_ctime:
        extratime = 0
        ctime = [0]
        t0 = time.perf_counter()
    if exit_crit is None:exit_crit = lambda x,y:npl.norm(x-y,2)
    y = np.copy(x)
    i = 0
    k = 0
    out = False
    restart_cond = False
    while i < Niter and out == False:
        i += 1
        k += 1
        xm = np.copy(x)
        x = ForwardBackward_step(y,s,proxh,Df)
        out = exit_crit(x,y)<epsilon
        restart_cond = sp(y-x,x-xm)>0
        if restart_cond:
            k = 0
            y = np.copy(x)
        else:
            y = x+(k-1)/(k+alpha-1)*(x-xm)
        if track_ctime:
            t_temp = time.perf_counter()
            ctime += [t_temp - extratime - t0]
        if F is not None:cost += [F(x)]
        if extra_function is not None:extra_cost += [extra_function(x)]
        if track_ctime:extratime += time.perf_counter()-t_temp
    output = (x,)
    if F is not None:output += (np.array(cost),)
    if track_ctime:output += (np.array(ctime),)
    if extra_function is not None:output += (np.array(extra_cost),)
    if len(output) == 1:output = x
    return output

def FISTA_func_restart(x,s,Niter,epsilon,Df,proxh,F,alpha=3,exit_crit=None,
                       extra_function=None,track_ctime=False,out_cost=False):
    """FISTA empirical restart based on the value of the objective function
    (see "Adaptive Restart for Accelerated Gradient Schemes" by O'Donghue and 
    Candès).
    
    Parameters
    ----------
    
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        Niter : integer
            Maximum number of iterations.
        epsilon: float
            Expected accuracy for the given exit criteria.
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        F : operator
            Function to minimize.
        alpha : float, optional
            Inertial parameter involved in the step 
            y_k=x_k+k/(k+alpha)*(x_k-x_{k-1}). The default value is 3.
        exit_crit : operator, optional
            Exit criteria, function of x_k and y_k. The default is the norm 
            of (x_k - y_k).
        extra_function : operator, optional
            Additional function to compute at each iterate.
        track_ctime : boolean, optional
            Parameter which states if the function returns an array containing
            the computation time at each iteration. The default value is False.
        out_cost : boolean, optional
            Parameter which states if the function returns the evolution of 
            F(x_k). The default value is True.
    Returns
    -------
        
        x : array_like
            Last iterate given by the method.
        cost : array_like, optional
            Array containing the value of F at each iterate. It is returned if
            out_cost is True.
        ctime : array_like, optional
            Array containing the computation time required at each iterate. 
            The last coefficient is the total computation time. It is returned
            if track_ctime is True.
        extra_cost : array_like, optional
            Array containing the value of extra_function at each iterate if a 
            function is given."""
    if extra_function is not None:extra_cost = [extra_function(x)]
    if track_ctime:
        extratime = 0
        ctime = [0]
        t0 = time.perf_counter()
    if exit_crit is None:exit_crit = lambda x,y:npl.norm(x-y,2)
    cost = [F(x)]
    y = np.copy(x)
    i = 0
    k = 0
    out = False
    restart_cond = False
    while i < Niter and out == False:
        i += 1
        k += 1
        xm = np.copy(x)
        x = ForwardBackward_step(y,s,proxh,Df)
        cost += [F(x)]
        out = exit_crit(x,y)<epsilon
        restart_cond = cost[-1]>cost[-2]
        if restart_cond:
            k = 0
            y = np.copy(x)
        else:
            y = x+(k-1)/(k+alpha-1)*(x-xm)
        if track_ctime:
            t_temp = time.perf_counter()
            ctime += [t_temp - extratime - t0]
        if extra_function is not None:extra_cost += [extra_function(x)]
        if track_ctime:extratime += time.perf_counter()-t_temp
    output = (x,)
    if out_cost:output += (np.array(cost),)
    if track_ctime:output += (np.array(ctime),)
    if extra_function is not None:output += (np.array(extra_cost),)
    if len(output) == 1:output = x
    return output
        
    
def FISTA_automatic_restart(x,s,Niter,epsilon,Df,proxh,F,alpha=3,C=6.38,
                            exit_crit=None,out_cost=True,estimated_ratio=1,
                            out_mu=False, extra_function=None,
                            track_ctime=False):
    """Automatic restart of FISTA (method introduced in "FISTA restart using
    an automatic estimation of the growth parameter").
    
    Parameters
    ----------
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        Niter : integer
            Maximum number of iterations.
        epsilon: float
            Expected accuracy for the given exit criteria.
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        F : operator
            Function to minimize.
        alpha : float, optional
            Inertial parameter involved in the step 
            y_k=x_k+k/(k+alpha)*(x_k-x_{k-1}). The default value is 3.
        C : float, optional
            Parameter of the restart method. Large values of C ensure frequent
            restarts. The default value is 6.38 (theoretically optimal).
        exit_crit : operator, optional
            Exit criteria, function of x_k and y_k. The default is the norm of
            (x_k - y_k).
        out_cost : boolean, optional
            Parameter which states if the function returns the evolution of 
            F(x_k). The default value is True.
        estimated_ratio : float, optional
            Low estimation of the condition number. The default value is 1.
        out_mu : boolean, optional
            Parameter which states if the function returns the successive 
            estimations of the growth parameter mu. The default value is False.
        extra_function : operator, optional
            Additional function to compute at each iterate.
        track_ctime : boolean, optional
            Parameter which states if the function returns an array containing
            the computation time at each iteration. The default value is False.
    Returns
    -------
        
        x : array_like
            Last iterate given by the method.
        cost : array_like, optional
            Array containing the value of F at each iterate. It is returned if
            out_cost is True.
        ctime : array_like, optional
            Array containing the computation time required at each iterate. 
            The last coefficient is the total computation time. It is returned
            if track_ctime is True.
        extra_cost : array_like, optional
            Array containing the value of extra_function at each iteration if a 
            function is given.
        growth_estimates : array_like, optional
            Array containing the estimations of the growth parameter mu. It is
            returned if out_mu is True."""
    if exit_crit is None:exit_crit = lambda x,y:npl.norm(x-y,2)
    objective = None
    if out_cost:objective = F
    if out_mu:growth_estimates = np.array([])
    if extra_function is not None: extra_cost = np.array([extra_function(x)])
    if track_ctime:t0 = time.perf_counter()
    i = 0
    n = int(2*C*np.sqrt(estimated_ratio))
    F_tab = [F(x)]
    n_tab = [n]
    if track_ctime:ctime = [time.perf_counter()-t0]
    if out_cost: cost = [F_tab[0]]
    outputs = FISTA(x,s,n,epsilon,Df,proxh,alpha,objective,exit_crit,
                    restarted=True,extra_function=extra_function,
                    track_ctime=track_ctime)
    x = outputs[0]
    j = 1
    if out_cost:
        cost_temp = outputs[j]
        cost = np.concatenate((cost,cost_temp))
        j+=1
    if track_ctime:
        ctime_temp = outputs[j]
        ctime = np.concatenate((ctime,ctime_temp[1:]+ctime[-1]))
        j+=1
    if extra_function is not None:
        extra_cost_temp = outputs[j]
        extra_cost = np.concatenate((extra_cost,extra_cost_temp[1:]))
        j+=1
    out = outputs[j]
    if track_ctime:t_temp = time.perf_counter()
    i += n
    F_tab = np.concatenate((F_tab,[F(x)]))
    n_tab += [n]
    while (i < Niter) and out==False:
        if track_ctime:ctime[-1] += time.perf_counter()-t_temp
        outputs = FISTA(x,s,np.minimum(n,Niter-i),epsilon,Df,proxh,alpha,
                        objective,exit_crit,restarted=True,
                        extra_function=extra_function,track_ctime=track_ctime)
        x = outputs[0]
        j = 1
        if out_cost:
            cost_temp = outputs[j]
            cost = np.concatenate((cost,cost_temp))
            j+=1
        if track_ctime:
            ctime_temp = outputs[j]
            ctime = np.concatenate((ctime,ctime_temp[1:]+ctime[-1]))
            j+=1
        if extra_function is not None:
            extra_cost_temp = outputs[j]
            extra_cost = np.concatenate((extra_cost,extra_cost_temp[1:]))
            j+=1
        out = outputs[j]
        
        if track_ctime:t_temp = time.perf_counter()
        i += np.minimum(n,Niter-i)
        F_tab = np.concatenate((F_tab,[F(x)]))
        tab_mu = (4/(s*(np.array(n_tab)[:-1]+1)**2)*(F_tab[:-2]-F_tab[-1])/
                  (F_tab[1:-1]-F_tab[-1]))
        mu = np.min(tab_mu)
        if out_mu:growth_estimates = np.r_[growth_estimates,mu]
        if (n <= C/np.sqrt(s*mu)):
            n = 2*n
        n_tab += [n]
    output = (x,)
    if out_cost:output += (cost,)
    if track_ctime:output += (ctime,)
    if extra_function is not None:output += (extra_cost,)
    if out_mu:output += (growth_estimates,)
    if len(output) == 1:output = x
    return output
    

    
def FISTA_for_AKL(x,s,Niter,epsilon,Df,proxh,F,alpha=3,exit_crit=None,
                  extra_function=None,track_ctime=False
                 ,out_cost=True):
    """FISTA with the exit condition introduced by Alamo, Krupa and Limon 
    in "Restart FISTA with global linear convergence". 
    
    Parameters
    ----------
    
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        Niter : integer
            Minimum number of iterations.
        epsilon: float
            Expected accuracy for the given exit criteria.
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        F : operator
            Function to minimize. 
        alpha : float, optional
            Inertial parameter involved in the step 
            y_k=x_k+k/(k+alpha)*(x_k-x_{k-1}). The default value is 3.
        exit_crit : operator, optional
            Exit criteria, function of x_k and y_k. The default is the norm 
            of (x_k - y_k).
        extra_function : operator, optional
            Additional function to compute at each iterate.
        track_ctime : boolean, optional
            Parameter which states if the function returns an array containing
            the computation time at each iteration. The default value is False.
        out_cost : boolean, optional
            Parameter which states if the function returns the evolution of 
            F(x_k). The default value is True.
    Returns
    -------
        
        x : array_like
            Last iterate given by the method.
        out : boolean
            Parameter reporting if the exit condition is satisfied at the last
            iterate.
        last_cost : float
            Value of the objective function at the last iterate.
        i : integer
            Number of iterations computed.
        cost : array_like, optional
            Array containing the value of F at each iterate. It is returned if
            out_cost is True.
        ctime : array_like, optional
            Array containing the computation time required at each iterate. 
            The last coefficient is the total computation time. It is returned
            if track_ctime is True.
        extra_cost : array_like, optional
            Array containing the value of extra_function at each iterate if a 
            function is given. """
    if out_cost:cost = []
    if extra_function is not None:extra_cost = [extra_function(x)]
    if track_ctime:
        extratime = 0
        ctime = [0]
        t0 = time.perf_counter()
    if exit_crit is None:exit_crit = lambda x,y:npl.norm(x-y,2)
    y = np.copy(x)
    i = 0
    out = False
    Ec = False
    initial_cost = F(x)
    #mandatory_cost is the table containing only the values of F(x_k) which are 
    #necessary for computing Ec. 
    mandatory_cost = [initial_cost]
    threshold_iter = int(Niter/2)#first iteration k such that F(x_k) is computed
    #it allows to reduce computational cost
    while (i < Niter or Ec==False) and out == False:
        i += 1
        xm = np.copy(x)
        x = ForwardBackward_step(y,s,proxh,Df)
        out = exit_crit(x,y)<epsilon
        y = x+(i-1)/(i+alpha-1)*(x-xm)
        if i >= threshold_iter:
            mandatory_cost += [F(x)]
        if i >= Niter:
            m=int(i/2)-threshold_iter+1
            Ec=((np.exp(1)*(mandatory_cost[m]-mandatory_cost[-1])
                 <=(initial_cost-mandatory_cost[m])) and 
                (mandatory_cost[-1]<=initial_cost))
        if track_ctime:
            t_temp = time.perf_counter()
            ctime += [t_temp - extratime - t0]
        if out_cost and i >= threshold_iter:cost += [mandatory_cost[-1]]
        if out_cost and i < threshold_iter:cost += [F(x)]
        if extra_function is not None:extra_cost += [extra_function(x)]
        if track_ctime:extratime += time.perf_counter()-t_temp
    output = (x,out,mandatory_cost[-1],i,)
    if out_cost:output += (np.array(cost),)
    if track_ctime:output += (np.array(ctime),)
    if extra_function is not None:output += (np.array(extra_cost),)
    return output

def FISTA_restart_AKL(x,s,Niter,n0,epsilon,Df,proxh,F,alpha=3,exit_crit=None,
                      out_cost=True, extra_function=None,track_ctime=False):
    """FISTA restart by Alamo, Krupa and Limon introduced in "Restart FISTA
    with global linear convergence".
    
    Parameters
    ----------
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        Niter : integer
            Maximum number of iterations.
        n0 : integer
            Minimum number of iterations for the first run of FISTA.
        epsilon: float
            Expected accuracy for the given exit criteria.
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        F : operator
            Function to minimize.
        alpha : float, optional
            Inertial parameter involved in the step 
            y_k=x_k+k/(k+alpha)*(x_k-x_{k-1}). The default value is 3.
        exit_crit : operator, optional
            Exit criteria, function of x_k and y_k. The default is the norm of
            (x_k - y_k).
        out_cost : boolean, optional
            Parameter which states if the function returns the evolution of 
            F(x_k). The default value is True.
        extra_function : operator, optional
            Additional function to compute at each iterate.
        track_ctime : boolean, optional
            Parameter which states if the function returns an array containing
            the computation time at each iteration. The default value is False.
    Returns
    -------
        
        x : array_like
            Last iterate given by the method.
        cost : array_like, optional
            Array containing the value of F at each iterate. It is returned if
            out_cost is True.
        ctime : array_like, optional
            Array containing the computation time required at each iterate. 
            The last coefficient is the total computation time. It is returned
            if track_ctime is True.
        extra_cost : array_like, optional
            Array containing the value of extra_function at each iteration if a 
            function is given."""
    if exit_crit is None:exit_crit = lambda x,y:npl.norm(x-y,2)
    if extra_function is not None: extra_cost = np.array([extra_function(x)])
    if track_ctime:t0 = time.perf_counter()
    i = 0
    n = n0
    F_tab = [F(x)]
    if track_ctime:ctime = [time.perf_counter()-t0]
    if out_cost: cost = [F_tab[0]]
    outputs = FISTA_for_AKL(x,s,n,epsilon,Df,proxh,F,alpha,exit_crit,
                            extra_function=extra_function,
                            track_ctime=track_ctime,out_cost=out_cost)
    x,out,lastF,n = outputs[:4]
    j = 4
    if out_cost:
        cost_temp = outputs[j]
        cost = np.concatenate((cost,cost_temp))
        j+=1
    if track_ctime:
        ctime_temp = outputs[j]
        ctime = np.concatenate((ctime,ctime_temp[1:]+ctime[-1]))
        j+=1
    if extra_function is not None:
        extra_cost_temp = outputs[j]
        extra_cost = np.concatenate((extra_cost,extra_cost_temp[1:]))
        j+=1
    if track_ctime:t_temp = time.perf_counter()
    i += n
    F_tab = np.concatenate((F_tab,[lastF]))
    while (i < Niter) and out==False:
        if track_ctime:ctime[-1] += time.perf_counter()-t_temp
        outputs = FISTA_for_AKL(x,s,np.minimum(n,Niter-i-1),epsilon,Df,proxh,
                                F,alpha,exit_crit,
                                extra_function=extra_function,
                                track_ctime=track_ctime,out_cost=out_cost)
        x,out,lastF,n = outputs[:4]
        j = 4
        if out_cost:
            cost_temp = outputs[j]
            cost = np.concatenate((cost,cost_temp))
            j+=1
        if track_ctime:
            ctime_temp = outputs[j]
            ctime = np.concatenate((ctime,ctime_temp[1:]+ctime[-1]))
            j+=1
        if extra_function is not None:
            extra_cost_temp = outputs[j]
            extra_cost = np.concatenate((extra_cost,extra_cost_temp[1:]))
            j+=1
        
        if track_ctime:t_temp = time.perf_counter()
        i += np.minimum(n,Niter-i)
        F_tab = np.concatenate((F_tab,[lastF]))
        if (np.exp(1)*(F_tab[-2]-F_tab[-1]))>(F_tab[-3]-F_tab[-2]):
            n = 2*n
    output = (x,)
    if out_cost:output += (cost,)
    if track_ctime:output += (ctime,)
    if extra_function is not None:output += (extra_cost,)
    if len(output) == 1:output = x
    return output

def FISTA_gradient_restart_AKL(x,s,Niter,epsilon,Df,proxh,F=None,alpha=3,
                               exit_crit=None,extra_function=None,
                               track_ctime=False):
    """FISTA restart scheme introduced by Alamo, Krupa and Limon 
    in "Gradient Based Restart FISTA". 
    
    Parameters
    ----------
    
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        Niter : integer
            Minimum number of iterations.
        epsilon: float
            Expected accuracy for the given exit criteria.
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        F : operator, optional
            Function to minimize. If the user gives F, the function will 
            compute the value of F at each iterate.
        alpha : float, optional
            Inertial parameter involved in the step 
            y_k=x_k+k/(k+alpha)*(x_k-x_{k-1}). The default value is 3.
        exit_crit : operator, optional
            Exit criteria, function of x_k and y_k. The default is the norm 
            of (x_k - y_k).
        extra_function : operator, optional
            Additional function to compute at each iterate.
        track_ctime : boolean, optional
            Parameter which states if the function returns an array containing
            the computation time at each iteration. The default value is False.
    Returns
    -------
        
        x : array_like
            Last iterate given by the method.
        cost : array_like, optional
            Array containing the value of F at each iterate. It F is given by
            the user.
        ctime : array_like, optional
            Array containing the computation time required at each iterate. 
            The last coefficient is the total computation time. It is returned
            if track_ctime is True.
        extra_cost : array_like, optional
            Array containing the value of extra_function at each iterate if a 
            function is given. """
    if F is not None:cost = [F(x)]
    if extra_function is not None:extra_cost = [extra_function(x)]
    if track_ctime:
        extratime = 0
        ctime = [0]
        t0 = time.perf_counter()
    if exit_crit is None:exit_crit = lambda x,y:npl.norm(x-y,2)
    y = np.copy(x)
    x = ForwardBackward_step(y,s,proxh,Df)
    gamma = 1/s*npl.norm(x-y)
    y = np.copy(x)
    i = 0
    k = 0
    out = False
    yp = ForwardBackward_step(y,s,proxh,Df)
    while (i < Niter) and out == False:
        i += 1
        k += 1
        xm = np.copy(x)
        x = np.copy(yp)
        out = exit_crit(x,y)<epsilon
        y = x+(k)/(k+alpha)*(x-xm)
        if track_ctime:
            t_temp = time.perf_counter()
            ctime += [t_temp - extratime - t0]
        if F is not None:cost += [F(x)]
        if extra_function is not None:extra_cost += [extra_function(x)]
        if track_ctime:extratime += time.perf_counter()-t_temp
        
        yp = ForwardBackward_step(y,s,proxh,Df)
        gp = 1/s*npl.norm(y-yp)
        if gp <= gamma/np.exp(1) and out == False:
            i += 1
            k = 0
            gamma = gp
            x = np.copy(yp)
            y = np.copy(yp)
            yp =ForwardBackward_step(y,s,proxh,Df)
            if track_ctime:
                t_temp = time.perf_counter()
                ctime += [t_temp - extratime - t0]
            if F is not None:cost += [F(x)]
            if extra_function is not None:extra_cost += [extra_function(x)]
            if track_ctime:extratime += time.perf_counter()-t_temp
    output = (x,)
    if F is not None:output += (np.array(cost),)
    if track_ctime:output += (np.array(ctime),)
    if extra_function is not None:output += (np.array(extra_cost),)
    return output