#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

class To_Plot:
    def __init__(self,cost,legend,xaxis=None):
        self.cost=cost
        self.legend=legend
        if xaxis is None:
            self.xaxis = np.linspace(0,len(cost)-1,len(cost))
        else:
            self.xaxis = xaxis
        
def Plot(tab,ite=True,costmin=None,plage=None,fontsize=None,style=None,
         eps=1e-8):
    """Function for displaying convergence curves.
    
    Parameters
    ----------
    
        tab : array of To_Plot
            Array containing the curves to plot and associated legends.
        ite : boolean, optional
            Boolean which states if the convergence curves are displayed
            according to the number of iterations or according to the 
            array tab[?].xaxis
        costmin : float, optional
            Minimum F of the function to minimize.
        plage : array, optional
            Array containing two values [start,finish]. The function displays 
            the curves from the start-th iteration to the finish-th iteration.
        fontsize : integer, optional
            Fontsize of the legend.
        style : array, optional
            Array having the same size as tab, containing the curve style of 
            each curve. For example if there are two curves, style=['-','--'] 
            could be a choice.
        eps : float, optional
            Expected precision for the last iterates.
    """
    plt.rcParams["figure.figsize"] = (30,12)
    if fontsize==None:
        fontsize=18
    n=len(tab)
    if costmin is None:
        cost_min=np.min(tab[0].cost)
        for i in range(1,n):
            temp=np.min(tab[i].cost)
            if temp<cost_min:cost_min=temp
        cost_min=cost_min*(1-eps)
    else:
        cost_min=costmin
    if plage is None:
        ite_start=0
        ite_fin=-1
        xmin = np.infty
        xmax = -np.infty
    else:
        ite_start=int(plage[0])
        ite_fin=int(plage[1])
        xmin = plage[0]
        xmax = plage[1]
    tab_leg=[]
    ite_max=0
    for i in range(n):
        tab_leg+=[tab[i].legend]
        if style==None:
            st='-'
        else:
            st=style[i]
        if ite:
            plt.plot(np.log10(tab[i].cost[ite_start:
                                          np.minimum(ite_fin,
                                                     len(tab[i].cost))]
                              -cost_min),linestyle=st)
            n_ite=len(tab[i].cost[ite_start:
                                  np.minimum(ite_fin,len(tab[i].cost))])
            if n_ite>ite_max:ite_max=n_ite
        else:
            plt.plot(tab[i].xaxis,np.log10(tab[i].cost-cost_min),linestyle=st)
            if tab[i].xaxis[0]<xmin and plage is None:xmin=tab[i].xaxis[0]
            if tab[i].xaxis[-1]>xmax and plage is None:xmax=tab[i].xaxis[-1]
    plt.legend(tab_leg,fontsize=fontsize)
    
    if ite:plt.xlim([0,ite_max])
    if ite==False:plt.xlim([xmin,xmax])
    pass
