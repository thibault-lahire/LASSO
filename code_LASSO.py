#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:46:00 2019

@author: macbookthibaultlahire
"""

########################################################

# Some useful imports

########################################################

import numpy as np
import matplotlib.pyplot as plt


########################################################

# Parameters setting

########################################################


global A
global b
global Q
global p
global X
global y

toy_problem = 1 # change the value of this variable in a second time

if toy_problem == 1:
    n = 2
    d = 1

    X = np.zeros((n, d))
    X[0][0] = 1
    X[1][0] = 2
    y = np.asarray([1,2])

    lambd = 10.0

    Q = 0.5 * np.eye(n)
    p = y
    A = np.concatenate((X.T, -X.T))
    b = lambd * np.ones(2*d)
    
else:
    n = 8
    d = 1200

    np.random.seed(123) # you can comment this line

    # data generation
    X = np.random.rand(n, d)
    hidden_w = np.zeros(d)
    for i in range(d):
        if np.random.rand()<0.01:
            hidden_w[i] = np.random.rand()
    y = X.dot(hidden_w) + np.random.normal(loc = 0, scale = 0.1, size = n)


    lambd = 10.0

    Q = 0.5 * np.eye(n)
    p = y
    A = np.concatenate((X.T, -X.T))
    b = lambd * np.ones(2*d)
    


########################################################

# Some basic functions

########################################################

def lasso_objective(w):
    return 0.5 * np.linalg.norm(X.dot(w) - y) ** 2 + lambd * np.linalg.norm(w, 1)

def is_in_domain(v):
    return np.all((A.dot(v) - b) <= 0)
    
def compute_obj(v, t):
    obj_QP = v.dot(Q.dot(v)) + p.dot(v)
    barrier = - np.sum(np.log(b - A.dot(v)))
    return t * obj_QP + barrier
    
def compute_grad(v, t):
    grad = 2 * Q.dot(v) + p
    barrier_grad = A.T.dot(1/(b - A.dot(v)))
    return t * grad + barrier_grad
    
def compute_hessian(v, t):
    hessian = 2 * Q        
    barrier_hessian = A.T.dot(np.diag(1/(b - A.dot(v))) ** 2).dot(A)
    return t * hessian + barrier_hessian





def backtracking_line_search(v, direction, alpha, beta, t):
    step_size = 1.0
    obj_in_v = compute_obj(v, t)
    grad_in_v = compute_grad(v, t)
    v_trial = v + step_size * direction
        
    while not(is_in_domain(v_trial)) or compute_obj(v_trial, t) > obj_in_v + alpha*step_size * grad_in_v.dot(direction):
            
        step_size *= beta
        v_trial = v + step_size * direction
            
    return step_size


########################################################

# Implementation of the function centering_step
# Here is the Newton method

########################################################


def centering_step(t, v0, eps):
    
    v = v0    
    v_seq = [v0]

    grad = compute_grad(v, t)
    hessian = compute_hessian(v, t)
    direction = - np.linalg.inv(hessian).dot(grad)
    stop_criterion = np.dot(grad, - direction) / 2

    while stop_criterion > eps:
        
        step_size = backtracking_line_search(v, direction, 0.25, 0.5, t)
        
        v = v + step_size * direction
        v_seq.append(v)      
        
        grad = compute_grad(v, t)
        hessian = compute_hessian(v, t)
        direction = - np.linalg.inv(hessian).dot(grad)
        stop_criterion = np.dot(grad, - direction) / 2
    
    return v_seq




########################################################

# Implementation of the function barr_method

########################################################


def barr_method(t0, v0, mu, eps, eps_centering):
    
    m = len(A)
    t = t0
    
    v_array = []
    t_array = [] # to visualize the duality gap
    number_nt_steps = [] # the number of Newton (inner) steps
    
    outer_steps = centering_step(t, v0, eps_centering)
    
    v_array.append(outer_steps[-1])
    t_array.append(t)
    number_nt_steps.append(len(outer_steps) - 1)
    
    while m/t >= eps:
        t = t*mu
            
        outer_steps = centering_step(t, v_array[-1], eps_centering)
        
        v_array.append(outer_steps[-1])
        t_array.append(t)
        number_nt_steps.append(number_nt_steps[-1] + len(outer_steps) - 1)
    
    return v_array, t_array, number_nt_steps


########################################################

# Run simulations

########################################################


def first_tests(t0, v0, mu, eps, eps_centering):   
    '''
    The idea of this function is just to check is the methods are well-implemented.
    The discussions on "elaborated results" (i.e. value of mu, size of the support of w, etc...) 
    will be found by executing the function "further_tests"
    '''
    v_array, t_array, number_nt_steps = barr_method(t0, v0, mu, eps, eps_centering)
    new_v_array = np.asarray(v_array).T
    # solution for lasso
    dual_point = - 1 / (t_array[-1] * (A.dot(v_array[-1]) - b))
    w_found = dual_point[d:] - dual_point[:d]
    
    print("Point optimizing the QP problem :",v_array[-1])
    print("Value of the cost function of the lasso problem for the weights w associated to the point optimizing the QP problem:",w_found)
    print("\n")
    
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(0,0) # starting point of the optimization process
    ax.scatter(new_v_array[0,:-1],new_v_array[1,:-1]) # all points are blue
    ax.scatter(new_v_array[0,-1],new_v_array[1,-1], c='red') # except the last one

    ax.set_ylabel('$v_2$')
    ax.set_xlabel('$v_1$')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)


    # we define a function similar to compute_obj for layout purposes only
    def compute_obj_adapt(v,t):
        if is_in_domain(v)==False:
            return 100
        else:
            obj_QP = v.dot(Q.dot(v)) + p.dot(v)
            barrier = - np.sum(np.log(b - A.dot(v)))
            return t * obj_QP + barrier

    x , y = np.meshgrid(np.linspace(-5,5,401),np.linspace(-5,5,401))
    z = np.zeros_like(x)

    for i in range(0,len(x)):
        for j in range(0,len(x[0])):
            aux = np.asarray([x[0][i],x[0][j]])
            z[i][j] = compute_obj_adapt(aux,20)

    ax.contour(x,y,z,6)
    fig.savefig('first_tests_1.png')



def further_tests(t0, v0, mu_tab, eps, eps_centering):

    obj_array = []
    duality_gap_array = []
    iterations = []
    w_found_array = []

    for mu in mu_tab:
        v_array, t_array, number_nt_steps = barr_method(t0, v0, mu, eps, eps_centering)
    
        # the aim is to see the decrease of the QP objective function and the duality gap over iterations
        obj_array.append([v.dot(Q.dot(v)) + p.dot(v) for v in v_array])
        duality_gap_array.append([len(A) / t for t in t_array])
        iterations.append(number_nt_steps)
    
        # solution for lasso
        dual_point = - 1 / (t_array[-1] * (A.dot(v_array[-1]) - b))
        w_found_array.append(dual_point[d:] - dual_point[:d])


    
    print("Comparison between hidden_w and w found after optimization : \n\n")
    print("Value of the cost function of the primal problem (lasso):\n")
    print("For hidden_w :",lasso_objective(hidden_w))
    print("For w found :",lasso_objective(w_found_array[-1]))
    print("\n\n")
    print("Size of the support of w :\n")
    print("For hidden_w :",np.size(np.where(hidden_w != 0)))
    e_moy = np.exp((np.log(np.max(w_found_array[-1]))+np.log(np.min(w_found_array[-1])))/2)
    print("For w found :",np.size(np.where(w_found_array[-1] > e_moy)))  


    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[0].set_title('Decrease of the gap between the optimal value and the value of the cost function over iterations')
    ax[0].axhline(y=eps_centering, color='black', label='centering precision')
    for i in range(len(obj_array)):
        ax[0].plot(iterations[i][:-1], obj_array[i][:-1] - obj_array[i][-1], 'o', linestyle='-', label='$\mu={}$'.format(mu_tab[i]))
    ax[0].set_yscale('log')
    ax[0].set_ylabel('Gap $f(v_t)-f^{*}$')
    ax[0].set_xlabel('Newton iterations')
    ax[0].legend()
    

    ax[1].set_title('Decrease of the duality gap over iterations')
    ax[1].axhline(y=eps, color='black', label='global precision')
    for i in range(len(duality_gap_array)):
        ax[1].step(iterations[i], duality_gap_array[i], linestyle='-', label='$\mu={}$'.format(mu_tab[i]))
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Duality gap')
    ax[1].set_xlabel('Newton iterations')
    ax[1].legend()

    fig.tight_layout()
    fig.savefig('further_tests.png')


########################################################

# Execution for a certain initialization discussed 
# in the joint pdf

########################################################


v0 = np.zeros(n)
t0 = 10.0
eps = 1e-6
eps_centering = 1e-6
mu = 20

mu_tab = [2, 15, 50, 100, 300, 500]

if toy_problem == 1:
    first_tests(t0, v0, mu, eps, eps_centering)
else:
    further_tests(t0, v0, mu_tab, eps, eps_centering)





