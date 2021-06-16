#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:35:28 2021

@author: jayanth
"""

import numpy as np
import tensorflow as tf
import math
from scipy import integrate


time = np.linspace(0,1,10)
L = np.linspace(0,1,10)
n = np.zeros((10,10))
c = np.zeros((10,10))
## Temperature calculation
def T(t):
    return 33 - 2*(1-math.exp(-t/100))


## sigma calculation 
def sigma(c,t):
    csat = 0.00548 - 1.93*0.0001*T(t) + 1e-06*T(t)**2
    return abs(c/csat - 1)


def dd(x1,x2):
    if(x1==x2):
        return 1
    else:
        return 0 
    
def integral1(l):
    x = tf.constant([[curr_time,np.array(l,dtype='float64')]])
    k = model(x)
    k = k.numpy()
    h = k[0][0]
    return (2/l)*1e-6*l**1.25*h
   
    
def integral2(l):
    x = tf.constant([[curr_time,np.array(l,dtype='float64')]])
    k = model(x)
    k = k.numpy()
    h = k[0][1]
    return 0.337*sigma(k[0][1],curr_time)*l**2*h    

def volume(l):
    x = tf.constant([[curr_time,np.array(l,dtype='float64')]])
    k = model(x)
    k = k.numpy()
    h = k[0][0]
    return l**2*h


## Cost function for the DNN

def cost_function(grads,model,l,t):
    
    ## extracting gradients
    deln_delt = grads[0][0][0][0]
    delc_delt = grads[0][0][0][1]
    deln_dell = grads[0][1][0][1]
    
    ## computing integral terms
    t1 = integrate.romberg(integral1,l+1e-10,1)
    t2 = integrate.romberg(integral2,0.000000000000006,1)
    vol = integrate.romberg(volume,1e-6,1)
    
    
    nf = 1/(0.04*(2*math.pi))**0.5 * (1/200) * math.exp(-(l-0.15)**2/(2*0.04**2))
    cf = 5.48*1e-3 - 1.93*1e-4*313 + 7.09*1e-6*313**2
    
    ## cinputing model output
    x = tf.constant([[np.array(t,dtype='float64'),np.array(l,dtype='float64')]])
    k = model(x)
    k = k.numpy()
    
    n_cost = deln_delt + 0.337*sigma(k[0][1],t)*deln_dell - 1.39*vol*1e5*sigma(k[0][1],t)**2.62*dd(l,0.001) - t1 + (1e-6*l**1.25*k[0][0]) - (nf-k[0][0])/0.2
    c_cost = delc_delt + 3*1412*0.32*t2 + 1412*0.32*1.39*1e5*vol*sigma(k[0][1],t)**2.62*0.001**3 - (1/0.2)*(cf-k[0][1])
    
    ## Computing boundary condition costs
    x = tf.constant([[np.array(0.0,dtype='float64'),np.array(l,dtype='float64')]])
    k = model(x)
    k = k.numpy()
    n0 = k[0][0]
    c0 = k[0][1]
    
    ## Computing bc2
    x = tf.constant([[t,100.0]])
    k2 = model(x)
    k2 = k2.numpy()
    ninf = k2[0][0]
    
    cost = abs(n_cost) + abs(c_cost) + 100000*abs(n0-nf) + 100000*abs(c0-cf) + 100000*abs(ninf)
    
    return cost

def dummy_cost_func(y_true,y_pred):
    
    return tf.math.abs(cost)
    
    
for i in range(0,np.size(time)):
    
    for j in range(0,np.size(L)):
        
        # Initializing DNN
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(30, input_shape=(2,),activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(30,activation='relu'))
        model.add(tf.keras.layers.Dense(2,activation='relu'))
        
        
        ## Gradient Tape for automatic differentitation
        with tf.GradientTape() as tape:
             x = tf.constant([[time[i],L[j]]])
             tape.watch(x)
             pred= model(x)
                
        ## Jacobian matrix for all outputs wrt inputs         
        grad = tape.jacobian(pred, x)
        g = grad.numpy()
        
        
        global cost
        global curr_time
        
        curr_time = time[i]
        
        cost = cost_function(g,model,L[j],time[j])
        
        model.compile(loss=dummy_cost_func, optimizer='adam')
        
        x = tf.constant([[time[i],L[j]]])
        k = model(x)
        k = k.numpy()
        n[i,j] = k[0][0]
        c[i,j] = k[0][1]


    
    


