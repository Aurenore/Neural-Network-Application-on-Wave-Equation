# Neural-Network-Application-on-Wave-Equation
CSE Semester Project

## Abstract 
The 1D wave equation describes the physical phenomena of mechanical waves or electromagnetic waves. We consider this equation with an initial condition which is a linear combination of sinusoidal functions, where the weights depend on some instances of i.i.d random variables following a uniform distribution.  

The aim of this study is to approximate the solution of the wave equation at the final time given the instances of the random variables, employing a deep neural network. In order to achieve this task, a multi-layer perceptron is used, coupled with the grid-search method. The performance of the machine learning algorithm is studied with respect to the dimensions of the input and of the output of the model, and is compared to the performance of a finite difference scheme.

In the end, we found that the neural network achieves to approximate the solution well. However, the architecture of the model depends highly on the input/output dimensions. 

## Structure
The written report can be found under the name of ... . The rest consists of the code used for the report, going from the implementations of the Newmark scheme, to neural networks and various plots. 
To run the code, the following packages are needed: numpy, matplotlib, torch, pandas, math, time.

### Helpers
Helpers functions are contained in "functions.py". This file is imported in the next notebooks.

### Notebooks
The notebooks are organized according to the report. 

#### 1. Introduction
"plot_condition_and_sol.ipynb" plots the initial condition and final solution with respect to K. 

#### 2. Finite difference approximation of the 1D wave equation
"convergence Newmark.ipynb" checks the convergence of the Newmark scheme. Moreover, "Newmark K=2" checks that the difference is null for several $N_\mu$ and $N_h$.

#### 3.2.1 Integration of $\exp{\left(-\frac{1}{2}x^2\right)}$
"MonteCarlo.ipynb" checks the Monte-Carlo method on the approximation of the integration of an exponential. 

#### 3.2.2. Integration of the solution
"Integration+MC.ipynb" checks the Monte-Carlo method on the approximation of the integration of the solution.

#### 4.3 Implementation
"grid search K=2.ipynb" applies the grid search method for $K=2$, $N_h = 39$ and $N_\mu = 5$.

#### 4.4 Convergence of $\frac{1}{N_{M}} \left \Vert \mathbf{u}(\boldsymbol{\mu}) - \mathbf{u}_{DNN}(\boldsymbol{\mu}; \tilde{\boldsymbol{\Theta}})\right \Vert ^2$
"train table conv NN K=2.ipynb" trains different models. 
"table conv NN.ipynb" computes average final errors. 
