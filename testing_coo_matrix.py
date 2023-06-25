#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 18:55:05 2022

@author: krishnanganesh
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix


# Timing some stuff :
    
import time                                                
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print( '%r (%r, %r) % sec' % \
              (method.__name__, args, kw, te-ts))
        return te - ts

    return timed

# Found a nice decorator for timing functions on StackExchange. Just add @timeit

# Okay I wanna test if generating a numpy array and then sparsifying is more/less efficient than populating the sparse matrix in coo format

# The 4th-year krishnan way:
    

N = 5
a = np.arange(N)

@timeit    
def make_matrix(N):
    M = np.zeros(shape = (N , N) , dtype = int)
    n = 0
    for i in range(N):
            M[i , (i + 1)%N] = a[n] + a[(n+1)%N]
            M[(i+1)%N , i] =  a[n] + a[(n - 1)%N]
            n += 1
   

@timeit    
def make_matrix_sparse(N):
    M = np.zeros(shape = (N , N) , dtype = int)
    n = 0
    for i in range(N):
            M[i , (i + 1)%N] = a[n] + a[(n+1)%N]
            M[(i+1)%N , i] =  a[n] + a[(n - 1)%N]
            n += 1
    M_sparse = csr_matrix(M)



# So I get something that's almost 5 times longer. 

# Now doing it using coo_matrices:
    
# Coo_matrix is apparently a fast way to construct large sparse matrices.
# You need to supply it with an ordered arrays of row and column indices for the values in the matrix.
# And an array of the corresponding matrix element values. Then it constructs the matrix based on this info using magic or something.

# constructing row indices:
row = np.concatenate((a , (a + 1)%N))
col = np.concatenate(((a + 1)%N , a))

@timeit
def make_matrix_coo(N):
    val1 = np.zeros(N)
    val2 = np.zeros(N)
    n = 0
    for i in range(N):
        val1[i] = a[n]+ a[(n+1)%N]
        val2[i] = a[n] + a[(n - 1)%N]
        n += 1
    val = np.concatenate((val1 , val2))
    M_sparse_coo = coo_matrix((val , (row , col)) , shape = (N , N), dtype = int)
    

    

f = make_matrix(N)
g = make_matrix_sparse(N)
h = make_matrix_coo(N)

print('sparsify/matrix:' + '' + str((g / f)))

print('sparsify/coo_matrix :' + '' + str((g/h)))

print('coo_matrix/matrix:' + '' + str(h/f))


# Nice so coo_matrix on average is able to match the normal numpy matrix function.
# Just need to try this for some of the more complicated matrices in skyrmpy_functions.








