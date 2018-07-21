""" 
 	Collection of methods that are useful for dealing with optimization 
 	over (scaled) diagonally dominant sums of squares polynomials.
"""

import sympy as sp
import numpy as np

import mosek
from mosek.fusion import *

import sys
from posipoly.utils import k_to_ij, ij_to_k

def _sdd_index(i,j,n):
	""" An n x n sdd matrix A can be written as A = sum Mij.
		Given Mij's stored as a (n-1)*n/2 x 3 matrix, where each row represents a 2x2 symmetric matrix, return
	    the indices i_s, j_s such that A_ij = sum_s Mij(i_s, j_s) """
	num_vars = int(n*(n-1)/2)
	if i == j:
		return [ [ij_to_k(min(i,l), max(i,l)-1, num_vars),(0 if i<l else 1)] for l in range(n) if l != i ]
	else:
		return [[ij_to_k(min(i,j), max(i,j)-1, num_vars),2]]


def add_sdd_mosek(task, start, length):
	''' 
		Given a mosek task with variable vector x,
		add variables and constraints to task such that
		x[ start, start + length ] = vec(A),
		for A a sdd matrix
	'''

	# number of existing variables / constraints
	numvar = task.getnumvar()
	numcon = task.getnumcon()

	assert(start >= 0)
	assert(start + length <= numvar)

	# side of matrix
	n = int((np.sqrt(1+8*length) - 1)/2)
	assert( n == (np.sqrt(1+8*length) - 1)/2 )

	# add new vars and constraints as
	# 
	#   [ old_constr   0  ]  [ old_vars ]    [old_rhs ]
	#   [  0   -I  0    D ]  [ new_vars ]  = [  0     ]
	#
	# where I as at pos start:start+length

	# we need 3 x this many new variables
	numvar_new = n * (n-1) // 2

	# add new variables and make them unbounded
	task.appendvars(3 * numvar_new)
	task.putvarboundslice( numvar, numvar + 3 * numvar_new, 
				[mosek.boundkey.fr] * 3 * numvar_new, 
				[0.] * 3 * numvar_new, 
				[0.] * 3 * numvar_new  )
	
	# add new constraints
	task.appendcons(length)

	# put negative identity matrix
	task.putaijlist( range(numcon, numcon + length), range(start, start+length), [-1.] * length)

	# build 'D' matrix
	D_row_idx = []
	D_col_idx = []
	D_vals = []

	for row in range(length):
		i,j = k_to_ij(row, length)
		sdd_idx = _sdd_index(i,j,n)
		D_row_idx += [numcon + row] * len(sdd_idx) 
		D_col_idx += [numvar + 3*k + l for (k,l) in sdd_idx ]
		D_vals += [ 2. if l == 0 else 1. for (k,l) in sdd_idx ]

	task.putaijlist( D_row_idx, D_col_idx, D_vals ) # add it

	# put = 0 for new constraints
	task.putconboundslice( numcon, numcon + length, [mosek.boundkey.fx] * length, [0.] * length, [0.] * length )

	# add cone constraints
	task.appendconesseq( [mosek.conetype.rquad] * numvar_new, [0.0] * numvar_new, [3] * numvar_new, numvar )

def is_dd(A):
	""" Returns 'True' if A is dd (diagonally dominant), 'False' otherwise """

	epsilon = 1e-10 # have some margin

	A_arr = np.array(A)
	if A_arr.shape[0] != A_arr.shape[1]:
		return False

	n = A_arr.shape[0]
	for i in range(n):
		if not A[i,i] + epsilon >= sum(np.abs(A[i, [j for j in range(n) if i != j]])):
			return False

	return True

def is_sdd(A):
	""" Returns 'True' if A is sdd (scaled diagonally dominant), 'False' otherwise """

	epsilon = 1e-5

	A_arr = np.array(A)
	n = A_arr.shape[0]

	# Define a LP
	M = Model()

	Y = M.variable(n, Domain.greaterThan(1.))
	for i in range(n):
		K_indices = [i,[j for j in range(n) if i != j]]
		Y_indices = [j for j in range(n) if i != j]
		M.constraint( Expr.sub( Expr.mul(A_arr[i,i], Y.index(i) ), 
					            Expr.dot(np.abs(A_arr[K_indices]).tolist(), 
					            	Y.pick(Y_indices) )
					          ),
					          Domain.greaterThan(-epsilon)
					) 

	M.objective(ObjectiveSense.Minimize, Expr.sum(Y))

	M.solve()

	return False if M.getDualSolutionStatus() == SolutionStatus.Certificate else True