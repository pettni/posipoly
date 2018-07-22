""" 
  Collection of methods that are useful for dealing with optimization 
  over (scaled) diagonally dominant sums of squares polynomials.
"""

from math import sqrt
import numpy as np
import scipy.sparse as sp

import mosek

from posipoly.utils import *

def sdd_index(i,j,n):
  """ An n x n sdd matrix A can be written as A = sum Mij.
    Given Mij's stored as a (n-1)*n/2 x 3 matrix, where each row represents a 2x2 symmetric matrix, return
      the indices i_s, j_s such that A_ij = sum_s Mij(i_s, j_s) """
  num_vars = int(n*(n-1)/2)
  if i == j:
    return [ [ij_to_k(min(i,l), max(i,l)-1, num_vars),(0 if i<l else 1)] for l in range(n) if l != i ]
  else:
    return [[ij_to_k(min(i,j), max(i,j)-1, num_vars),2]]


def setup_ppp(c, Aeq, beq, Aiq, biq, ppp_list, pp_cone):
  '''set up a positive polynomial programming problem
      min    c' x
      s.t.   Aeq x = beq
             Aiq x <= biq

             x[ ppp_list[i][0], ppp_list[i][0] + ppp_list[i][1] ]   for i in range(len(ppp_list)) 
             is a positive matrix C stored as a list 
             [c00 c01 ... c0n  c11 c12 ... c1n ... cnn] 
             of length n * (n+1)/2

     positivity constraints are enforces according to the value of pp_cone
     pp_cone = 'psd'   C is positive semi-definite
     pp_cone = 'sdd'   C scaled diagonally dominant
  '''
  numvar = len(c)

  if Aeq is None:
    Aeq = sp.coo_matrix( (0, numvar) )
    beq = np.zeros(0)

  if Aiq is None:
    Aiq = sp.coo_matrix( (0, numvar) )
    biq = np.zeros(0)

  if type(Aeq) is not sp.coo_matrix:
    Aeq = sp.coo_matrix(Aeq)

  if type(Aiq) is not sp.coo_matrix:
    Aiq = sp.coo_matrix(Aiq)

  numcon_eq = Aeq.shape[0]
  numcon_iq = Aiq.shape[0]

  if numcon_eq != len(beq) or numcon_iq != len(biq) :
    raise Exception('invalid dimensions')

  if Aeq.shape[1] != numvar or Aiq.shape[1] != numvar:
    raise Exception('invalid dimensions')

  env = mosek.Env() 
  task = env.Task(0,0)

  # Add free variables and objective
  task.appendvars(numvar)
  task.putvarboundslice(0, numvar, [mosek.boundkey.fr] * numvar, [0.]*numvar, [0.]*numvar )
  task.putcslice(0, numvar, c)
  task.putobjsense(mosek.objsense.minimize)

  # add eq & iq constraints
  A = sp.bmat([[Aeq], [Aiq]])
  task.appendcons(numcon_eq + numcon_iq)
  task.putaijlist(A.row, A.col, A.data)
  task.putconboundslice(0, numcon_eq, [mosek.boundkey.fx] * numcon_eq, beq, beq )
  task.putconboundslice(numcon_eq, numcon_eq+numcon_iq, [mosek.boundkey.up] * numcon_iq, [0.] * numcon_iq, biq )

  # add pp constraints
  for start, length in ppp_list:
    if pp_cone == 'psd':
      add_psd_mosek(task, start, length)
    elif pp_cone == 'sdd':
      add_sdd_mosek(task, start, length)
    else:
      raise Exception('unknown cone')

  return task

def solve_ppp(c, Aeq, beq, Aiq, biq, ppp_list, pp_cone='sdp'):
  '''solve a positive polynomial programming problem
      min    c' x
      s.t.   Aeq x = beq
             Aiq x <= biq

             x[ ppp_list[i][0], ppp_list[i][0] + ppp_list[i][1] ]   for i in range(len(ppp_list)) 
             is a positive matrix C stored as a list 
             [c00 c01 ... c0n  c11 c12 ... c1n ... cnn] 
             of length n * (n+1)/2

     positivity constraints are enforces according to the value of pp_cone
     pp_cone = 'psd'   C is positive semi-definite
     pp_cone = 'sdd'   C scaled diagonally dominant
  '''

  task = setup_ppp(c, Aeq, beq, Aiq, biq, ppp_list, pp_cone)

  task.optimize()

  solsta = task.getsolsta(mosek.soltype.itr)

  if (solsta == mosek.solsta.optimal):
      solution = [0.] * len(c)
      task.getxxslice(mosek.soltype.itr, 0, len(c), solution)
      return solution, solsta
  else:
      return None, solsta

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
  n = int((sqrt(1+8*length) - 1)/2)
  assert( n == (sqrt(1+8*length) - 1)/2 )

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
  task.putaijlist( range(numcon, numcon+length), range(start, start+length), [-1.] * length)

  # build 'D' matrix
  D_row_idx = []
  D_col_idx = []
  D_vals = []

  for row in range(length):
    i,j = k_to_ij(row, length)
    sdd_idx = sdd_index(i,j,n)
    D_row_idx += [numcon + row] * len(sdd_idx) 
    D_col_idx += [numvar + 3*k + l for (k,l) in sdd_idx ]
    D_vals += [ 2. if l == 0 else 1. for (k,l) in sdd_idx ]

  task.putaijlist( D_row_idx, D_col_idx, D_vals ) # add it

  # put = 0 for new constraints
  task.putconboundslice( numcon, numcon + length, [mosek.boundkey.fx] * length, [0.] * length, [0.] * length )

  # add cone constraints
  task.appendconesseq( [mosek.conetype.rquad] * numvar_new, [0.0] * numvar_new, [3] * numvar_new, numvar )


def add_psd_mosek(task, start, length):
  ''' 
    Given a mosek task with variable vector x,
    add variables and constraints to task such that
    x[ start, start + length ] = vec(A),
    for A an psd matrix
  '''

  # number of existing variables / constraints
  numvar = task.getnumvar()
  numbarvar = task.getnumbarvar()
  numcon = task.getnumcon()

  assert(start >= 0)
  assert(start + length <= numvar)

  # side of matrix
  n = int((sqrt(1+8*length) - 1)/2)
  assert( n == (sqrt(1+8*length) - 1)/2 )

  # add semidef variable
  task.appendbarvars([n])

  # add length equality constraints
  task.appendcons(length)
  
  # add constraint that x[ start, start+length ] is equal to sdp var
  for k in range(length):
    i, j = k_to_ij(k, length) 
    mat_k = task.appendsparsesymmat(n, [j], [i], [1. if j==i else 0.5])
    task.putarow(numcon + k, [start + k], [-1.])
    task.putbaraij(numcon + k, numbarvar, [mat_k], [1.])

  # put = 0 for new constraints
  task.putconboundslice( numcon, numcon + length, [mosek.boundkey.fx] * length, [0.] * length, [0.] * length )


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

  env = mosek.Env() 
  task = env.Task(0,0)

  n = A.shape[0]
  assert(A.shape[0] == A.shape[1])
  
  vec = mat_to_vec(A)

  numvar = int(n*(n+1)/2)
  numcon = numvar

  # add variables and constraints
  task.appendvars(numvar)
  task.appendcons(numcon)

  # make vars unbounded (fr: free)
  task.putvarboundslice(0, numvar, [mosek.boundkey.fr] * numvar, [0.]*numvar, [0.]*numvar )

  # add constraints
  task.putaijlist(range(numvar), range(numvar), [1 for i in range(numvar)])
  task.putconboundslice(0, numvar, [mosek.boundkey.fx] * numcon, vec, vec)

  add_sdd_mosek(task, 0, numcon)

  task.optimize()

  if task.getsolsta(mosek.soltype.itr) != mosek.solsta.prim_infeas_cer:
    return True

  return False
