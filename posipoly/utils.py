from math import ceil, sqrt

from scipy.special import binom
import scipy.sparse as sp

def veclen_to_matsize(L):
  n = (sqrt(1+8*L) - 1)/2
  assert n == int(n)
  return int(n)

def matsize_to_veclen(n):
  return n*(n+1)//2

def k_to_ij(k, L):
  '''
  Given a gram matrix Q represented by a vector
    V = [Q_00, ... Q_0n, Q_11, ..., Q_1n, Q22, ..., ] of length L,
    for given k, compute i,j s.t. Q_ij = V[k]
  '''
  if (k >= L):
    raise IndexError("Index out of range")
  # inverse formula for arithmetic series
  n = (sqrt(1+8*L) - 1)/2
  # get first index
  i = int(ceil( (2*n+1)/2 - sqrt( ((2*n+1)/2)**2 -2 * (k+1)  ) ) - 1)
  # second index
  k1 = (2*n+1-i)*i/2 - 1
  j = int(i + k - k1 - 1)
  return i,j

def ij_to_k(i,j,L):
  ''' 
  Given a symmetric matrix Q represented by a vector
  V = [Q_00, ... Q_0n, Q_11, ..., Q_1n] of length L,
  for given i,j , compute k s.t. Q_ij = V(k)
  '''
  n = (sqrt(1+8*L) - 1)/2
  i_at1 = min(i,j)+1
  j_at1 = max(j,i)+1
  k_at1 = int((n + n-i_at1)*(i_at1-1)/2 + j_at1)
  return k_at1 - 1

def vec_to_mat(vec):
  '''convert vector representation of gram matrix to gram matrix'''
  L = len(vec)
  n = int((sqrt(1+8*L) - 1)/2)
  return [[vec[ij_to_k(i,j,L)] for i in range(n) ] for j in range(n)]

def mat_to_vec(mat):
  '''retrieve vector representation of gram matrix'''
  n = mat.shape[0]
  L = int(n*(n+1)/2)
  ret = [0. for i in range(L)]
  for k in range(L):
    i,j = k_to_ij(k, L)
    ret[k] = (mat[i,j] + mat[j,i]) / 2
  return ret

def multinomial(params):
  if len(params) == 1:
    return 1
  return binom(sum(params), params[-1]) * multinomial(params[:-1])

def double_factorial(n):
  ret = 1
  for i in range(n, 0, -2):
    ret *= i
  return ret

def speye(n, pos=0, tot=None):
  '''
  return a sparse identity matrix [ 0  I  0]
  with size n x tot and where the identity matrix
  starts at pos
  '''
  if tot is None:
    tot = n
  return sp.coo_matrix( ([1.] * n, (range(n), range(pos, pos+n))), (n,tot) )

def spzeros(n, m):
  return sp.coo_matrix( (n,m) )
