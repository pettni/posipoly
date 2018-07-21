from math import ceil, sqrt

def k_to_ij(k, L):
  """ 
  Given a symmetric matrix Q represented by a vector
    V = [Q_00, ... Q_0n, Q_11, ..., Q_1n, Q22, ..., ] of length L,
    for given k, compute i,j s.t. Q_ij = V[k]
  """

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
  # Given a symmetric matrix Q represented by a vector
  # V = [Q_00, ... Q_0n, Q_11, ..., Q_1n] of length L,
  # for given i,j , compute k s.t. Q_ij = V(k)
  n = (sqrt(1+8*L) - 1)/2
  i_at1 = min(i,j)+1
  j_at1 = max(j,i)+1
  k_at1 = int((n + n-i_at1)*(i_at1-1)/2 + j_at1)
  return k_at1 - 1


def vec_to_mat(vec):
  # convert vector to square matrix
  L = len(vec)
  n = int((sqrt(1+8*L) - 1)/2)

  return [[vec[ij_to_k(i,j,L)] for i in range(n) ] for j in range(n)]


def mat_to_vec(mat):

  n = mat.shape[0]
  L = int(n*(n+1)/2)

  ret = [0. for i in range(L)]
  for k in range(L):
    i,j = k_to_ij(k, L)
    ret[k] = mat[i,j]

  return ret
