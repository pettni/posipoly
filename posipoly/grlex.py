'''utilities for grlex ordering'''

from math import sqrt, ceil
from functools import cmp_to_key
import scipy.sparse
from scipy.misc import comb

def count_monomials_leq(n, d):
  '''Number of monomials in n variables of degree less than or equal to d'''
  return int(comb(n+d, d))

def count_monomials_eq(n, d):
  '''Number of monomials in n variables of degree equal to d'''
  return int(comb(n+d-1, d))

def grlex_to_index(multiindex):
  """Returns the grlex ordering number for a given multi-index. Can be expensive for large degrees"""
  total_degree = sum(multiindex)
  n = len(multiindex)

  index = count_monomials_leq(n, total_degree - 1)
  remaining_degree = total_degree
  for i in range(n):
    for k in range(multiindex[i]):
      index += count_monomials_eq(n-1-i, remaining_degree - k)
    remaining_degree -= multiindex[i]
  return index

def index_to_grlex(index, n):
  """Returns the multi-index of length n for a given grlex ordering number"""
  grlex = [0 for i in range(n)]

  # find sum of grlex
  total_degree = 0
  while count_monomials_leq(n, total_degree) <= index:
    total_degree += 1

  # index among all multiindices with sum total_degree
  rel_idx_search = index - count_monomials_leq(n, total_degree - 1)

  rem_deg = total_degree # keep track of sum of remaining indices
  rel_idx = 0 # keep track of number of multiindices to the left 
  for i in range(n-1):
    for k in range(total_degree+1):
      new_rel_idx = rel_idx + count_monomials_eq(n-(i+1), rem_deg-k)
      if new_rel_idx > rel_idx_search:
        grlex[i] = k
        rem_deg -= k
        break
      rel_idx = new_rel_idx

  grlex[-1] = rem_deg

  return tuple(grlex)


def grlex_comp(midx1, midx2):
  '''comparison function in grlex'''
  return (sum(midx1) > sum(midx2)) - (sum(midx1) < sum(midx2)) or  \
         (midx1 > midx2) - (midx1 < midx2)

def multi_grlex_iter(midx, groups, degrees):
  '''
  Create an iterator that produces ordered exponents, starting
  with the multiindex 'midx', such that total degree of the variables
  in groups[i] does not exceed degrees[i].

  The ordering is based on grlex groupwise, but does not result
  in an overall grlex ordering.

  Example: The iterator grlex_iter( (0,0), [ [0], [1] ], [2], [1] ) 
       produces the sequence
    (0,0) (1,0) (2,0) (0,1) (1,1) (2,1)
  '''

  # make sure 'groups' is a valid partition
  assert(set([dim for group in groups for dim in group]) == set(range(len(midx))))
  assert(len(groups) == len(degrees))

  # starting multiindices
  start_midx = [ tuple([ midx[dim] for dim in group ]) for group in groups ]

  iterators = [grlex_iter(m0, deg) for m0, deg in zip(start_midx, degrees) ]
  mons = [next(iterator) for iterator in iterators]

  ret = list(midx)
  while True:

    yield tuple(ret)

    for ind in range(len(degrees)):
      # find first that is not at max
      try:
        mons[ind] = next(iterators[ind])
        break
      except StopIteration:

        if ind == len(degrees) - 1:
          raise StopIteration
        
        # was at max, reset it
        iterators[ind] = grlex_iter(start_midx[ind], degrees[ind])
        mons[ind] = next(iterators[ind])

    # Fill out return tuple
    for group_nr, group in enumerate(groups):
      for pos_nr, pos in enumerate(group):
        ret[pos] = mons[group_nr][pos_nr]

def grlex_iter(midx, deg = -2):
  '''
  Create an iterator that produces ordered grlex exponents, starting
  with the multiindex 'midx'. The iterator stops when the total degree 
  reaches 'deg'+1. (default: never stop)

  Example: The iterator grlex_iter( (0,2) ) produces the sequence
    (0,2) (1,2) (2,0) (0,3) (1,2) (2,1) (3,0) ...
  '''

  midx = list(midx)

  assert(min(midx) >= 0)

  if max(midx) == 0:
    right_ptr = 0
  else:
    # find right-most non-zero element
    for i in range(len(midx)-1, -1, -1):
      if midx[i] != 0:
        right_ptr = i
        break

  while True:

    if sum(midx) == deg + 1:
      raise StopIteration

    yield tuple(midx)

    if right_ptr == 0:
      midx = [0] * (len(midx) - 1) + [sum(midx) + 1]
      right_ptr = len(midx) - 1
    else:
      at_right_ptr = midx[right_ptr]
      midx[right_ptr] = 0
      midx[right_ptr-1] += 1
      midx[-1] = at_right_ptr - 1
      right_ptr = len(midx) - 1 if at_right_ptr != 1 else right_ptr - 1

def vec_to_grlex(L, dim):
  ''' 
  Given vec(A) for a symmetric matrix A, such that len(vec(A)) = L,
  compute the mapping  vec(A) |--> x^T A x 
  to the grlex dim-dimensional monomial basis on multiindex - coefficient form.

  Example: [ a b c ] represents the matrix [ a b ; b c], and the corresponding 1d
  polynomial is a + 2b x + c x^2. Thus the mapping can be represented as
  ( 1., 2., 1. ), [ (0.), (1.), (2.) ]
  '''
  n = (sqrt(1 + 8 * L) - 1)/2  # side of matrix

  if not ceil(n) == n:
    raise TypeError('wrong size for vec(A) for A symmetric')

  j = 0
  i = 0
  i_moniter = grlex_iter( (0,) * dim )
  j_moniter = grlex_iter( (0,) * dim )
  i_mon = next(i_moniter)
  j_mon = next(j_moniter)

  exp = []
  coef = [] 

  for k in range(L):
    # loop over all elements
    exp += [tuple( [ii+jj for ii,jj in zip(i_mon, j_mon) ] )]
    coef += [1. if i == j else 2.]

    if j == n-1:
      i += 1
      j = i
      i_mon = next(i_moniter)
      j_moniter = grlex_iter( i_mon )
      j_mon = next(j_moniter)
  
    else:
      j_mon = next(j_moniter)
      j += 1
  return coef, exp

def vec_to_sparse_matrix(L, dim):
  '''
  Sparse matrix representation of the mapping
    vec(A) |--> c,
  such that x_1^T A x_1 = c^T x_2,
  using grlex ordering both for both x_1 and x_2
  '''

  # Obtain coef - exp representation of mapping
  coef, exp = vec_to_grlex(L, dim)

  # sort coef and row indices simultaneously w.r.t grlex ordering
  sorted_ks = zip(coef, range(L))
  key = cmp_to_key(lambda a1, a2: grlex_comp(exp[a1[1]],exp[a2[1]])) # py3 hack
  sorted_ks = sorted(sorted_ks, key = key)

  # go through to add cols
  mon_iter = grlex_iter( (0,) * dim )

  cols = [  ]
  current_mon = next(mon_iter)
  current_idx = 0
  for vi,ki in sorted_ks:
    while current_mon != exp[ki]:
      current_mon = next(mon_iter)
      current_idx += 1
    cols += [current_idx]

  # unwrap coef, row
  coef, row = zip(*sorted_ks)

  return scipy.sparse.coo_matrix( (coef, (cols, row) ) )
