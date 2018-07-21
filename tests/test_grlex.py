import numpy as np
import sympy as sp
import itertools

from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key
from sympy.abc import x, y, z

from posipoly.grlex import *

def test_index_to_grlex():
  np.testing.assert_equal(index_to_grlex(29, 3), (2,0,2))
  np.testing.assert_equal(index_to_grlex(0, 3), (0,0,0))
  np.testing.assert_equal(index_to_grlex(9, 3), (2,0,0))

def test_index_to_grlex_to_index():
  for i in range(100):
    np.testing.assert_equal(grlex_to_index(index_to_grlex(i, 3)), i)

def test_grlex_iter():
  iterator = grlex_iter( (0,0,0,0) )

  idx1 = next(iterator)

  for k in range(600):
    idx2 = next(iterator)
    np.testing.assert_equal( (grlex_key( idx1) < grlex_key(idx2 )), True )
    idx1 = idx2

def test_grlex_iter2():
  iterator = grlex_iter( (0,0,0,0), 4 )

  for k in range(count_monomials_leq(4,4)):
    next(iterator)
  
  try:
    next(iterator)
  except StopIteration:
    pass
  except:
    self.fail('Unexpected exception thrown:')
  else:
    self.fail('ExpectedException not thrown')

def test_grlex_iter_multi():
  m1 = 5
  m2 = 3
  m3 = 10
  iterator = multi_grlex_iter( (0,0,0), [[0], [1], [2]], [m1,m2,m3] )

  comb_iter = itertools.product( range(m1+1), range(m2+1), range(m3+1) )

  for k in range((m1+1) * (m2+1) * (m3+1)):
    asd =  next(iterator)
    if k == m1+1:
      np.testing.assert_equal(asd, (0,1,0))

  np.testing.assert_equal(asd , (m1,m2,m3))

  try:
    next(iterator)
  except StopIteration:
    pass
  except:
    self.fail('Unexpected exception thrown:')
  else:
    self.fail('ExpectedException not thrown')

def test_vec_to_grlex1():
  coef, exp = vec_to_grlex(10,3)
  np.testing.assert_equal(exp, [(0,0,0), (0,0,1), (0,1,0), (1,0,0), (0,0,2), (0,1,1), (1,0,1), (0,2,0), (1,1,0), (2,0,0) ])
  np.testing.assert_equal(coef, [1,2,2,2,1,2,2,1,2,1])

def test_vec_to_grlex2():
  coef, exp = vec_to_grlex(10,2)
  np.testing.assert_equal(exp, [(0,0), (0,1), (1,0), (0,2), (0,2), (1,1), (0,3), (2,0), (1,2), (0,4) ])
  np.testing.assert_equal(coef, [1,2,2,2,1,2,2,1,2,1])

def test_vec_to_grlex3():
  coef, exp = vec_to_grlex(28,1)
  np.testing.assert_equal(exp, [tuple([i+j]) for j in range(7) for i in range(j,7)])
  np.testing.assert_equal(coef, [1 if i==j else 2 for j in range(7) for i in range(j,7)])


def test_vec_to_sparse_matrix1():
  poly_vec = [1,2,3,4,5,6,7,8,9,10]

  # represents 4 x 4 matrix :
  poly_mat = np.array([
        [1, 2, 3, 4],
              [2, 5, 6, 7],
              [3, 6, 8, 9],
              [4, 7, 9, 10]])

  trans = vec_to_sparse_matrix(10,2)

  mon_vec = trans.dot(np.array(poly_vec))

  mon1 = sorted(itermonomials([x, y], 4), key=monomial_key('grlex', [x, y]))[0:4]
  mon2 = sorted(itermonomials([x, y], 4), key=monomial_key('grlex', [x, y]))[0:len(mon_vec)]

  np.testing.assert_equal( sp.simplify(np.dot(mon_vec, mon2) - np.dot(np.dot( poly_mat, mon1 ), mon1 )), sp.numbers.Zero )

def test_vec_to_sparse_matrix2():
  poly_vec = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

  # represents 4 x 4 matrix :
  poly_mat = np.array([
        [1, 2, 3, 4, 5],
              [2, 6, 7, 8, 9],
              [3, 7, 10, 11, 12],
              [4, 8, 11, 13, 14],
              [5, 9, 12, 14, 15]])

  trans = vec_to_sparse_matrix(15,3)

  mon_vec = trans.dot(np.array(poly_vec))

  mon1 = sorted(itermonomials([x, y, z], 4), key=monomial_key('grlex', [x, y, z]))[0:5]
  mon2 = sorted(itermonomials([x, y, z], 4), key=monomial_key('grlex', [x, y, z]))[0:len(mon_vec)]

  np.testing.assert_equal( sp.simplify(np.dot(mon_vec, mon2) - np.dot(np.dot( poly_mat, mon1 ), mon1 )), sp.numbers.Zero )