import numpy as np
import sympy
import copy

from posipoly.grlex import *
from posipoly.utils import multinomial

DTYPE_COEF = np.float32

class Polynomial(object):
  """sparse polynomial"""
  def __init__(self, exps_coefs):
    self.data = dict()
    for exponent, coefficient in exps_coefs.items():
      self.data[exponent] = DTYPE_COEF(coefficient)

  def __str__(self):
    return 'polynomial of degree {} in {} variables:\n'.format(self.d, self.n) + \
           ' + '.join('{:.2f} * x**{}'.format(c, e) for e, c in self.data.items())

  def prune(self, tol=1e-10):
    '''remove entries close to zero'''
    self.data = {exp: coef for exp, coef in self.data.items() if abs(coef) > tol}

  def check(self):
    for exp, coef in self.data:
      if len(exp) != self.n:
        raise Error('invalid polynomial, exponent vectors must be of same length')

      if min(exp) <= 0:
        raise Error('all exponents must be positive') 

  @property
  def d(self):
    return max(sum(exp) for exp in self.data.keys())
  
  @property 
  def n(self):
    if len(self.data) > 0:
      return len(next(iter(self.data.keys())))
    return 0

  @staticmethod
  def from_sympy(poly, vars):
    return Polynomial( dict( sympy.polys.Poly(poly, *vars).terms()) )

  @staticmethod
  def from_mon_coefs(n, mon_coefs):
    ''' create a polynomial in n variables from a list of coefficients (grlex ordering) '''
    max_deg = sum(index_to_grlex(len(mon_coefs), n))

    it = grlex_iter((0,) * n, max_deg)

    coefs = {exp: coef for exp, coef in zip(it, mon_coefs)}

    return Polynomial(coefs)

  def __pow__(self, p):
    '''compute new polynomial g(x)^p'''

    if p <= 0:
      raise Exception('power must be >= 1')

    self.prune()
    N = len(self.data)

    # create iterator over all N-tuples that sum to p 
    start_term = [0] * N
    start_term[N-1] = p
    it = grlex_iter(start_term, p)

    new_data = {}

    for k_vec in it:
      coef = multinomial(k_vec) * np.prod(np.array([p**k for p,k in zip(self.data.values(), k_vec)]))
      exp = tuple(np.sum([k * np.array(exp) for exp, k in zip(self.data.keys(), k_vec)], axis=0))

      if exp in new_data.keys():
        new_data[exp] += coef
      else:
        new_data[exp] = coef

    return Polynomial(new_data)

  def __mul__(self, other):
    '''multiply polynomial with other'''
    if type(other) is Polynomial:
      new_data = {}
      for exp0, coef0 in self.data.items():
        for exp1, coef1 in other.data.items():
          new_exp = tuple(np.sum([exp0, exp1], axis=0))
          if new_exp in new_data:
            new_data[new_exp] += coef0 * coef1
          else:
            new_data[new_exp] = coef0 * coef1

    else:
      new_data = copy.copy(self.data)
      for exp in new_data.keys():
        new_data[exp] *= other
    
    return Polynomial(new_data)

  def mon_coefs(self, max_deg):
    '''return grlex-ordered vector of all coefficients up to max_deg (including zeros)'''
    return np.array([self.data[exponent] if exponent in self.data.keys() else 0
                     for exponent in grlex_iter( tuple(0 for i in range(self.n)), max_deg)])
  
  def evaluate(self, *args):
    '''evaluate polynomial at given point'''
    if not len(args) == self.n:
      raise Error('invalid number of inputs')

    ret = 0.
    for exps, coef in self.data.items():
      ret += coef * np.prod([arg**exp for arg, exp in zip(args, exps)])

    return ret
