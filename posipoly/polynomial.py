import numpy as np

from posipoly.grlex import *

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
  def from_sympy(poly):
    return Polynomial( dict(poly.as_poly().terms()) )

  @staticmethod
  def from_mon_coefs(n, mon_coefs):
    ''' create a polynomial in n variables from a list of coefficients (grlex ordering) '''

    max_deg = sum(index_to_grlex(len(mon_coefs), n))

    it = grlex_iter((0,) * n, max_deg)

    coefs = {exp: coef for exp, coef in zip(it, mon_coefs)}

    return Polynomial(coefs)

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