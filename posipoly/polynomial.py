import numpy as np
import copy

from .grlex import index_to_grlex, grlex_iter
from .utils import multinomial

DTYPE_COEF = np.float32

class Polynomial(object):
  """sparse polynomial"""
  def __init__(self, n, exps_coefs=dict()):
    '''create polynomial in n variables from a dictionary of exponent: coefficient pairs'''
    self._n = n
    self._data = dict()

    for exponent, coefficient in exps_coefs.items():
      self[exponent] = DTYPE_COEF(coefficient)

  def __len__(self):
    '''number of non-zero terms'''
    return len(self._data)

  def __str__(self):
    return 'polynomial of degree {} in {} variables:\n'.format(self.d, self.n) + \
           ' + '.join('{:.2f} * x**{}'.format(c, e) for e, c in self.terms())

  def __call__(self, *x):
    '''evaluate polynomial at given point'''
    if not len(x) == self.n:
      raise Exception('invalid number of inputs')
    return np.sum(coef * np.prod([arg**exp for arg, exp in zip(x, exps)]) 
                  for exps, coef in self.terms())

  def __getitem__(self, exp):
    '''get coefficient for given exponent'''
    if not len(exp) == self.n:
      raise Exception('invalid number of inputs')
    if exp not in self._data.keys():
      return 0.
    else:
      return self._data[exp]

  def __setitem__(self, exp, val):
    '''set coefficient to val for given exponent'''
    if not len(exp) == self.n:
      raise Exception('invalid number of inputs')
    self._data[exp] = val

  def terms(self):
    '''view of exponent/coefficient pairs'''
    return self._data.items()

  def exponents(self):
    '''view of exponents'''
    return self._data.keys()

  def coefficients(self):
    '''view of coefficients'''
    return self._data.values()

  @property 
  def n(self):
    '''number of variables'''
    return self._n

  @property
  def d(self):
    '''degree'''
    if len(self) > 0:
      return max(sum(exp) for exp, _ in self.terms())
    return 0

  @staticmethod
  def zero(n):
    '''n variable zero polynomial p(x1, ..., xn) = 0'''
    return Polynomial(n)

  @staticmethod
  def one(n):
    '''n variable one polynomial p(x1, ..., xn) = 1'''
    return Polynomial(n, {(0,) * n: 1})

  @staticmethod
  def from_sympy(expr, vars):
    '''
    create a polynomial in variables 'vars' from a sympy expression 'expr' (requires sympy)
    
    Example
    -------
    >> from sympy.abc import x,y
    >> p = Polynomial.from_sympy(x**2 * y + 2*x, [x,y])
    '''
    import sympy
    return Polynomial(len(vars), dict( sympy.polys.Poly(expr, *vars).terms()) )

  @staticmethod
  def from_mon_coefs(n, mon_coefs):
    '''create a polynomial in 'n' variables from a list 'mon_coefs' of coefficients (grlex ordering)'''
    max_deg = sum(index_to_grlex(len(mon_coefs), n))
    return Polynomial(n, dict(zip(grlex_iter((0,) * n, max_deg), mon_coefs)))

  def __pow__(self, p):
    '''compute new polynomial g(x)**p'''
    if p <= 0:
      raise Exception('power must be >= 1')
    self.prune()
    ret = Polynomial(self.n)
    # iterator over over all len(self)-tuples that sum to p 
    for k_vec in grlex_iter([0] * (len(self)-1) + [p], p):
      coef = multinomial(k_vec) * np.prod(np.array([p**k for p,k in zip(self.coefficients(), k_vec)]))
      exp = tuple(np.sum([k * np.array(exp) for exp, k in zip(self.exponents(), k_vec)], axis=0))
      ret[exp] += coef
    return ret

  def __mul__(self, other):
    '''multiply polynomial with other'''
    if type(other) is Polynomial:
      if self.n != other.n:
        raise Exception('can not multiply polynomials with different numbers of variables')
      ret = Polynomial(self.n)
      for exp0, coef0 in self.terms():
        for exp1, coef1 in other.terms():
          ret[tuple(np.sum([exp0, exp1], axis=0))] += coef0 * coef1
      return ret
    else:
      # other is scalar
      return Polynomial(self.n, {exp: coef*other for exp, coef in self.terms()})

  def mon_coefs(self, max_deg):
    '''return grlex-ordered vector of all coefficients up to max_deg (including zeros)'''
    return np.array([self[exponent] if exponent in self.exponents() else 0
                     for exponent in grlex_iter( tuple(0 for i in range(self.n)), max_deg)])
  
  def prune(self, tol=1e-10):
    '''remove entries close to zero'''
    self._data = {exp: coef for exp, coef in self._data.items() if abs(coef) > tol}