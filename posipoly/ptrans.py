from math import ceil, sqrt
from itertools import chain
import copy

import scipy.sparse as sp

from .polynomial import Polynomial
from .grlex import grlex_iter, grlex_to_index, vec_to_grlex, count_monomials_leq
from .utils import double_factorial

class PTrans(object):
  '''
  Linear transformation from C(n0,d0) to C(n1,d1), where
  C(n,d) is the space of coefficients for n-dimensional polynomials
  of degree d 
  '''

  def __init__(self, n0, n1):
    '''
    initialize a transformation from C(n0, 0) to c(n1, 0), where
    C(n, d) is the space of coefficients for n-dimensional polynomials of degree d 
    '''
    if n0 < 1:
      raise Exception('n0 must be >= 1')
    if n1 < 1:
      raise Exception('n0 must be >= 1')

    self._n0 = n0            # initial  number of variables
    self._n1 = n1            # final number of variables
    self._d0 = 0             # initial degree
    self._d1 = 0             # final degree
    self._cols = {}

  ## ACCESS METHODS

  def cols(self):
    '''view of columns'''
    return self._cols.items()

  def __getitem__(self, midx):
    '''retrieve coefficients via [grlex0][grlex1] operation'''
    if len(midx) != self._n0:
      raise TypeError('Multiindex does not match polynomial dimension')
    try:
      return self._cols[midx]
    except KeyError:
      self._cols[midx] = PTransCol(self._n1)
      return self._cols[midx]

  @property
  def n0(self):
    '''initial number of variables'''
    return self._n0

  @property
  def n1(self):
    '''final number of variables'''
    return self._n1

  @property
  def d0(self):
    '''initial degree'''
    return self._d0

  @property
  def d1(self):
    '''final degree'''
    return self._d1

  @property
  def numcon(self):
    '''maximal number of terms'''
    return count_monomials_leq(self.n1, self.d1)  

  def Acc(self):
    ''' 
    obtain coefficient-format to coefficient-format matrix representation (grlex ordering)
    
    Example
    -------
    >> A = T.Acc()
    If p(x) = c'*Z(x), then T.p(x) = (A*c)'*Z(x)

    '''
    i = []
    j = []
    v = []
    nrow = count_monomials_leq(self.n1, self.d1)
    ncol = count_monomials_leq(self.n0, self.d0)
    for midx1, col in self.cols():
      idx1 = grlex_to_index(midx1)
      for midx2, val in col.items():
        i.append(grlex_to_index(midx2))
        j.append(idx1)
        v.append(val)
    return sp.coo_matrix( (v, (i, j)), shape = (nrow, ncol) )

  def Acg(self):
    '''
    obtain gram-format to coefficient-format matrix representation (grlex ordering)
    
    Example
    -------
    >> A = T.Acg()
    If p(x) = Z(x)'*S*Z(x), then T.p(x) = (A*mat_to_vec(S))'*Z(x)
    '''
    half_deg = int(ceil(float(self._d0)/2))
    num_mon = count_monomials_leq(self._n0, half_deg)
    len_vec = num_mon*(num_mon+1)//2  # py3 integer division
    coefs, exps = vec_to_grlex(len_vec, self._n0)
    
    i = []
    j = []
    v = []

    for k, (coef,midx_mid) in enumerate(zip(coefs, exps)):
      for midx_f, val in self[midx_mid].items():
        i.append( grlex_to_index(midx_f) )
        j.append( k )
        v.append( val * coef )

    nrow = count_monomials_leq(self._n1, self._d1)
    ncol = len_vec
    return sp.coo_matrix( (v, (i, j)), shape = (nrow, ncol) )

  ## STATIC METHODS ##

  @staticmethod
  def eye(n0, d0, n1=None, d1=None):
    '''
    identity transformation 
    p(x1, ..., xn0) |--> p(x1, ..., xn0)

    Parameters
    ----------
    n0 : int
      initial number of variables
    d0 : int
      initial degree
    n1: int (optional)
      final number of variables (must be >= n0)
    d1: int (optional)
      final degree

    Returns
    ----------
    T : PTrans
      transformation from C(n0, d0) to C(n1, d1) 
      such that T.p(x1, ..., xn0) is a polynomial in n1 variables
      (but does not depend on last n1-n0 variables)

    '''

    if n1 is None:
      n1 = n0
    if d1 is None:
      d1 = d0

    if n1 < n0:
      raise Exception('eye requires n1 >= n0')
    T = PTrans(n0, n1)
    T._d0 = d0
    for idx in grlex_iter((0,) *  n0, d0):
      idx_mod = idx + (0,) * (n1-n0)
      T[idx][idx_mod] = 1.
    
    T._d1 = d1
    return T

  @staticmethod
  def diff(n0, d0, i):
    '''
    differentiation transformation
    p(x1, ..., xn0) |--> (d/dxi) p(x1, ..., xn0)
    result has degree d0-1

    Parameters
    ----------
    n0 : int
      initial number of variables
    d0 : int
      initial degree
    i: int
      variable to differentiate

    Returns
    ----------
    T : PTrans
      transformation from C(n0, d0) to C(n0, d0-1)

    '''
    T = PTrans(n0,n0)
    T._d0 = d0
    for idx in grlex_iter((0,)*n0, d0):
      k = idx[i]
      new_idx = tuple([(idx[j] if j != i else idx[j]-1) for j in range(n0)])
      if min(new_idx) >= 0:
        T[idx][new_idx] = float(k)
    T._d1 = d0-1
    return T

  @staticmethod
  def int(n0, d0, i):
    '''
    integration transformation
    p(x1, ..., xn0) |--> \int p(x1, ..., xn0) dxi 

    Parameters
    ----------
    n0 : int
      initial number of variables
    d0 : int
      initial degree
    i: int
      dimension to integrate

    Returns
    ----------
    T : PTrans
      transformation from C(n0, d0) to C(n0, d0+1)
    '''
    T = PTrans(n0,n0)
    T._d0 = d0
    T._d1 = d0+1
    for idx in grlex_iter((0,)*n0, d0):
      k = idx[i]
      new_idx = tuple([(idx[j] if j != i else idx[j]+1) for j in range(n0)])
      T[idx][new_idx] = 1./(float(k)+1)
    return T

  @staticmethod
  def eval0(n0, d0, keep_dims=False):
    '''
    extract constant term
    p(x1, ..., xn) |--> p(0, ..., 0)

    Parameters
    ----------
    n0 : int
      initial number of variables
    d0 : int
      initial degree
    keep_dims : bool (optional)
      target polynomial is in n0 variables 

    Returns
    ----------
    T : PTrans 
      transformation from C(n0, d0) to C(n1, 0)
      n1 = n0 if keep_dims else 1

    '''
    n1 = n0 if keep_dims else 1
    T = PTrans(n0, n1)
    T._d0 = d0
    T._d1 = 0
    T[(0,)*n0][(0,) * n1] = 1
    return T

  @staticmethod
  def eval(n0, d0, i_list, val_list, keep_dims=False):
    '''
    evaluation transformation
    p(x1, ..., xn) |--> p(x1, ..., xn) /. xi -> val_list[i] for i in i_list 

    Parameters
    ----------
    n0 : int
      initial number of variables
    d0 : int
      initial degree
    i_list: list of int
      variables to substitute
    val_list: list of numbers
      substitution values
    keep_dims : bool (optional)
      if True, target polynomial is in n0 variables, otherwise n0-len(i_list)

    Returns
    ----------
    T : PTrans
    '''

    if keep_dims:
      raise NotImplementedError

    if isinstance(i_list, list):
      i_list, val_list = [list(x) for x in zip(*sorted(zip(i_list, val_list), key=lambda pair: pair[0]))]
      if len(i_list) > 1:
        return PTrans.eval(n0-1,d0,i_list[:-1], val_list[:-1]) * PTrans.eval(n0,d0,i_list[-1], val_list[-1])
      else:
        return PTrans.eval(n0,d0,i_list[0], val_list[0])
    T = PTrans(n0, max(1, n0-1))

    if n0 == 1:
      for idx in grlex_iter((0,)*n0, d0):
        T[idx][(0,)] += val_list**idx[0]
    else:
      for idx in grlex_iter((0,)*n0, d0):
        new_idx = tuple(idx[i] for i in range(len(idx)) if i != i_list)
        T[idx][new_idx] += val_list**idx[i_list]
    T.updated()
    return T

  @staticmethod
  def mul_pol(n0, d0, g):
    '''
    multiplication transformation
    p(x1, ..., xn) |--> p(x1, ..., xn) * g(x1, ..., xn)

    Parameters
    ----------
    n0 : int
      initial number of variables
    d0 : int
      initial degree
    g: Polynomial
      polynomial in n0 variables to multiply with

    Returns
    ----------
    T : PTrans
      transformation from C(n0, d0) to C(n0, d1+g.d)

    '''
    if not n0 == g.n:
      raise Exception('g must a polynomial in n0 variables')

    T = PTrans(n0,n0)
    maxdeg = 0
    for midx, cf in g.terms():
      for idx in grlex_iter((0,)*n0, d0):
        new_idx = tuple([idx[i] + midx[i] for i in range(n0)])
        T[idx][new_idx] = float(cf)
      maxdeg = max([maxdeg, sum(midx)])
    T._d0 = d0
    T._d1 = d0 + maxdeg
    return T

  @staticmethod
  def integrate(n0, d0, i_list, ival_list, keep_dims=False):
    '''
    integration over box transformation
    p(x1, ..., xn) |--> \int_{xi \in ival_list[i] for i in i_list} p(x1, ..., xn) dx(i for i in i_list)

    Parameters
    ----------
    n0 : int
      initial number of variables
    d0 : int
      initial degree
    i_list: list of int
      variables to integrate over
    ival_list list of list of numbers
      bounds for integration
    keep_dims : bool (optional)
      if True, target polynomial is in n0 variables, otherwise n0-len(i_list)

    Returns
    ----------
    T : PTrans

    Example
    -------
    The mapping p(x,y,z) |--> q(x,z) = \int_0^1 p(x,y,z) dy is obtained by 
      >> T_int = PTrans.integrate(3,2,[1],[[0,1]])
    '''

    if keep_dims:
      raise NotImplementedError

    dim_box = sorted(zip(i_list, ival_list), key = lambda obj : -obj[0])

    T = PTrans.eye(n0,d0)  # start with right-hand identity

    for (dim, box) in dim_box:
      int_trans = PTrans.int(T.n1,d0,dim)  # p.n1 -> p.n1
      upper_trans = PTrans.eval(T.n1,d0+1,dim,box[1]) # p.n1 -> p.n1-1
      lower_trans = PTrans.eval(T.n1,d0+1,dim,box[0]) # p.n1 -> p.n1-1
      T = (upper_trans - lower_trans) * int_trans * T
    T.updated()
    return T

  @staticmethod
  def composition(n0, d0, g_list, keep_dims=False):
    '''
    composition transformation
    p(x1, ..., xn0) |--> p(g_list[0], ..., g_list[n0-1])

    Parameters
    ----------
    n0 : int
      initial number of variables
    d0 : int
      initial degree
    g_list: list of Polynomial
      polynomials representing composition operation
    keep_dims : bool (optional)
      if True, target polynomial is in n0 variables, otherwise n0-len(g_list)

    Returns
    ----------
    T : PTrans
    '''
    
    if keep_dims:
      raise NotImplementedError

    if not (n0 == len(g_list)):
      raise Exception('composition can only be done from n0=1')

    n1 = g_list[0].n
    gd = max(g.d for g in g_list)

    if not all(g.n == n1 for g in g_list):
      raise Exception('all gs must have same number of variables')

    T = PTrans(n0, n1)
    T._d0 = d0
    T._d1 = d0 * gd

    T[(0,) * n0][(0,) * n1] = 1.   # constant term stays the same

    for exp0 in grlex_iter( [0] * (n0-1) + [1], d0 ):
      
      # product of all raised polynomials
      exp_pol = 1.
      for k in range(n0):
        if exp0[k] >= 1:
          exp_pol = (g_list[k]**exp0[k]) * exp_pol

      for exp1, coef in exp_pol.terms():
        if abs(coef) > 1e-10:
          T[exp0][exp1] += coef
    return T

  @staticmethod
  def gaussian_expectation(n0, d0, i, sigma, keep_dims=False):
    '''
    Gaussian expectation transformation
    p(x1, ..., xn) = E_i[p(x1, ..., xn)]  for xi ~ N(0, sigma^2)

    Parameters
    ----------
    n0 : int
      initial number of variables
    d0 : int
      initial degree
    i: int
      index of Gaussian variable
    sigma: number
      standard deviation of Gaussian variable
    keep_dims : bool (optional)
      if True, target polynomial is in n0 variables, otherwise n0-1

    Returns
    ----------
    T : PTrans
    '''

    if keep_dims:
      raise NotImplementedError

    T = PTrans(n0,n0-1)
    T._d0 = d0
    for idx in grlex_iter((0,)*n0, d0):
      new_idx = tuple(idx[j] for j in range(len(idx)) if j != i)
      if idx[i] > 0 and (idx[i] % 2) == 0:
        T[idx][new_idx] = sigma**idx[i] * double_factorial(idx[i]-1)
      if idx[i] == 0:
        T[idx][new_idx] = 1
    T.updated()
    return T

  ### OVERLOADED OPERATORS ###

  def __str__(self):
    '''string representation'''
    ret = 'Transformation from n=%d, d=%d to n=%d, d=%d : \n' % (self._n0, self._d0, self._n1, self._d1)
    for key1, col in self.cols():
      for key2, val in col.items():
        ret += str(key1) + ' --> ' + str(key2) + ' : ' + str(val) + '\n'
    return ret

  def __add__(self, other):
    '''binary addition '+' '''
    if not self._n0 == other.n0 and self._n1 == other.n1:
      raise TypeError('Dimension mismatch')
    ret = PTrans(self._n0, self._n1)
    for midx1, col in chain(self.cols(), other.cols()):
      for midx2, val in col.items():
        try:
          ret[midx1][midx2] += val
        except KeyError:
          # didn't exist yet
          ret[midx1][midx2] = val
    ret._d0 = max(self.d0, other.d0)
    ret._d1 = max(self.d1, other.d1)
    return ret

  def __iadd__(self, other):
    '''unitary addition "+=" '''
    if not self._n0 == other.n0 and self._n1 == other.n1:
      raise TypeError('Dimension mismatch')
    for midx1, col in other.cols():
      for midx2, val in col.items():
        try:
          self[midx1][midx2] += val
        except KeyError:
          # didn't exist yet
          self[midx1][midx2] = val
    self._d0 = max(self.d0, other.d0)
    self._d1 = max(self.d1, other.d1)
    return self

  def __sub__(self, other):
    '''binary subtraction "-" '''
    if not self._n0 == other.n0 and self._n1 == other.n1:
      raise TypeError('Dimension mismatch')
    ret = copy.deepcopy(self)
    for midx1, col in other.cols():
      for midx2, val in col.items():
        try:
          ret[midx1][midx2] -= val
        except KeyError:
          # didn't exist yet
          ret[midx1][midx2] = -val
    ret._d0 = max(self.d0, other.d0)
    ret._d1 = max(self.d1, other.d1)
    return ret

  def __mul__(self, other):
    '''
    binary product "*"
    PTrans * PTrans --> PTrans          (composition)
    PTrans * Polynomial --> Polynomial  (apply transformation)
    PTrans * number --> PTrans          (scaling)
    '''

    if isinstance(other, PTrans):
      if not self._n0 == other.n1:
        raise TypeError('Dimension mismatch')
      ret = PTrans(other.n0, self._n1)
      for midx1, col in other.cols():
        for midxk, val1 in other[midx1].items():
          for midx2, val2 in self[midxk].items():
            ret[midx1][midx2] += val1 * val2

      ret._d0 = other.d0
      ret._d1 = self.d1
      return ret

    if isinstance(other, Polynomial):
      '''transform a Polynomial'''
      if not other.d <= self._d0:
        raise Exception('polynomial has too high degree')

      new_mon_coefs = self.Acc().dot(other.mon_coefs(self._d0))
      return Polynomial.from_mon_coefs(self._n1, new_mon_coefs)

    # assume its a scalar
    ret = copy.deepcopy(self)
    for midx1, col in ret.cols():
      for midx2, val in col.items():
        ret[midx1][midx2] *= other
    return ret

  def __neg__(self):
    '''unitary negation "-"" '''
    ret = copy.deepcopy(self)
    for midx1, col in ret.cols():
      for midx2, val in col.items():
        col[midx2] = -val
    return ret

  ## HOUSEKEEPING ##

  def purge(self):
    ''' Remove zeros in representation '''
    for midx1, col in self.cols():
      remove = [k for k,v in col.items() if v == 0.]
      for k in remove: del col[k]
    remove = [k for k, col in self.cols() if len(col) == 0]
    for k in remove: del self._cols[k]

  def updated(self):
    '''compute final degree'''
    self._d0 = 0
    self._d1 = 0
    for key1, col in self.cols():
      for key2, val in col.items():
        self._d1 = max(self.d1, sum(key2))
      self._d0 = max(self.d0, sum(key1))

class PTransCol(object):
  def __init__(self, n):
    '''
    holds PTrans info for given grlex tuple
    '''
    self.n = n
    self._data = {}

  def __len__(self):
    return len(self._data)

  def __getitem__(self, midx):
    if len(midx) != self.n:
      raise TypeError('Multiindex does not match polynomial dimension')
    try:
      return self._data[midx]
    except KeyError:
      return 0.

  def __setitem__(self, midx, val):
    if len(midx) != self.n:
      raise TypeError('Multiindex does not match polynomial dimension')
    self._data[midx] = float(val)

  def items(self):
    return self._data.items()
