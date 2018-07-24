import numpy as np
import scipy.sparse as sp
from math import sqrt, ceil
from collections import OrderedDict

import mosek

from .utils import mat_to_vec, vec_to_mat, ij_to_k, k_to_ij
from .grlex import count_monomials_leq
from .ptrans import PTrans
from .polynomial import Polynomial

def sdd_index(i,j,n):
  """ An n x n sdd matrix A can be written as A = sum Mij.
    Given Mij's stored as a (n-1)*n/2 x 3 matrix, where each row represents a 2x2 symmetric matrix, return
      the indices i_s, j_s such that A_ij = sum_s Mij(i_s, j_s) """
  num_vars = int(n*(n-1)/2)
  if i == j:
    return [ [ij_to_k(min(i,l), max(i,l)-1, num_vars),(0 if i<l else 1)] for l in range(n) if l != i ]
  else:
    return [[ij_to_k(min(i,j), max(i,j)-1, num_vars),2]]


class PPP(object):
  """Positive Polynomial Program"""
  def __init__(self, varinfo=dict()):

    if any(value[2] not in ['pp', 'coef'] for value in varinfo.values()):
      raise Exception('type must be "pp" or "coef"')

    self.varinfo = OrderedDict(varinfo)
    self.Aeq = sp.coo_matrix((0, self.numvar))
    self.beq = np.zeros(0)

    self.Aiq = sp.coo_matrix((0, self.numvar))
    self.biq = np.zeros(0)

    self.c = np.zeros(self.numvar)

    self.sol = None

  @property
  def varnames(self):
    '''get names of all variables'''
    return list(self.varinfo.keys())

  @property
  def varsizes(self):
    '''get coefficient sizes of all polynomial variables'''
    return [self.varsize(name) for name in self.varinfo.keys()]

  @property
  def numvar(self):
    '''total number of coefficient variables'''
    return sum(self.varsizes)

  @property
  def numcon(self):
    '''total number of constraints'''
    return self.Aeq.shape[0] + self.Aiq.shape[0]
  
  def varsize(self, varname):
    '''get size of variable "varname"'''
    if varname not in self.varinfo.keys():
      raise Exception('unknown variable {}'.format(varname))

    n = self.varinfo[varname][0]
    d = self.varinfo[varname][1]

    if self.varinfo[varname][2] == 'pp':
      num_mon = count_monomials_leq(n, int(ceil(float(d)/2)))
      return num_mon*(num_mon+1)//2

    if self.varinfo[varname][2] == 'coef':
      return count_monomials_leq(n, d)

  def varpos(self, varname):
    '''return starting position of variable'''
    if varname not in self.varinfo.keys():
      raise Exception('unknown variable {}'.format(varname))

    ret = 0
    for name in self.varnames:
      if name == varname:
        break
      ret += self.varsize(name)
    return ret

  def add_var(self, name, n, deg, tp):
    
    if name in self.varinfo.keys():
      raise Exception('varname {} already present'.format(name))

    self.varinfo[name] = (n, deg, tp)
    self.Aeq = sp.bmat([[self.Aeq, sp.coo_matrix((self.numcon, self.varsize(name)))]])
    self.Aiq = sp.bmat([[self.Aiq, sp.coo_matrix((self.numcon, self.varsize(name)))]])

  def add_row(self, Aop_dict, b, tp):
    ''' 
    add a constraint to problem of form
    T1 var1 + T2 var2 <= b   tp='iq'
    T1 var1 + T2 var2 = b    tp='eq'


    Parameters
    ----------
    Aop_dict : dict
        dict with values PTrans. 
        Variables that are not present as keys are assumed 
        to have the zero operator.
    b : Polynomial
        Right-Hand side of constraint
    tp : {'eq', 'iq'}
        Type of constraint ('eq'uality or 'in'equality).
  
    Example
    ----------
    >>> prob = PPP({'x': (2, 2, 'gram'), 'y': (3, 3, 'gram')})
    >>> T = PTrans.eye(2,2)
    >>> b = Polynomial({(1,2): 1})  # x * y**2
    >>> prob.add_row({'x': T}, b, 'eq')
    '''

    if tp not in ['eq', 'iq']:
      raise Exception('tp must be "eq" or "iq"')

    if tp == 'iq' and b.d > 0:
      print('Warning: adding coefficient-wise inequality constraint. make sure this is what you want')

    for name in Aop_dict.keys():
      if name not in self.varinfo.keys():
        raise Exception('unknown variable {}'.format(name))

    n1_list = [Aop_dict[name].n1 for name in Aop_dict.keys()]
    d1_list = [Aop_dict[name].d1 for name in Aop_dict.keys()]
    if max(n1_list) != min(n1_list) or max(d1_list) != min(d1_list):
      raise Exception('final degrees and dimensions must match')

    if n1_list[0] != b.n or d1_list[0] < b.d:
      raise Exception('must have b.n = Aop.n1 and b.d <= Aop.d1 for all Aop')

    numcon = count_monomials_leq(n1_list[0], d1_list[0])

    matrices = dict()
    for varname in self.varnames:
      if varname in Aop_dict.keys():
        if Aop_dict[varname].d0 != self.varinfo[varname][1] or Aop_dict[varname].n0 != self.varinfo[varname][0]:
          raise Exception('operator for {} has wrong initial dimension or degree'.format(varname))
        if self.varinfo[varname][2] == 'pp':
          # pp variable
          matrices[varname] = Aop_dict[varname].Acg()
        else:
          # coefficient variable
          matrices[varname] = Aop_dict[varname].Acc()

        # check varsize
        if not matrices[varname].shape[1] == self.varsize(varname):
          raise Exception('transformation for {} is wrong size, check initial degree'.format(varname))
      else:
        # add zero matrix
        matrices[varname] = sp.coo_matrix((numcon, self.varsize(varname)))

    newrow = sp.bmat([[matrices[name] for name in self.varnames]])

    if tp == 'eq':
      self.Aeq = sp.bmat([[self.Aeq], [newrow]])
      self.beq = np.hstack([self.beq, b.mon_coefs(d1_list[0])])
    else:
      self.Aiq = sp.bmat([[self.Aiq], [newrow]])
      self.biq = np.hstack([self.biq, b.mon_coefs(d1_list[0])])

  def set_objective(self, c_dict):
    ''' 
    add objective to problem 

    Parameters
    ----------
    c_dict : dict
        keys: varnames
        values: PTrans with final degree 0 (scalar) or array_like

    Example
    ----------
    >>> T = PTrans.integrate(n,d,dims,boxes)  # results in scalar
    >>> prob.add_row({'a': [0,1 ], 'b': T} )
    '''

    if not type(c_dict) is dict:
      raise Exception('c_dict must be dict')

    vectors = dict()
    for varname in self.varnames:
      if varname in c_dict:
        if type(c_dict[varname]) is PTrans:
          if self.varinfo[varname][2] == 'pp':
            vectors[varname] = c_dict[varname].Acg().todense().getA1()
          else:
            vectors[varname] = c_dict[varname].Acc().todense().getA1()
        else:
          #  array_like
          vectors[varname] = c_dict[varname]
      else:
        # zero cost
        vectors[varname] = np.zeros(self.varsize(varname))
      if not len(vectors[varname]) == self.varsize(varname):
        raise Exception('weight for {} has wrong size'.format(varname))
    self.c = np.hstack([vectors[varname] for varname in self.varnames])

  def solve(self, pp_cone):
    ''' 
    solve Positive Polynomial Program 

    Parameters
    ----------
    pp_cone : {'psd', 'sdd'}
        cone for positivity constraints
  
    Returns
    ----------
    sol : solution vector
    sta : status
    '''

    pp_list = [[self.varpos(varname), self.varsize(varname)] for varname in self.varnames 
               if self.varinfo[varname][2] == 'pp']

    sol, sta = solve_ppp(self.c, self.Aeq, self.beq, self.Aiq, self.biq, pp_list, pp_cone)

    self.sol = sol
 
    if sol is not None:
      self.check_sol(self.sol)

    return sol, sta

  def check_sol(self, sol, tol=1e-6):
    '''
    check solution against tolerance to see if constraints are met,
    prints warnings messages if violations above tol are found
    '''
    if self.Aiq.shape[0] > 0 and min(self.biq - self.Aiq.dot(sol)) < -tol:
        print('warning, iq constraint violated by {:f}'.format(abs(min(self.biq - self.Aiq.dot(sol)))))
        
    if self.Aeq.shape[0] > 0 and max(np.abs(self.Aeq.dot(sol)-self.beq)) > tol:
        print('warning, eq constraint violated by {:f}'.format(max(np.abs(self.Aeq.dot(sol)-self.beq))) )

    for varname in self.varnames:
      if self.varinfo[varname][2] == 'pp':
        a = self.varpos(varname)
        b = a + self.varsize(varname)
        mat = vec_to_mat(sol[a:b])
        v, _ = np.linalg.eig(mat)
        if min(v) < -tol:
            print('warning, pp constraint violated by {:f}'.format(abs(min(v))))

  def get_poly(self, varname):
    '''return a Polynomial object from solution'''
    if self.sol is None:
      raise Exception('no solution stored')
    if varname not in self.varnames:
      raise Exception('unknown variable {}'.format())

    a = self.varpos(varname)
    b = a + self.varsize(varname)
    n = self.varinfo[varname][0]
    d = self.varinfo[varname][1]

    if self.varinfo[varname][2] == 'coef':
      mon_coefs = self.sol[a:b]
    else:
      mon_coefs = PTrans.eye(n, d).Acg().dot(self.sol[a,b])

    return Polynomial.from_mon_coefs(n, mon_coefs)

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
  task.putconboundslice(0, numcon_eq + numcon_iq, 
                        [mosek.boundkey.fx] * numcon_eq + [mosek.boundkey.up] * numcon_iq,
                        list(beq) +  [0.] * numcon_iq, list(beq) + list(biq) )

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
