from math import sqrt, ceil
from collections import OrderedDict
import time

import numpy as np
import scipy.sparse as sp
import mosek

from .utils import mat_to_vec, vec_to_mat, ij_to_k, k_to_ij, veclen_to_matsize, speye, spzeros
from .grlex import count_monomials_leq
from .ptrans import PTrans
from .polynomial import Polynomial

class PPP(object):
  """Positive Polynomial Program"""
  def __init__(self, varinfo_dict=dict()):

    if any(value[2] not in ['pp', 'coef'] for value in varinfo_dict.values()):
      raise Exception('type must be "pp" or "coef"')

    self.varinfo = OrderedDict()
    self.constraints = []
    self.c = np.zeros(self.numvar)

    for name, info in varinfo_dict.items():
      self.add_var(name, *info)

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

  def add_var(self, name, n, d, tp):
    if name in self.varinfo.keys():
      raise Exception('varname {} already present'.format(name))
    self.varinfo[name] = (n, d, tp)

  def add_constraint(self, Aop_dict, b, tp):
    ''' 
    add a constraint to problem of form
    T1 var1 + T2 var2 <= b     if tp=='iq'   (coefficient-wise inequality)
    T1 var1 + T2 var2  = b     if tp=='eq'   (coefficient-wise equality)
    T1 var1 + T2 var2 - b  pp  if tp=='pp'   (positive polynomial)

    Parameters
    ----------
    Aop_dict : dict
        dict with values PTrans. 
        Variables that are not present as keys are assumed 
        to have the zero operator.
    b : Polynomial
        Right-Hand side of constraint
    tp : {'eq', 'iq', 'pp'}
        Type of constraint ('eq'uality, 'in'equality, or 'p'ositive 'p'olynomial).
  
    Example
    ----------
    >>> prob = PPP({'x': (2, 2, 'gram'), 'y': (2, 3, 'gram')})
    >>> T = PTrans.eye(2,2)
    >>> b = Polynomial({(1,2): 1})  # x * y**2
    >>> prob.add_constraint({'x': T}, b, 'eq')
    '''

    if tp not in ['eq', 'iq', 'pp']:
      raise Exception('tp must be "eq" or "iq" or "pp"')

    for name in Aop_dict.keys():
      if name not in self.varinfo.keys():
        raise Exception('unknown variable {}'.format(name))

    if not all(Aop_dict[name].n0 == self.varinfo[name][0] for name in Aop_dict.keys()):
      raise Exception('PTrans initial dimensions do not agree')

    if not all(Aop_dict[name].d0 >= self.varinfo[name][1] for name in Aop_dict.keys()):
      raise Exception('PTrans initial degrees too low')

    n1_list = [Aop_dict[name].n1 for name in Aop_dict.keys()]
    d1_list = [Aop_dict[name].d1 for name in Aop_dict.keys()]
    if max(n1_list) != min(n1_list) or max(d1_list) != min(d1_list):
      raise Exception('final degrees and dimensions must match')

    if n1_list[0] != b.n or d1_list[0] < b.d:
      raise Exception('must have b.n = Aop.n1 and b.d <= Aop.d1 for all Aop')

    if tp == 'iq' and b.d > 0:
      print('Warning: adding coefficient-wise inequality constraint. make sure this is what you want')

    n = n1_list[0]
    d = d1_list[0]
    self.constraints.append((Aop_dict, b, n, d, tp))

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
    >>> prob.add_constraint({'a': [0,1 ], 'b': T} )
    '''

    if not type(c_dict) is dict:
      raise Exception('c_dict must be dict')

    def pick_vec(varname):
      if varname not in c_dict.keys():
        # zeros
        return np.zeros(self.varsize(varname))
      if type(c_dict[varname]) is PTrans:
        if not c_dict[varname].d1 == 0:
          raise Exception('cost for {} not scalar'.format(varname))
        if self.varinfo[varname][2] == 'pp':
          return c_dict[varname].Acg().todense().getA1()
        else:
          return c_dict[varname].Acc().todense().getA1()
      else:
        #  array_like
        if len(c_dict[varname]) != self.varsize(varname):
          raise Exception('cost for {} is wrong size'.format(varname))
        return c_dict[varname]

    self.c = np.hstack([pick_vec(varname) for varname in self.varnames])
    if not len(self.c) == self.numvar:
      raise Exception('wrong size of objective')

  def get_bmat(self, Aop_dict, numcon):
    ''' 
    concatenate matrices in dict in variable order, fill in with zero matrix
    for those variables that do not appear
    '''
    def pick_mat(varname):
      if varname not in Aop_dict.keys():
        return spzeros(numcon, self.varsize(varname))
      if self.varinfo[varname][2] == 'pp':
        return Aop_dict[varname].Acg()
      if self.varinfo[varname][2] == 'coef':
        return Aop_dict[varname].Acc()

    return sp.bmat([[pick_mat(vn) for vn in self.varnames]])


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

    Aeq = sp.coo_matrix((0, self.numvar))
    Aiq = sp.coo_matrix((0, self.numvar))
    beq = np.zeros(0)
    biq = np.zeros(0)

    for (Aop_dict, b, n, d, tp) in self.constraints:
      Amat = self.get_bmat(Aop_dict, count_monomials_leq(n, d))
      if tp == 'eq':
        Aeq = sp.bmat([[Aeq], [Amat]])
        beq = np.hstack([beq, b.mon_coefs(d)])
      if tp == 'iq':
        Aiq = sp.bmat([[Aiq], [Amat]])
        biq = np.hstack([biq, b.mon_coefs(d)])

    numcon_iq = Aiq.shape[0]
    numcon_eq = Aeq.shape[0]

    # Set up optimization problem
    env = mosek.Env() 
    task = env.Task(0,0)

    # Add free variables and objective
    task.appendvars(self.numvar)
    task.putvarboundslice(0, self.numvar, [mosek.boundkey.fr] * self.numvar, [0.]*self.numvar, [0.]*self.numvar )

    # Add objective
    task.putcslice(0, self.numvar, self.c)
    task.putobjsense(mosek.objsense.minimize)

    # add eq & iq constraints
    A = sp.bmat([[Aeq], [Aiq]])
    task.appendcons(numcon_eq + numcon_iq)
    task.putaijlist(A.row, A.col, A.data)
    task.putconboundslice(0, numcon_eq + numcon_iq, 
                          [mosek.boundkey.fx] * numcon_eq + [mosek.boundkey.up] * numcon_iq,
                          list(beq) +  [0.] * numcon_iq, list(beq) + list(biq) )

    # add variable pp constraints
    for varname in self.varnames:
      if self.varinfo[varname][2] == 'pp':
        Asp = speye(self.varsize(varname), self.varpos(varname), self.numvar)
        if pp_cone == 'psd':
          add_psd_mosek(task, Asp, np.zeros(self.varsize(varname)) )
        if pp_cone == 'sdd':
          add_sdd_mosek(task, Asp, np.zeros(self.varsize(varname)) )

    # add pp constraints
    for (Aop_dict, b_pol, n, d, tp) in self.constraints:
      if tp == 'pp':
          if pp_cone == 'psd':
            add_psd_mosek(task, self.get_bmat(Aop_dict, count_monomials_leq(n, d)), b_pol.mon_coefs(d), PTrans.eye(n, d).Acg())
          if pp_cone == 'sdd':
            add_sdd_mosek(task, self.get_bmat(Aop_dict, count_monomials_leq(n, d)), b_pol.mon_coefs(d), PTrans.eye(n, d).Acg())

    print('optimizing...')
    t_start = time.clock()
    task.optimize()
    print("solved in {:.2f}s".format(time.clock() - t_start))

    solsta = task.getsolsta(mosek.soltype.itr)
    print(solsta)
    if (solsta == mosek.solsta.optimal):
        sol = [0.] * self.numvar
        task.getxxslice(mosek.soltype.itr, 0, self.numvar, sol)
        self.sol = sol
        self.check_sol(Aiq, biq, Aeq, beq)
        return sol, solsta
    else:
        return None, solsta

  def check_sol(self, Aiq, biq, Aeq, beq, tol=1e-6):
    '''
    check solution against tolerance to see if constraints are met,
    prints warnings messages if violations above tol are found
    REMARK: currently does not check manually added pp constraints
    '''
    if Aiq.shape[0] > 0 and min(biq - Aiq.dot(self.sol)) < -tol:
        print('warning, iq constraint violated by {:f}'.format(abs(min(biq - Aiq.dot(self.sol)))))
        
    if Aeq.shape[0] > 0 and max(np.abs(Aeq.dot(self.sol)-beq)) > tol:
        print('warning, eq constraint violated by {:f}'.format(max(np.abs(Aeq.dot(self.sol)-beq))) )

    for varname in self.varnames:
      if self.varinfo[varname][2] == 'pp':
        a = self.varpos(varname)
        b = a + self.varsize(varname)
        mat = vec_to_mat(self.sol[a:b])
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
      mon_coefs = PTrans.eye(n, d).Acg().dot(self.sol[a:b])

    return Polynomial.from_mon_coefs(n, mon_coefs)


def add_sdd_mosek(task, A, b, Ac=None):
  ''' 
    add a variable C and constraints
      A*x - Ac vec(C) = b
      C SDD
    which with an appropriate choice of Ac makes A*x - b >= 0

    if Ac is not given it is set to identity
  '''

  if Ac is None:
    n_Ac = A.shape[0]
    Ac = sp.coo_matrix( (np.ones(n_Ac), (range(n_Ac), range(n_Ac))), (n_Ac, n_Ac) )

  if len(b) != A.shape[0]:
    raise Exception('invalid size of A')
  
  if len(b) != Ac.shape[0]:
    raise Exception('invalid size of Ac')

  # size of C
  veclen = Ac.shape[1]
  C_size = veclen_to_matsize(veclen)

  # number of constraints
  numcon = task.getnumcon()
  numvar = task.getnumvar()

  # we need 3x this many new variables
  numvar_new = C_size * (C_size-1) // 2
  numcon_new = A.shape[0]

  # add new vars and constraints as
  # 
  #   [ old_constr   0  ]  [    x     ]    [old_rhs ]
  #   [     A     -Ac D ]  [ new_vars ]  = [  b     ]
  #
  # where D such that vec(C) = D * new_vars

  # build 'D' matrix
  D_row_idx = []
  D_col_idx = []
  D_vals = []
  for row in range(veclen):
    i,j = k_to_ij(row, veclen)
    sdd_idx = sdd_index(i,j,C_size)
    D_row_idx += [row] * len(sdd_idx) 
    D_col_idx += [3*k + l for (k,l) in sdd_idx ]
    D_vals += [ 2. if l == 0 else 1. for (k,l) in sdd_idx ]

  D = sp.coo_matrix( (D_vals, (D_row_idx, D_col_idx)), (veclen, 3 * numvar_new) )

  # add new variables and make them unbounded
  task.appendvars(3 * numvar_new)
  task.putvarboundslice( numvar, numvar + 3 * numvar_new, 
        [mosek.boundkey.fr] * 3 * numvar_new, 
        [0.] * 3 * numvar_new, 
        [0.] * 3 * numvar_new  )
  
  # add new constraints
  task.appendcons(numcon_new)

  # put [A  -Ac*D] [x; new_vars] = b matrix
  new_A = sp.bmat([[A, spzeros(numcon_new, numvar - A.shape[1]), -Ac.dot(D)]]).tocsr()
  task.putarowslice(numcon, numcon + numcon_new, new_A.indptr[:-1], new_A.indptr[1:], new_A.indices, new_A.data)
  task.putconboundslice( numcon, numcon + numcon_new, [mosek.boundkey.fx] * numcon_new, b, b )

  # add cone constraints on new_vars
  task.appendconesseq( [mosek.conetype.rquad] * numvar_new, [0.0] * numvar_new, [3] * numvar_new, numvar )

def add_psd_mosek(task, A, b, Ac = None):
  ''' 
  add a variable C and constraints
    A*x - Ac*vec(C) = b
    C PSD
  which with an appropriate choice of Ac makes A*x - b >= 0

  if Ac is not given it is set to identity
  '''
  if Ac is None:
    n_Ac = A.shape[0]
    Ac = sp.coo_matrix( (np.ones(n_Ac), (range(n_Ac), range(n_Ac))), (n_Ac, n_Ac) )

  if len(b) != A.shape[0] or task.getnumvar() != A.shape[1]:
    raise Exception('invalid size of A')
  
  if len(b) != Ac.shape[0]:
    raise Exception('invalid size of Ac')

  # add PSD matrix of appropriate size
  veclen = Ac.shape[1]
  C_size = veclen_to_matsize(veclen)

  # number of existing variables / constraints
  numbarvar = task.getnumbarvar()
  numcon = task.getnumcon()

  task.appendbarvars([C_size])

  # add numcon_new equality constraints
  numcon_new = len(b)
  task.appendcons(numcon_new)

  Ac_csr = Ac.tocsr()
  for it in range(numcon_new):
    it_row = Ac_csr.getrow(it)
    if it_row.nnz > 0:
      i_list, j_list = zip(*[k_to_ij(k, veclen) for k in it_row.indices])
      val = [it_row.data[l] if i_list[l]==j_list[l] else it_row.data[l]/2 for l in range(len(i_list))]
      mat_it = task.appendsparsesymmat(C_size, j_list, i_list, -np.array(val))
      task.putbaraij(numcon + it, numbarvar, [mat_it], [1.])

  A_csr = A.tocsr()
  task.putarowslice(numcon, numcon + numcon_new, A_csr.indptr[:-1], A_csr.indptr[1:], A_csr.indices, A_csr.data)
  task.putconboundslice(numcon, numcon + numcon_new, [mosek.boundkey.fx] * numcon_new, b, b)

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

  add_sdd_mosek(task, sp.coo_matrix( (np.ones(numvar), (range(numvar), range(numvar)))  ), np.zeros(numvar))

  task.optimize()

  if task.getsolsta(mosek.soltype.itr) != mosek.solsta.prim_infeas_cer:
    return True

  return False

def sdd_index(i,j,n):
  """ An n x n sdd matrix A can be written as A = sum Mij.
    Given Mij's stored as a (n-1)*n/2 x 3 matrix, where each row represents a 2x2 symmetric matrix, return
      the indices i_s, j_s such that A_ij = sum_s Mij(i_s, j_s) """
  num_vars = int(n*(n-1)/2)
  if i == j:
    return [ [ij_to_k(min(i,l), max(i,l)-1, num_vars),(0 if i<l else 1)] for l in range(n) if l != i ]
  else:
    return [[ij_to_k(min(i,j), max(i,j)-1, num_vars),2]]
