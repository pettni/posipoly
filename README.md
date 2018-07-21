# posipoly

Collection of methods to efficiently create optimization problems over positive polynomials using Mosek as the underlying solver.

## Installation

```
pip install -r requirements.txt
python setup.py install   # s/install/develop to install `locally`
```

Run tests

```
nosetests
```

Look at an example in `examples/` to see how to do optimization.

## Features

 - Efficient representation of linear polynomial transformations via the PolyLinTrans class.
 - Unified interface for setting up and solving positive polynomial programming (ppp) problems. Currently supports optimization in the PSD (for SOS) and SDD (for SDSOS) cones.



## Documentation

### Retrieve constraints in coefficient form


### Solve a ppp problem

The `solve_ppp` function imposes positivity constraint on gram matrices that are represented in vector form. For a symmetric matrix
```
C = [C11 C12 ... C1n
     C12 C22 ... C2n
      :   :       :
     C1n C2n ... Cnn
```
the vector representation is
```
vec(C) = [C11 C12 ... C1n C22 ... C2n ... Cnn] 
```
of length n(n+1)/2. PP constraints are added by specifying that segments of the variable vector represent such matrices.  Let `mat` be the inverse transformation, i.e. `mat(vec(C)) = C`. 

```
# General format:
#   min  c' x   
#   s.t. Aeq x = beq
#        Aiq x <= beq
#        mat(x[s,s+l]) in pp_cone for s,l in ppp_list
solve_ppp(c, Aeq, beq, Aiq, biq, ppp_list, 'psd')   # optimize in the PSD cone
solve_ppp(c, Aeq, beq, Aiq, biq, ppp_list, 'sdd')   # same problem in the SDD cone
```

## TODO list

 - Implement class for sparsity patterns and enable sparse optimization
 - Interface with other solvers than Mosek (or interface with a package like cvxpy)
 - Abstract syntax to make it more user friendly (add variables, expressions...)

## Research questions

 - Is it possible to restrict the sparsity pattern of multiplier polynomials without loss of generality?
