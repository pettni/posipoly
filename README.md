# posipoly

Collection of methods to efficiently create optimization problems over positive polynomials using Mosek as the underlying solver.

## Installation

    pip install -r requirements.txt
    python setup.py install   # s/install/develop to install `locally`
  
Run tests

    nosetests

Look at an example in `examples/` to see how to do optimization.

## Features

 - Efficient representation of linear polynomial transformations via the PolyLinTrans class
 - Add SDSOS constraints to a Mosek optimization task

## TODO list

 - Create methods to set up SOS programs
 - Interface with other solvers than Mosek (or interface with a package like cvxpy)
 - Abstract syntax to make it more user friendly
