from setuptools import setup

def main():
  setup(
      name='posipoly',
      version='0.1',
      packages=['posipoly'],
      license='BSD-3',
      author='Petter Nilsson',
      author_email='pettni@caltech.edu',
      description='Tools for efficient optimization over positive polynomials',
      package_data={}
  )

if __name__ == '__main__':
  main()