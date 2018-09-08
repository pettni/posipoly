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
      classifiers=['Intended Audience :: Science/Research',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Operating System :: Microsoft :: Windows',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3'],
      package_data={}
  )

if __name__ == '__main__':
  main()