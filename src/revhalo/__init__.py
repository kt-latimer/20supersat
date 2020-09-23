"""
This is a package. (link to submodules with \:mod\:`<module_name>')
"""
import numpy as np

BASE_DIR = '/home/klatimer/proj/20supersat/'
DATA_DIR = BASE_DIR + 'data/revhalo/'
FIG_DIR = BASE_DIR + 'figures/revhalo/'

CAS_bins = {'upper': np.array([9.6e-07, 3.0e-06, 5.0e-06, \
                    7.2e-06, 1.5e-05, 2.0e-05, 2.5e-05, 3.0e-05, \
                    3.5e-05, 4.0e-05, 4.5e-05, 5.0e-05]), \
            'lower': np.array([8.9e-07, 9.6e-07, 3.0e-06, 5.0e-06, \
                    7.2e-06, 1.5e-05, 2.0e-05, 2.5e-05, 3.0e-05, \
                    3.5e-05, 4.0e-05, 4.5e-05])}

CDP_bins = {'upper': np.array([2.90e-06, 5.00e-06, 7.50e-06, \
                    1.02e-05, 1.18e-05, 1.56e-05, 1.87e-05, 2.07e-05, \
                    2.46e-05, 2.74e-05, 2.92e-05, 3.44e-05, 3.90e-05, \
                    4.25e-05, 4.60e-05]), \
            'lower': np.array([2.50e-06, 2.90e-06, 5.00e-06, \
                    7.50e-06, 1.02e-05, 1.18e-05, 1.56e-05, 1.87e-05, \
                    2.07e-05, 2.46e-05, 2.74e-05, 2.92e-05, 3.44e-05, \
                    3.90e-05, 4.25e-05])}

CIP_bins = {'upper': np.array([7.50e-05, 1.25e-04, 1.75e-04, 2.25e-04, \
                    2.75e-04, 3.25e-04, 4.00e-04, 4.75e-04, 5.50e-04, \
                    6.25e-04, 7.00e-04, 8.00e-04, 9.00e-04, 1.00e-03, \
                    1.20e-03, 1.40e-03, 1.60e-03, 1.80e-03, 2.00e-03]), \
            'lower': np.array([2.50e-05, 7.50e-05, 1.25e-04, 1.75e-04, \
                    2.25e-04, 2.75e-04, 3.25e-04, 4.00e-04, 4.75e-04, \
                    5.50e-04, 6.25e-04, 7.00e-04, 8.00e-04, 9.00e-04, \
                    1.00e-03, 1.20e-03, 1.40e-03, 1.60e-03, 1.80e-03])} 

PCASP_bins = {'upper': np.array([0.11e-6, 0.12e-6, 0.13e-6, 0.14e-6, \
                                 0.15e-6, 0.16e-6, 0.17e-6, 0.18e-6, \
                                 0.2e-6, 0.22e-6, 0.24e-6, 0.26e-6, \
                                 0.28e-6, 0.3e-6, 0.4e-6, 0.5e-6, \
                                 0.6e-6, 0.8e-6, 1.e-6, 1.2e-6, 1.4e-6, \
                                 1.6e-6, 1.8e-6, 2.e-6, 2.2e-6, 2.4e-6, \
                                 2.6e-6, 2.8e-6, 3.e-6]), \
              'lower': np.array([0.1e-6, 0.11e-6, 0.12e-6, 0.13e-6, 0.14e-6, \
                                 0.15e-6, 0.16e-6, 0.17e-6, 0.18e-6, \
                                 0.2e-6, 0.22e-6, 0.24e-6, 0.26e-6, \
                                 0.28e-6, 0.3e-6, 0.4e-6, 0.5e-6, \
                                 0.6e-6, 0.8e-6, 1.e-6, 1.2e-6, 1.4e-6, \
                                 1.6e-6, 1.8e-6, 2.e-6, 2.2e-6, 2.4e-6, \
                                 2.6e-6, 2.8e-6])}
