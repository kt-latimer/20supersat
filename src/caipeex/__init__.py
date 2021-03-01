"""
This is a package. (link to submodules with \:mod\:`<module_name>')
"""
import numpy as np

BASE_DIR = '/home/klatimer/proj/20supersat/'
DATA_DIR = BASE_DIR + 'data/caipeex/'
FIG_DIR = BASE_DIR + 'figures/caipeex/'

CDP_bins = {'lower': np.array([2.0000e-06, 3.0000e-06, 4.0000e-06, 5.0000e-06, \
                            6.0000e-06, 7.0000e-06, 8.0000e-06, 9.0000e-06, \
                            1.0000e-05, 1.1000e-05, 1.2000e-05, 1.3000e-05, \
                            1.4000e-05, 1.6000e-05, 1.8000e-05, 2.0000e-05, \
                            2.2000e-05, 2.4000e-05, 2.6000e-05, 2.8000e-05, \
                            3.0000e-05, 3.2000e-05, 3.4000e-05, 3.6000e-05, \
                            3.8000e-05, 4.0000e-05, 4.2000e-05, 4.4000e-05, \
                            4.6000e-05, 4.8000e-05, 5.0000e-05, 6.2500e-05, \
                            8.7500e-05, 1.1250e-04, 1.3750e-04, 1.6250e-04, \
                            1.8750e-04, 2.1250e-04, 2.3750e-04, 2.6250e-04, \
                            2.8750e-04, 3.1250e-04, 3.3750e-04, 3.6250e-04, \
                            3.8750e-04, 4.1250e-04, 4.3750e-04, 4.6250e-04, \
                            4.8750e-04, 5.1250e-04, 5.3750e-04, 5.6250e-04, \
                            5.8750e-04, 6.1250e-04, 6.3750e-04, 6.6250e-04, \
                            6.8750e-04, 7.1250e-04, 7.3750e-04, 7.6250e-04, \
                            7.8750e-04, 8.1250e-04, 8.3750e-04, 8.6250e-04, \
                            8.8750e-04, 9.1250e-04, 9.3750e-04, 9.6250e-04, \
                            9.8750e-04, 1.0125e-03, 1.0375e-03, 1.0625e-03, \
                            1.0875e-03, 1.1125e-03, 1.1375e-03, 1.1625e-03, \
                            1.1875e-03, 1.2125e-03, 1.2375e-03, 1.2625e-03, \
                            1.2875e-03, 1.3125e-03, 1.3375e-03, 1.3625e-03, \
                            1.3875e-03, 1.4125e-03, 1.4375e-03, 1.4625e-03, \
                            1.4875e-03, 1.5125e-03, 1.5375e-03]), \
            'upper': np.array([3.0000e-06, 4.0000e-06, 5.0000e-06, 6.0000e-06, \
                            7.0000e-06, 8.0000e-06, 9.0000e-06, 1.0000e-05, \
                            1.1000e-05, 1.2000e-05, 1.3000e-05, 1.4000e-05, \
                            1.6000e-05, 1.8000e-05, 2.0000e-05, 2.2000e-05, \
                            2.4000e-05, 2.6000e-05, 2.8000e-05, 3.0000e-05, \
                            3.2000e-05, 3.4000e-05, 3.6000e-05, 3.8000e-05, \
                            4.0000e-05, 4.2000e-05, 4.4000e-05, 4.6000e-05, \
                            4.8000e-05, 5.0000e-05, 6.2500e-05, 8.7500e-05, \
                            1.1250e-04, 1.3750e-04, 1.6250e-04, 1.8750e-04, \
                            2.1250e-04, 2.3750e-04, 2.6250e-04, 2.8750e-04, \
                            3.1250e-04, 3.3750e-04, 3.6250e-04, 3.8750e-04, \
                            4.1250e-04, 4.3750e-04, 4.6250e-04, 4.8750e-04, \
                            5.1250e-04, 5.3750e-04, 5.6250e-04, 5.8750e-04, \
                            6.1250e-04, 6.3750e-04, 6.6250e-04, 6.8750e-04, \
                            7.1250e-04, 7.3750e-04, 7.6250e-04, 7.8750e-04, \
                            8.1250e-04, 8.3750e-04, 8.6250e-04, 8.8750e-04, \
                            9.1250e-04, 9.3750e-04, 9.6250e-04, 9.8750e-04, \
                            1.0125e-03, 1.0375e-03, 1.0625e-03, 1.0875e-03, \
                            1.1125e-03, 1.1375e-03, 1.1625e-03, 1.1875e-03, \
                            1.2125e-03, 1.2375e-03, 1.2625e-03, 1.2875e-03, \
                            1.3125e-03, 1.3375e-03, 1.3625e-03, 1.3875e-03, \
                            1.4125e-03, 1.4375e-03, 1.4625e-03, 1.4875e-03, \
                            1.5125e-03, 1.5375e-03, 1.5625e-03])}

PCASP_bins = {'upper': np.array([0.1e-6, 0.11e-6, 0.12e-6, 0.13e-6, 0.14e-6, \
                                 0.15e-6, 0.16e-6, 0.17e-6, 0.18e-6, \
                                 0.2e-6, 0.22e-6, 0.24e-6, 0.26e-6, \
                                 0.28e-6, 0.3e-6, 0.4e-6, 0.5e-6, \
                                 0.6e-6, 0.8e-6, 1.e-6, 1.2e-6, 1.4e-6, \
                                 1.6e-6, 1.8e-6, 2.e-6, 2.2e-6, 2.4e-6, \
                                 2.6e-6, 2.8e-6, 3.e-6]), \
              'lower': np.array([0.09e-6, 0.1e-6, 0.11e-6, 0.12e-6, 0.13e-6, \
                                 0.14e-6, 0.15e-6, 0.16e-6, 0.17e-6, 0.18e-6, \
                                 0.2e-6, 0.22e-6, 0.24e-6, 0.26e-6, \
                                 0.28e-6, 0.3e-6, 0.4e-6, 0.5e-6, \
                                 0.6e-6, 0.8e-6, 1.e-6, 1.2e-6, 1.4e-6, \
                                 1.6e-6, 1.8e-6, 2.e-6, 2.2e-6, 2.4e-6, \
                                 2.6e-6, 2.8e-6])}
