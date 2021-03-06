"""
This is a package. (link to submodules with \:mod\:`<module_name>')
"""

import numpy as np

BASE_DIR = '/home/klatimer/proj/20supersat/'
DATA_DIR = BASE_DIR + 'data/rev11caipeex/'
FIG_DIR = BASE_DIR + 'figures/rev11caipeex/'

CIP_bins = {'upper': np.array([75, 125, 175, 225, 275, 325, 400, 475, 550, \
                               625, 700, 800, 900, 1000, 1200, 1400, 1600, \
                               1800, 2200, 2600, 3000, 3400, 3800, 4200, \
                               4600, 5000, 6000, 7000, 8000, 9000, 11000, \
                               15000, 20000, 25000, 30000])*1.e-6, \
            'lower': np.array([25, 75, 125, 175, 225, 275, 325, 400, 475, 550, \
                               625, 700, 800, 900, 1000, 1200, 1400, 1600, \
                               1800, 2200, 2600, 3000, 3400, 3800, 4200, \
                               4600, 5000, 6000, 7000, 8000, 9000, 11000, \
                               15000, 20000, 25000])*1.e-6}

FSSP_bins = {'upper': np.array([3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5, 15, \
                               16.5, 18, 19.5, 21, 22.5, 24, 25.5, 27, 28.5, \
                               30, 31.5, 33, 34.5, 36, 37.5, 39, 40.5, 42, \
                               43.5, 45, 47])*1.e-6, \
            'lower': np.array([1, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5, 15, \
                               16.5, 18, 19.5, 21, 22.5, 24, 25.5, 27, 28.5, \
                               30, 31.5, 33, 34.5, 36, 37.5, 39, 40.5, 42, \
                               43.5, 45])*1.e-6}

PCASP_bins = {'upper': np.array([0.11e-6, 0.12e-6, 0.13e-6, 0.14e-6, \
                                 0.15e-6, 0.16e-6, 0.17e-6, 0.18e-6, \
                                 0.2e-6, 0.22e-6, 0.24e-6, 0.26e-6, \
                                 0.28e-6, 0.3e-6, 0.4e-6, 0.5e-6, \
                                 0.6e-6, 0.8e-6, 1.e-6, 1.2e-6, 1.4e-6, \
                                 1.6e-6, 1.8e-6, 2.e-6, 2.2e-6, 2.4e-6, \
                                 2.6e-6, 2.8e-6]), \
              'lower': np.array([0.1e-6, 0.11e-6, 0.12e-6, 0.13e-6, \
                                 0.14e-6, 0.15e-6, 0.16e-6, 0.17e-6, 0.18e-6, \
                                 0.2e-6, 0.22e-6, 0.24e-6, 0.26e-6, \
                                 0.28e-6, 0.3e-6, 0.4e-6, 0.5e-6, \
                                 0.6e-6, 0.8e-6, 1.e-6, 1.2e-6, 1.4e-6, \
                                 1.6e-6, 1.8e-6, 2.e-6, 2.2e-6, 2.4e-6, \
                                 2.6e-6])}
                                 
DMA_bins = {'upper': np.array([0.02110, 0.02227, 0.02349, 0.02479, \
                               0.02615, 0.02759, 0.02911, 0.03072, 0.03241, \
                               0.03420, 0.03609, 0.03808, 0.04017, 0.04238, \
                               0.04472, 0.04719, 0.04979, 0.05253, 0.05543, \
                               0.05848, 0.06170, 0.06510, 0.06869, 0.07248, \
                               0.07647, 0.08069, 0.08513, 0.08983, 0.09478, \
                               0.10000, 0.10551, 0.11132, 0.11746, 0.12394, \
                               0.13077, 0.13797, 0.14557, 0.15360, 0.16207, \
                               0.17100, 0.18043, 0.19037, 0.20086, 0.21193, \
                               0.22361, 0.23593, 0.24893, 0.26265, 0.27713, \
                               0.29240, 0.30851, 0.32552, 0.34346, 0.36239, \
                               0.38236, 0.40343, 0.42567, 0.44913, 0.47388, \
                               0.50001])*1.e-6, \
            'lower': np.array([0.01999, 0.02110, 0.02227, 0.02349, 0.02479, \
                               0.02615, 0.02759, 0.02911, 0.03072, 0.03241, \
                               0.03420, 0.03609, 0.03808, 0.04017, 0.04238, \
                               0.04472, 0.04719, 0.04979, 0.05253, 0.05543, \
                               0.05848, 0.06170, 0.06510, 0.06869, 0.07248, \
                               0.07647, 0.08069, 0.08513, 0.08983, 0.09478, \
                               0.10000, 0.10551, 0.11132, 0.11746, 0.12394, \
                               0.13077, 0.13797, 0.14557, 0.15360, 0.16207, \
                               0.17100, 0.18043, 0.19037, 0.20086, 0.21193, \
                               0.22361, 0.23593, 0.24893, 0.26265, 0.27713, \
                               0.29240, 0.30851, 0.32552, 0.34346, 0.36239, \
                               0.38236, 0.40343, 0.42567, 0.44913, 0.47388, \
                               ])*1.e-6}
