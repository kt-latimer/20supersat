"""
Does NIXE-CAPS actually contain useful data?!?
"""
import numpy as np
from os import listdir

def main():
    """
    check if literally any NIXECAPS datasets contain a single timestamp \
    having non-error entries for every particle size bin below 50um
    """
    path = '/home/klatimer/proj/20supersat/data/halo/npy_proc/'
    files = [f for f in listdir(path)]    
    for f in files:
        if 'NIXECAPS' in f:
            nc_data = np.load(path + f, allow_pickle=True).item()
            n_full_rows = 0
            for i in range(len(nc_data['data']['nconc_1'])):
                full = True
                for j in range(1, 13):
                    key = 'nconc_' + str(j)
                    x = nc_data['data'][key][i]
                    if np.isnan(x):
                        full = False
                        break
                if full:
                    n_full_rows += 1
            print(f, 'full rows', n_full_rows)
#            n_real = 0
#            real_vals = []
#            for i in range(1, 72):
#                key = 'nconc_' + str(i)
#                bin_data = nc_data['data'][key]
#                for j, x in enumerate(bin_data):
#                    if x != 0 and not np.isnan(x):
#                        n_real += 1
#                        real_vals.append(x)
#            n_tot = 71*len(bin_data)
#            print(f, 'total points', n_tot, 'real points', n_real, 'avg', np.mean(real_vals))

if __name__ == "__main__":
    main()
