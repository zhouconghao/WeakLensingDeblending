import sys

sys.path.append(
    '/data/groups/jeltema/zhou/lsst_shear/WeakLensingDeblending/bin')
sys.path.append('/data/groups/jeltema/zhou/lsst_shear/WeakLensingDeblending')

from mpi4py import MPI
import sys, os
import time as t

from tqdm import tqdm, trange

from astropy.io import fits
from astropy.table import Table
import pickle
from sim_func import sim_func
import mdet_meas_tools as mmt
from gals_wideblend import *

if __name__ == "__main__":

    # from mpi4py import MPI
    # mpi_rank = MPI.COMM_WORLD.Get_rank()
    # mpi_size = MPI.COMM_WORLD.Get_size()
    # print(mpi_rank, mpi_size)

    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    print(mpi_rank)

    gal_list_path = "/data/groups/jeltema/zhou/lsst_shear/WeakLensingDeblending/data/gal_list.pkl"
    gal_list = pickle.load(open(gal_list_path, "rb"))

    N_lens_total = len(gal_list)
    N_step = round(N_lens_total / mpi_size)

    if mpi_rank != (mpi_size - 1):
        len_cat = gal_list[mpi_rank * N_step:(mpi_rank + 1) * N_step]
        N_lens = len(len_cat)
    else:
        len_cat = gal_list[mpi_rank * N_step:]
        N_lens = len(len_cat)

    del gal_list

    def calculate_R11_for_gal_list(gal_list_, result_list, error_list, noise):

        for gal, gal_id in tqdm(gal_list_):
            gal = gal[0][0]
            try:
                pdata, mdata, m, msd, c, csd, R11 = mmt.run_mdet_sims(
                    gal=gal,
                    sim_func=sim_func,
                    sim_kwargs={'noise': noise},
                    seed=123,
                    n_sims=10,
                    use_p=True,
                    use_m=True)

                result_list.append((gal_id, R11))
            except Exception as e:
                error_list.append((gal_id, e))
                continue

    error_list = []
    result_list = []

    calculate_R11_for_gal_list(len_cat,
                               result_list=result_list,
                               noise=432,
                               error_list=error_list)

    gal_id, R11 = zip(*result_list)
    result_table = Table([gal_id, R11], names=('gal_id', 'R11'))

    save_dir = "/data/groups/jeltema/zhou/lsst_shear/WeakLensingDeblending/mpi/output"
    out_path = os.path.join(save_dir, f"R11_{mpi_rank}.fits")
    result_table.write(out_path, overwrite=True)
