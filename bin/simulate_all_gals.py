import sys

sys.path.append(
    '/data/groups/jeltema/zhou/lsst_shear/WeakLensingDeblending/bin')
sys.path.append('/data/groups/jeltema/zhou/lsst_shear/WeakLensingDeblending')

import numpy as np
from astropy.io import fits
from astropy.table import Table
import pickle
from tqdm import tqdm

from gals_wideblend import init_wldeblend, get_gal_wldeblend


def get_gal():
    dc2_path = "../data/dc2_table.fits"
    dc2 = Table(fits.open(dc2_path)[1].data)
    dc2_df = dc2.to_pandas()

    gal_list = {}

    data = init_wldeblend(survey_bands='lsst-i',
                          catalog_path="../data/dc2_table.fits")

    i = 0
    for index, gal in tqdm(dc2_df.iterrows()):
        wl_gal = get_gal_wldeblend(rng=np.random.RandomState(1234),
                                   ind=i,
                                   data=data)
        with open("../data/gals/gals_{}.pkl".format(gal.name), "wb") as f:
            pickle.dump(wl_gal, f)
        i += 1


if __name__ == "__main__":
    get_gal()