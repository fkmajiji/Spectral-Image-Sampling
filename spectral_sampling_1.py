# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch

import utils_image as util

import random
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth

import NTIRE2022Util as spec
import h5py
from scipy.interpolate import interp1d
import os
from scipy.io import loadmat, savemat
from scipy import ndimage
"""
# --------------------------------------------
# Spectral Sampling
# --------------------------------------------
#
# Kai Feng (fengkainpu@gmail.com)
# --------------------------------------------
"""

def create_multispectral(hsi, hsi_bands):
    filterQEs, QEs_bands_list = spec.load_mine_filter('csf_IMEC25_600-875.csv', linesnum=25)
    CutofffilterQEs, CutoffQEs_bands_list = spec.load_mine_filter('cutoff_filter_response.csv', linesnum=2)
    illum, illum_bands_list = spec.load_mine_filter('illum-solarTT59.csv', linesnum=1)
    illum = spec.interpolate(illum, illum_bands_list, QEs_bands_list)
    illum = illum / np.max(illum)
    CutoffQE = CutofffilterQEs[:, 0] * CutofffilterQEs[:, 1] * illum[:, 0]
    qes = np.expand_dims(CutoffQE, axis=1) * filterQEs
    if np.all(QEs_bands_list == CutoffQEs_bands_list):
        qe_bands = QEs_bands_list
    else:
        print("wrong")
    msi = spec.projectHS(hsi, hsi_bands, qes, qe_bands, clipNegative=True)
    msi /= msi.max()
    return msi

if __name__ == '__main__':

    img = h5py.File('Cloth3.mat')
    img = np.transpose(img['img'])
    img = img[:, :, 17:48]
    hsi_bands_list = np.arange(590, 890.1, 10)
    interpfun = interp1d(hsi_bands_list, img, axis=2, kind='cubic')
    new_hsi_bands_list = np.arange(590, 890.1, 1)
    img = interpfun(new_hsi_bands_list)

    msicube = create_multispectral(img, new_hsi_bands_list)

    msicube_uint = util.single2uint(msicube)
    save_img_path = os.path.join('FalseHSIcolor.png')
    util.imsave(msicube_uint[:, :, [22, 12, 4]], save_img_path)
    save_img_path = os.path.join('FalseHSI.tif')
    util.save_msicube(msicube_uint.transpose(2, 1, 0), save_img_path)
