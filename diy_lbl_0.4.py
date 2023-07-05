## Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from astropy import constants
from astropy.io import fits
from lbl.core import math as mp
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import lmfit # source: https://lmfit.github.io/lmfit-py/
from lmfit import minimize, Parameters, fit_report
import multiprocessing
import time
from astropy.table import Table
import os

## Doppler shift

speed_of_light_ms = constants.c.value

def doppler_shift(wavegrid: np.ndarray, velocity: float) -> np.ndarray:
    """
    Apply a doppler shift

    :param wavegrid: wave grid to shift
    :param velocity: float, velocity expressed in m/s

    :return: np.ndarray, the updated wave grid
    """
    # relativistic calculation (1 - v/c)
    part1 = 1 - (velocity / speed_of_light_ms)
    # relativistic calculation (1 + v/c)
    part2 = 1 + (velocity / speed_of_light_ms)
    # return updated wave grid
    return wavegrid * np.sqrt(part1 / part2)

hp_width = 500 * 1000 # SPIRou HP_WIDTH = 500km/s

def get_velo_scale(wave_vector: np.ndarray, hp_width: float) -> int:
    """
    Calculate the velocity scale give a wave vector and a hp width

    :param wave_vector: np.ndarray, the wave vector
    :param hp_width: float, the hp width

    :return: int, the velocity scale in pixels
    """
    # work out the velocity scale
    dwave = np.gradient(wave_vector)
    dvelo = 1 / np.nanmedian((wave_vector / dwave) / speed_of_light_ms)
    # velocity pixel scale (to nearest pixel)
    width = int(hp_width / dvelo)
    # make sure pixel width is odd
    if width % 2 == 0:
        width += 1
    # return  velocity scale
    return width

## Load template and define line frontier

def line_frontier(lf_temp_flux, lf_temp_wave):
    d = np.gradient(lf_temp_flux, lf_temp_wave)
    dd = np.gradient(d, lf_temp_wave)
    linefrontier = [lf_temp_wave[0]]
    for l in range(1,len(lf_temp_wave)-1):
        if d[l]*d[l+1]<0 and dd[l]<0 :
            linefrontier.append(lf_temp_wave[l])
    linefrontier.append(lf_temp_wave[-1])
    return(linefrontier)

template = fits.open('/media/paul/One Touch2/SPIRou_Data/0.7.275/Gl_410/GL410_APERO/Template_s1d_GL410_sc1d_v_file_AB.fits')

temp_flux = template[1].data['flux']
temp_wave = template[1].data['wavelength']
temp_eflux = template[1].data['eflux']
temp_rms = template[1].data['rms']

linefront = np.array(line_frontier(temp_flux, temp_wave))

## Interpolate spectrums

def popnan(x, y):
    xout = x[np.invert(np.isnan(y))]
    yout = y[np.invert(np.isnan(y))]
    return(xout, yout)


def regrid(a, grid_a, b, grid_b):
    grid_start, grid_end = np.min(np.concatenate((grid_a, grid_b))), np.max(np.concatenate((grid_a, grid_b)))
    common_grid = np.linspace(grid_start, grid_end, len(grid_a)+len(grid_b))
    pop_grid_a, pop_a = popnan(grid_a, a)
    pop_grid_b, pop_b = popnan(grid_b, b)
    spl_a = CubicSpline(pop_grid_a, pop_a)
    spl_b = CubicSpline(pop_grid_b, pop_b)
    interpolate_a = spl_a(common_grid)
    interpolate_b = spl_b(common_grid)
    return(common_grid, interpolate_a, interpolate_b)

## Valid ?

def valid_condition(a, grid_a, b, grid_b):
    if len(grid_a) <= 3:
        return(False)
    elif len(grid_b) <= 3:
        return(False)
    elif np.sum(np.isfinite(a)) <= len(a)/2:
        return(False)
    elif np.sum(np.isfinite(b)) <= len(b)/2:
        return(False)
    elif np.isnan(a[0]):
        return(False)
    elif np.isnan(b[0]):
        return(False)
    elif np.isnan(a[-1]):
        return(False)
    elif np.isnan(b[-1]):
        return(False)
    else:
        return(True)

## Bouchy equation

def dv_template(tflux, twave):
    # get the derivative of the flux
    dflux = np.gradient(tflux, twave)
    # get the 2nd derivative of the flux
    d2flux = np.gradient(dflux, twave)
    # get the 3rd derivative of the flux
    d3flux = np.gradient(d2flux, twave)
    return(dflux, d2flux, d3flux)


def equation(params, d1, d2, d3, diff_seg):
    d0v = params.get('d0v').value
    dv  = params.get('dv').value
    d2v = params.get('d2v').value
    d3v = params.get('d3v').value
    zero = d0v + dv*d1 + d2v*d2 + d3v*d3 - diff_seg
    return(zero)

## Main func

def mainit(i):
    #Line selection routine
    s, e = linefront[i], linefront[i+1]
    select_bool = np.logical_and(twave >= s, twave <= e)
    li_r_wave, li_r_flux = twave[select_bool], tflux[select_bool]
    select_bool = np.logical_and(temp_wave >= s, temp_wave <= e)
    li_t_wave, li_t_flux = temp_wave[select_bool], lowpass_temp_flux[select_bool]

    #validity test
    if valid_condition(li_r_flux, li_r_wave, li_t_flux, li_t_wave):
        #interpolation
        interwave, interpotemp, interpoflux = regrid(li_t_flux, li_t_wave, li_r_flux, li_r_wave)

        #calculate dvsplines
        d1, d2, d3 = dv_template(interpotemp, interwave)
        diff_seg = interpoflux - interpotemp

        #fit
        params_ini = Parameters()
        params_ini.add('d0v', value=0, min=-np.inf, max=np.inf)
        params_ini.add('dv', value=0, min=-np.inf, max=np.inf)
        params_ini.add('d2v', value=0, min=-np.inf, max=np.inf)
        params_ini.add('d3v', value=0, min=-np.inf, max=np.inf)
        out = minimize(equation, params_ini, args = (d1, d2, d3, diff_seg))

        #fill resluts
        d0v = out.params['d0v'].value
        dv = out.params['dv'].value
        d2v = out.params['d2v'].value
        d3v = out.params['d3v'].value

        sd0v = out.params['d0v'].stderr
        sdv = out.params['dv'].stderr
        sd2v = out.params['d2v'].stderr
        sd3v = out.params['d3v'].stderr

    else:
        d0v, dv, d2v, d3v, sd0v, sdv, sd2v, sd3v = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    return(int(i), d0v, dv, d2v, d3v, sd0v, sdv, sd2v, sd3v)

##run


path = '/media/paul/One Touch2/SPIRou_Data/0.7.275/Gl_410/GL410_APERO/tcorr' #### PATH TO CHANGE ####
os.chdir(path)
for (root, dirs, file) in os.walk(path):
    for f in tqdm(sorted(file)):
        if 'tcorr_AB.fits' in f:
            tfile = fits.open(f, memmap=False)

            tflux = tfile[1].data['flux']
            tburv = tfile[0].header['BERV']
            ttime = tfile[0].header['BJD']
            twave = doppler_shift(tfile[1].data['wavelength'], -tburv*1000)
            teflux = tfile[1].data['eflux']
            tweight = tfile[1].data['weight']

            width = get_velo_scale(temp_wave, hp_width)
            lowpass_temp_flux = np.copy(temp_flux)
            lowpass_temp_flux -= mp.lowpassfilter(tflux, width=width)

            ratio = mp.lowpassfilter(tflux / lowpass_temp_flux, int(width))
            lowpass_temp_flux *= ratio

            #run paralellized
            with multiprocessing.Pool() as pool:
                results = pool.map(mainit, range(len(linefront)-1))
            results = np.array(results).T

            #Save results:
            Table_results= Table([linefront[np.array(results[0], dtype=int)], linefront[np.array(results[0], dtype=int)+1], results[1], results[2], results[3], results[4], results[5], results[6], results[7], results[8]], names=('wave_start', 'wave_end', 'd0v', 'dv', 'd2v', 'd3v', 'sd0v', 'sdv', 'sd2v', 'sd3v'))
            Table_results.write( '/media/paul/One Touch2/LBL_DIY_output/GL410/' + tfile[0].header['FILENAME'] + '_diylbl.fits', format='fits')
            fits.setval('/media/paul/One Touch2/LBL_DIY_output/GL410/' + tfile[0].header['FILENAME'] + '_diylbl.fits', 'BJD', value = ttime)
            fits.setval('/media/paul/One Touch2/LBL_DIY_output/GL410/' + tfile[0].header['FILENAME'] + '_diylbl.fits', 'BERV', value = tburv)
