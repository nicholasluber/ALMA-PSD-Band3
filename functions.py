import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
import scipy.constants as sc
from astropy.timeseries import LombScargle
from numba import *

def extract_each_base_vis(data):
    """
    Extract the time series visibilities for each baseline.
    Sort the data.
    """
    
    # Collect all the data into a more understandable array.
    final = np.zeros((903,int(len(data)/903),4))
    for ii in range(0,903): # Loop through each baseline.
        for jj in range(0,len(final[0])):
            final[ii][jj][0]=data[(jj*903)+ii][0]
            final[ii][jj][1]=data[(jj*903)+ii][1]
            final[ii][jj][2]=data[(jj*903)+ii][2]
            final[ii][jj][3]=data[(jj*903)+ii][3]
    
    # Go through and sort by baseline length.
    # First go through and find the index order of baseline lengths.
    lengths = np.zeros(903)
    for ii in range(0,903):
        temp = np.zeros((int(len(final)/903)))
        for jj in range(0,len(temp)):
            temp[jj] = final[ii][jj][1]
        lengths[ii] = np.mean(temp)
    sorted_lengths = np.argsort(lengths)
    
    # Go through and create an array that is now properly sorted.
    sorted_final = np.zeros(final.shape)
    for ii in range(0,903):
        sorted_final[ii] = final[sorted_lengths[ii]]
        
    return sorted_final


def base_phys(base, freq, dist):
    """
    Convert baseline lengths to size scale on the sun.
    Frequency for Band 3: 99.869 GHz
    Distance to Sun for Band 3 observatioons: 149.6e6 km
    """
    
    # Formula for resolution given by: https://almascience.eso.org/documents-and-tools/cycle-1/alma-es-primer
    # Eq. 1: ~0.2” x (300/freq) x [(1/baseline_length)^-1] -Baseline length in kilometers, frequency in GHz.
    # Eq. 2: tan(theta) = resolution/distance
    
    # Use Eq. 1 to find the angle per baseline.
    thetas_deg = (0.2*(300/freq)*(1/base))/3600
    thetas = np.radians(thetas_deg)
    
    # Use previous results and Eq. 2.
    angular_size = (np.tan(thetas)*dist)/1e3
    
    # Return results in Megameters.
    return angular_size

def tau_calc(omega, time_array):
    """
    Calculate the tau factor in the Lomb-Scargle periodogram.
    """
    return np.arctan(np.sum(np.sin(2*omega*time_array))/np.sum(np.cos(2*omega*time_array)))/(2*omega)


def lomb_scargle_iter(data, omega_iter, times):
    """
    Do the Lomb-Scargle calcullation for a single frequency.
    """
    
    tau = tau_calc(omega_iter, times)# Calculate the time delay.
    
    # Calculate the terms in the iteration.
    num1 = np.sum(data*np.cos(omega_iter*(times-tau)))**2
    den1 = np.sum(np.cos(omega_iter*(times-tau))**2)
    num2 = np.sum(data*np.sin(omega_iter*(times-tau)))**2
    den2 = np.sum(np.sin(omega_iter*(times-tau))**2)
    
    # Return the values.
    return 0.5*((num1/den1)+(num2/den2))


def Lomb_Scargle(freqs, alltimes, alldata):
    """
    Compute the Lomb-Scargle power spectrum.
    """
    
    final = np.zeros(len(freqs),dtype=complex)
    for ii in range(0,len(final)):
        final[ii] = lomb_scargle_iter(alldata, freqs[ii], alltimes)
    return (1/len(freqs))*np.abs(final)**2


def avg_data_ebin_comp(datain, Nbin):
    """
    1. Average the data into bins of equal amounts of baselines.
    2. Find the absolute value of the complex average baseline.
    3. Zero mean the data.
    4. Calculate the probed size-scales.
    """
    
    # This code chunk calculates the base boundaries.
    avgbases = np.average(datain[:,:,1],axis=1)
    boundindlist = np.arange(0,len(avgbases),len(avgbases)/Nbin,dtype=int).tolist()
    boundindlist.append(len(avgbases)-1)
    boundind = np.array(boundindlist)
    sortbases = np.sort(avgbases)
    base_bounds = np.zeros(len(boundind))
    for ii in range(0,len(boundind)):
        base_bounds[ii] = sortbases[boundind[ii]]
    base_bounds[-1] += 1
    

    # Average the data down.
    finaldata = np.zeros((Nbin,len(datain[0]),4)) # The averaged data.
    absdata = np.zeros((Nbin,len(datain[0]),3)) # The averaged absolute value data.
    for ii in range(1,len(base_bounds)): # Loop through each bin.
        kk = 0 # Counter for the averaging.
        for jj in range(0,len(datain)): # Loop through each visibility pair to see if its in the bin.
            if np.average(datain[jj,:,1]) < base_bounds[ii] and np.average(datain[jj,:,1]) >= base_bounds[ii-1]:
                finaldata[ii-1] += datain[jj]
                kk += 1
        finaldata[ii-1] *= (1/kk) # Properly average.
        
    # Calculate average size on sun probed.
    scales = np.zeros(len(finaldata))
    for ii in range(0,len(finaldata)):
        scales[ii] = np.average(finaldata[ii,:,1])
        
    # Return a list of relevant arrays.
    return [scales, finaldata, base_bounds]


def LS_avg_vis(avgdata, ind, freq):
    """
    Run the Lomb-Scargle periodogram on the averaged visbility data.
    """
    
    # First put the data into a single array of time and complex data.
    time = avgdata[ind,:,0]
    comp_data = np.zeros(len(time), dtype=complex)
    for ii in range(0,len(comp_data)):
        comp_data[ii] = complex(avgdata[ind,ii,2],avgdata[ind,ii,3])
        
    # Subtract of a second order fit from the averaged visibility data.
    fit = np.polyfit(time, comp_data, 2)
    fitdata = (fit[0]*time*time)+(fit[1]*time)+fit[2]
    subdata = comp_data - fitdata
    
    # Do the power spectrum calculation.
    power = Lomb_Scargle(freq, time, subdata)
    
    return power
