# -*- coding: utf-8 -*-
# Python 3.7.7
# Pandas 1.0.5
# Author: Ioannis Ropotos


"""
Replication code of Corwin & Schultz (2012) - A simple way to estimate
bid-ask spreads from daily high and low prices (JF).

Supplementary material for the paper can be found in Corwin's website:
https://sites.nd.edu/scorwin/research/

The estimator uses daily high and low prices to isolate the true spread of 
a stock from its return variance by noting that high (low) prices are most
likely buy (sell) orders.

The script follows the logic and structure of the SAS code provided 
in Corwin's website. It should be regarded as a translation from SAS to Python.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Workind directory
wdir = r'C:\Users\ropot\Desktop\Python Scripts\Corwin_Schultz_2012'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     IMPORT AND FILTER DATASET         #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Import CRSP daily data (from WRDS)
# The necessary columns should be date, PERMNO, high and low price, 
# close price (all prices are split-adjusted).
ctotype32 = {'date' : np.int32,
             'PERMNO' : np.int32,
             'SHRCD' : np.float32,
             'EXCHCD' : np.float32,
             'BIDLO' : np.float32,
             'ASKHI' : np.float32,
             'PRC' : np.float32,
             'VOL' : np.float32}
crspd = pd.read_csv(os.path.join(wdir, 'CRSP_daily_PRC_2020and2021.csv'))
# Before that, I may need to rename the CRSP columns
crspd = crspd.astype(ctotype32)


# Filter for common shares (SHRCD = 10, 11) and 
# major exchanges (EXCHCD = 1,2,3)
shrcd_filter = [10, 11]
shrcd_mask = crspd['SHRCD'].isin(set(shrcd_filter))
exchd_filter = [1, 2 ,3]
exchd_mask = crspd['EXCHCD'].isin(set(exchd_filter))
# Apply filter
crspd = crspd.loc[shrcd_mask & exchd_mask].copy()


# Rename BIDLO and ASKHI for consistency with the original 
# SAS code.
sas_prc_cols = {'BIDLO' : 'LOPRC',
                'ASKHI' : 'HIPRC'}
crspd = crspd.rename(columns = sas_prc_cols)



# Function that calculates the Corwin-Schultz (2012) 
# high-low price estimates
def corwin_schultz_2012_estimator(crspd):
    """    
    Replication code of Corwin & Schultz (2012) - A simple way to estimate
    bid-ask spreads from daily high and low prices (JF).
    
    Parameters
    ----------
    crspd : DataFrame
        crspd is the CRSP daily dataset with the following columns:
            date : daily date in YYYYmmdd integer format
            PERMNO : permanent CRSP security identifier
            LOPRC : low daily price (BIDLO)
            HIPRC : high daily price (ASKHI)
            PRC : end-of-day closing price
            VOL : number of trades in a day

    Returns
    -------
    mspreads : DataFrame
        mspreads is the final output of the function that contains 
        the monthly high-low price estimates of Corwin & Schultz (2012)
        with the following columns:
            date_m : monthly date in YYYYmm integer format
            PERMNO : permanent CRSP security identifier
            MSPREAD : monthly average of SPREAD
            MSPREAD_0 : monthly average of SPREAD_0 where negative 
                        daily spreads are set to zero -- PRIMARY ESTIMATE
            MSPREAD_MISS : monthly average of SPREAD_MISS
            XSPREAD_0 : negative values of MSPREAD are set to zero.
            MSIGMA : monthly average of SIGMA
            MSIGMA_0 : monthly average of SIGMA_0 where the daily 
                       negative estimates of SIGMA are set to zero.

    """
    
    # /////////////////////////////////////////////////////////////////
    #                START OF CORWIN-SCHULTZ CODE 
    # /////////////////////////////////////////////////////////////////
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   CONSTRUCT GOOD HIGH-LOW PRICES     #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # Create high-low reset index (HLRESET)
    # It is zero, if no change was needed. Otherwise it is positive (1,2,3)
    crspd['HLRESET'] = 0
    
    # Check if there was no trade and create INOTRADE column 
    trade_mask1 = crspd['PRC'] <= 0
    trade_mask2 = crspd['VOL'] == 0
    crspd['INOTRADE'] = np.where(trade_mask1 | trade_mask2, 1, 0)
    # Check if the high-low prices are the same and create ISAMEPRC columns
    sameprc = crspd['LOPRC'] == crspd['HIPRC']
    crspd['ISAMEPRC'] = np.where(sameprc, 1, 0)
    
    # Set bad high-low prices to null 
    neg_low = crspd['LOPRC'] <= 0
    neg_high = crspd['HIPRC'] <= 0
    notrade = crspd['INOTRADE'] == 1
    # Low price
    crspd['LOPRC'] = np.where(neg_low | notrade | sameprc, np.nan, crspd['LOPRC'])
    # High price
    crspd['HIPRC'] = np.where(neg_high | notrade | sameprc, np.nan, crspd['HIPRC'])
    
    # Make closing prices positive
    crspd['PRC'] = np.abs(crspd['PRC'])
    
    # Set with null values the high-low prices on the first trading day
    # of the security.
    first_date_idx = crspd.groupby('PERMNO').apply(lambda x: x.index[0]).values
    # High price
    crspd.loc[first_date_idx, 'HIPRC'] = np.nan
    # Low price
    crspd.loc[first_date_idx, 'LOPRC'] = np.nan
    
    # Replace missing/bad high-low prices with retained values
    order_mask1 = 0 <= crspd['LOPRC']
    order_mask2 = crspd['LOPRC'] <= crspd['HIPRC']
    order_prc = order_mask1 & order_mask2
    order_prc_not = ~ order_prc
    
    # Create columns for previous day prices
    crspd['PRC_lag'] = crspd.groupby('PERMNO')['PRC'].shift()
    crspd['HIPRC_lag'] = crspd.groupby('PERMNO')['HIPRC'].shift()
    crspd['LOPRC_lag'] = crspd.groupby('PERMNO')['LOPRC'].shift()
    
    
    #                  HLRESET = 1
    
    # Replace current high-low prices with previous day's prices
    # if current PRC is within prior's day range 
    hlreset1_mask1 = crspd['LOPRC_lag'] <= crspd['PRC'] 
    hlreset1_mask2 = crspd['PRC'] <= crspd['HIPRC_lag']
    hlreset1_mask = hlreset1_mask1 & hlreset1_mask2
    # High price
    crspd['HIPRC'] = np.where(order_prc_not & hlreset1_mask, crspd['HIPRC_lag'], crspd['HIPRC'])
    # Low price
    crspd['LOPRC'] = np.where(order_prc_not & hlreset1_mask, crspd['LOPRC_lag'], crspd['LOPRC'])
    # Code the reason of replacement with the value of 1
    crspd['HLRESET'] = np.where(order_prc_not & hlreset1_mask, 1, crspd['HLRESET'])
    
    #                  HLRESET = 2
    
    # Replace current high-low prices 
    # if current PRC is below the prior's day range 
    hlreset2_mask = crspd['PRC'] < crspd['LOPRC_lag']
    # High price
    crspd['HIPRC'] = np.where(order_prc_not & hlreset2_mask,
                              crspd['HIPRC_lag'] - (crspd['LOPRC_lag'] - crspd['PRC']),
                              crspd['HIPRC'])
    # Low price
    crspd['LOPRC'] = np.where(order_prc_not & hlreset2_mask, crspd['PRC'], crspd['LOPRC'])
    # Code the reason of replacement with the value of 2
    crspd['HLRESET'] = np.where(order_prc_not & hlreset2_mask, 2, crspd['HLRESET'])
    
    
    #                  HLRESET = 3
    
    # Replace current high-low prices 
    # if current PRC is above the prior's day range 
    hlreset3_mask = crspd['PRC'] > crspd['HIPRC_lag']
    # High price
    crspd['HIPRC'] = np.where(order_prc_not & hlreset3_mask, crspd['PRC'], crspd['HIPRC'])
    # Low price
    crspd['LOPRC'] = np.where(order_prc_not & hlreset3_mask,
                              crspd['LOPRC_lag'] + (crspd['PRC'] - crspd['HIPRC_lag']),
                              crspd['LOPRC'])
    
    # Code the reason of replacement with the value of 3
    crspd['HLRESET'] = np.where(order_prc_not & hlreset3_mask, 3, crspd['HLRESET'])
    
    
    # A ratio of high/low price over 8 is not allowed
    zero_low_not = crspd['LOPRC'] != 0
    hl_ratio = crspd['HIPRC']/crspd['LOPRC'] > 8 
    # High price
    crspd['HIPRC'] = np.where(zero_low_not & hl_ratio, np.nan, crspd['HIPRC'])
    # Low price
    crspd['LOPRC'] = np.where(zero_low_not & hl_ratio, np.nan, crspd['LOPRC'])
    
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   ADJUST FOR OVERNIGHT RETURNS   #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
    # Create overnight adjustement index (RETADJ)
    # It is zero, if no change was needed. Otherwise it is positive (1,2)
    crspd['RETADJ'] = 0
    
    
    #             RETADJ = 1 
    
    # Adjust when prior close is below the current low price
    retadj1_mask1 = crspd['PRC_lag'] < crspd['LOPRC']
    retadj1_mask2 = crspd['PRC_lag'] > 0 
    retadj1_mask = retadj1_mask1 & retadj1_mask2
    # High price
    crspd['HIPRC'] = np.where(retadj1_mask, 
                              crspd['HIPRC'] - (crspd['LOPRC']-crspd['PRC_lag']),
                              crspd['HIPRC'])
    # Low price
    crspd['LOPRC'] = np.where(retadj1_mask, crspd['PRC_lag'], crspd['LOPRC'])
    # Code the reason of adjustment with the value of 1
    crspd['RETADJ'] = np.where(retadj1_mask, 1, crspd['RETADJ'])
    
    
    #             RETADJ = 2 
    
    # Adjust when prior close is above the current high price
    retadj2_mask1 = crspd['PRC_lag'] > crspd['HIPRC']
    retadj2_mask2 = crspd['PRC_lag'] > 0
    retadj2_mask = retadj2_mask1 & retadj2_mask2
    # High price
    crspd['HIPRC'] = np.where(retadj2_mask, crspd['PRC_lag'], crspd['HIPRC'])
    # Low price
    crspd['LOPRC'] = np.where(retadj2_mask, 
                              crspd['LOPRC'] + (crspd['PRC_lag']-crspd['HIPRC']),
                              crspd['LOPRC'])
    
    # Code the reason of adjustment with the value of 2
    crspd['RETADJ'] = np.where(retadj2_mask, 2, crspd['RETADJ'])
    
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  CALCULATE DAILY HIGH-LOW SPREAD ESTIMATES     #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
    # Define constants
    PI = np.pi
    K = 1/(4*np.log(2))
    K1 = 4*np.log(2)
    K2 = np.sqrt(8/PI)
    CONST = 3 - 2*np.sqrt(2)
    # The nanmax operator is faster than pandas.max(). 
    # It returns a NaN value only when both operands are NaN
    crspd['HIPRC2'] = np.nanmax(crspd[['HIPRC', 'HIPRC_lag']].values, axis=1)
    crspd['LOPRC2'] = np.nanmin(crspd[['LOPRC', 'LOPRC_lag']].values, axis=1)
    # Get BETA
    beta_mask1 = crspd['LOPRC'] > 0
    beta_mask2 = crspd['LOPRC_lag'] > 0
    beta_mask = beta_mask1 & beta_mask2
    crspd['BETA'] = np.where(beta_mask, 
                             (np.log(crspd['HIPRC']/crspd['LOPRC']))**2 + 
                             (np.log(crspd['HIPRC_lag']/crspd['LOPRC_lag']))**2,
                             np.nan)
    # Get GAMMA
    gamma_mask = crspd['LOPRC2'] > 0 
    crspd['GAMMA'] = np.where(gamma_mask,
                              (np.log(crspd['HIPRC2']/crspd['LOPRC2']))**2,
                              np.nan)
    # Get ALPHA
    crspd['ALPHA'] = (np.sqrt(2*crspd['BETA']) - np.sqrt(crspd['BETA']))/CONST  \
                     - np.sqrt(crspd['GAMMA']/CONST)
    # Get the daily spread
    crspd['SPREAD'] = 2*(np.exp(crspd['ALPHA'])-1)/(1+np.exp(crspd['ALPHA'])) 
    
    
    # SPREAD_0 : negative spread estimates set to zero
    crspd['SPREAD_0'] = np.where(crspd['SPREAD'].isnull(), 
                                 np.nan, np.maximum(crspd['SPREAD'], 0) )
    
    # SPREAD_MISS : drop negative daily estimates
    miss_mask = crspd['SPREAD'] > 0
    crspd['SPREAD_MISS'] = np.where(miss_mask, crspd['SPREAD'], np.nan)
    
    # Get the sigma (standard deviation of the GBM)
    crspd['SIGMA'] = ((np.sqrt(crspd['BETA']/2)-np.sqrt(crspd['BETA']))) \
                     /(K2*CONST)+np.sqrt(crspd['GAMMA']/(K2*K2*CONST))
                     
    # SIGMA_0 : negative sigmas are set to zero
    crspd['SIGMA_0'] = np.where(crspd['SIGMA'].isnull(), 
                                 np.nan, np.maximum(crspd['SIGMA'], 0) )
    
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  CALCULATE MONTHLY HIGH-LOW SPREAD ESTIMATES    #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # Define the date_m column (month) in YYYYmm integer format
    crspd['date_m'] = np.floor(crspd['date']/100).astype(np.int32)
    
    # Isolate the necessary columns
    keep_cols = ['PERMNO', 
                 'date',
                 'date_m', 
                 'SPREAD', 
                 'SPREAD_0', 
                 'SPREAD_MISS',
                 'SIGMA',
                 'SIGMA_0']
    spreads = crspd[keep_cols].copy()
    
    # Delete crspd dataset to free memory
    del crspd
    
    # Monthly spreads are averages of daily spreads in a given month
    # when at least 12 non-null values have been retained.
    
    # Create a new dataframe to store monthly spreads
    mspreads = spreads[['PERMNO', 'date_m']].drop_duplicates().reset_index(drop = True).sort_values(by = ['PERMNO', 'date_m'])
    
    # Count the number of non-null values for SPREAD and SPREAD_0
    # The tranform() method is a bit slow
    spreads['COUNT'] = spreads.groupby(['PERMNO', 'date_m'])['SPREAD'].transform(lambda x: x.count()) 
    count_mask = spreads['COUNT'] >= 12
    # Count the number of non-null values for SPREAD_MISS
    spreads['COUNT_MISS'] = spreads.groupby(['PERMNO', 'date_m'])['SPREAD_MISS'].transform(lambda x: x.count())
    count_miss_mask = spreads['COUNT_MISS'] >= 12
    
    
    # MSPREAD 
    mspread = spreads[count_mask].groupby(['PERMNO', 'date_m'])['SPREAD'].mean().reset_index()
    mspread = mspread.rename(columns = {'SPREAD' : 'MSPREAD'})
    mspreads = pd.merge(mspreads, mspread, how = 'left', on = ['PERMNO', 'date_m'])
    del mspread
    
    # MSPREAD_0 -- primary HL monthly estimate
    mspread_0 = spreads[count_mask].groupby(['PERMNO', 'date_m'])['SPREAD_0'].mean().reset_index()
    mspread_0 = mspread_0.rename(columns = {'SPREAD_0' : 'MSPREAD_0'})
    mspreads = pd.merge(mspreads, mspread_0, how = 'left', on = ['PERMNO', 'date_m'])
    del mspread_0
    
    # MSPREAD_MISS
    mspread_miss = spreads[count_miss_mask].groupby(['PERMNO', 'date_m'])['SPREAD_MISS'].mean().reset_index()
    mspread_miss = mspread_miss.rename(columns = {'SPREAD_MISS' : 'MSPREAD_MISS'})
    mspreads = pd.merge(mspreads, mspread_miss, how = 'left', on = ['PERMNO', 'date_m'])
    del mspread_miss
    
    # MSIGMA
    msigma = spreads[count_mask].groupby(['PERMNO', 'date_m'])['SIGMA'].mean().reset_index()
    msigma = msigma.rename(columns = {'SIGMA' : 'MSIGMA'})
    mspreads = pd.merge(mspreads, msigma, how = 'left', on = ['PERMNO', 'date_m'])
    del msigma
    
    # MSIGMA_0
    msigma_0 = spreads[count_mask].groupby(['PERMNO', 'date_m'])['SIGMA_0'].mean().reset_index()
    msigma_0 = msigma_0.rename(columns = {'SIGMA_0' : 'MSIGMA_0'})
    mspreads = pd.merge(mspreads, msigma_0, how = 'left', on = ['PERMNO', 'date_m'])
    del msigma_0
    
    # XSPREAD_0 : set negative MSPREAD estimates to zero
    mspreads['XSPREAD_0'] = np.where(mspreads['MSPREAD'].isnull(), 
                                 np.nan, np.maximum(mspreads['MSPREAD'], 0) )
    
    
    # /////////////////////////////////////////////////////////////////
    #                    END OF CORWIN-SCHULTZ CODE 
    # /////////////////////////////////////////////////////////////////
    
    return mspreads
    



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     APPLY THE ESTIMATOR       #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mspreads = corwin_schultz_2012_estimator(crspd)
print(mspreads.head(20))
# Save it
mspreads.to_csv(os.path.join(wdir,'monthly_HL_estimates_2020_2021.csv'), index = False)

# Plot the distribution of HL estimates
plt.figure()
mspreads['MSPREAD_0'].plot(kind = 'hist', bins = 100)
plt.xlabel('HL estimate')
plt.ylabel('Frequency')
plt.savefig(os.path.join(wdir, 'Histogram_HL_montly_estimates.png'))







