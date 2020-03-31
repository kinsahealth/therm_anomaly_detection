import pandas as pd
import numpy as np
import datetime as dt

''' Generation time (w): Rate at which flu spreads from host 1-5 days
See Carrat et al. AJE 2018 https://doi.org/10.1093/aje/kwm375
Modeled by a gamma distribution and hardcoded below. Here, the average
generation time is 2.5 days (range 1.4-4.0 days).
Can be computed as follows:

from scipy.stats import gamma
mean_w = 2.5
var_w = 0.6
w = gamma.pdf(x=[1, 2, 3, 4, 5], a=mean_w**2/var_w, scale=var_w/mean_w)
w = w/np.sum(w)
'''
w = [0.04753979, 0.50372795, 0.35549083, 0.08275124, 0.01049019]


def ibar(ili, w):
    ''' Effective incidence (ibar) is the expected flu incidence
    based upon past incidence levels and the contact rate at which
    flu spreads over the past 5 days. This is commonly modeled with
    a gamma distribution, and via matrix multiplication of the past
    5 days of ILI and w one can compute effective incidence'''

    return np.flip(ili).dot(w)


def get_epivars(df, w):
    ''' A few variables derived from the Kinsa ILI are required to
    calculate the R0 forecast. These include effective incidence (ibar),
    R0 (referred to here as R), and the spread contact rate for flu (w).
    We hardcode w, as this is estimated from literature, and compute the
    rest for each region. The daily mean R is used to propogate the
    long-lead forecast.
    '''

    df['ibar'] = df['percent_ill'].rolling(5).apply(lambda x: ibar(x, w))
    df['ibar'].iloc[:5] = np.mean(df['percent_ill'].iloc[:5])

    # R0 is smoothed data to control volatility that impacts predictions
    df['R'] = (df['percent_ill'].shift(-1).rolling(30).mean(center=True) /
               df['ibar'].rolling(30).mean(center=True))
    df['R'].fillna(df['R'].mean(), inplace=True)

    return df


def r0_forecast(df, w, horizon, social_mod=None):
    ''' Forward propogated forecast using the daily mean
    reproductive number of a region. Current ibar is multiplied
    by mean daily R0 to get ILI at the next time point and
    repeated until the forecast horizon is completed.
    R0 is estimated from all past data.'''

    cutoff = df['ds'].max()
    horiz_days = horizon*7
    incubation_days = len(w)

    # Get the mean daily R
    avgR = (df.groupby('doy')['R']
            .median().reset_index()
            .rename(columns={'R': 'avg_r'})
            )
    if social_mod is not None:
        ''' Major social distancing begins with school closures on
        March 16, 2020 (DOY 76). This is hardcoded for Science example
        in Brooklyn, NY '''
        avgR['avg_r'] = np.where(avgR['doy'] >= 76, avgR['avg_r']*social_mod,
                                 avgR['avg_r'])

    # Prepare timeseries df for prediction
    pred_dates = pd.date_range(cutoff - dt.timedelta(4),
                               cutoff + dt.timedelta(horiz_days))
    forc_df = pd.DataFrame({'ds': pred_dates})
    forc_df['doy'] = forc_df.ds.dt.dayofyear

    # Get average R and the last 5 days of ILI required for forecasting
    forc_df = forc_df.merge(avgR, on='doy')
    forc_df = forc_df.merge(df[['percent_ill', 'ds']], on='ds', how='left')

    # Predict next ILI timestep based on ibar and R0
    for i in np.arange(incubation_days, (horiz_days+incubation_days)):
        ibar_out = ibar(forc_df.iloc[(i-5):i]['percent_ill'], w)
        forc_df.loc[i, 'percent_ill'] = ibar_out * forc_df.loc[i, 'avg_r']

    # Save forecasts only
    forc_df = forc_df[forc_df.ds > cutoff]
    forc_df['cutoff'] = cutoff

    return forc_df[['ds', 'cutoff', 'percent_ill']]


def anomaly_wrapper(df, run_dates, horizon, simulations=1, social_mod=None):
    ''' Following the same format as local effects forecast,
    we make forecasts across a number of regions on a given day
    and then use this output to define an expected influenza range.
    '''

    dfs = []

    for i, region in enumerate(df['region'].unique()):
        print(f'Forecasting for {region}')
        region_df = df[df.region == region].copy()
        region_df.sort_values('ds').reset_index(drop=True, inplace=True)
        region_df = get_epivars(region_df, w)

        for date in run_dates:
            forc_df = region_df[region_df.ds <= date].copy()

            # Get in-season error rate for simulations
            forc_df['error'] = (forc_df['percent_ill']
                                - forc_df['percent_ill'].rolling(30).mean(center=True))
            error_scale = np.std(forc_df.error)

            # No-peaking in backfill, 30day rolling R leaks data
            # THIS IS REALLY IMPORTANT
            forc_df['R'].iloc[-15:] = np.nan

            # Jitter the starting point to estimate uncertainty
            forc_list = []
            for i in np.arange(0, simulations):
                sim_df = forc_df.copy()

                if i == 0:
                    sim_df = r0_forecast(sim_df, w, horizon=horizon,
                                         social_mod=social_mod)
                else:
                    # for all additional forecasts add ili-based error
                    sim_df['percent_ill'] = (
                        sim_df['percent_ill'] +
                        np.random.normal(loc=0, scale=error_scale)
                        )
                    sim_df = r0_forecast(sim_df, w, horizon=horizon,
                                         social_mod=social_mod)

                sim_df['run'] = i
                forc_list.append(sim_df)

            forc_df = pd.concat(forc_list, axis=0)
            forc_df['region'] = region
            forc_df['error_scale'] = error_scale
            dfs.append(forc_df)
            del forc_df

        del region_df

    return pd.concat(dfs, axis=0)