import os
import zipfile
import urllib.request
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


TAXI_ZONES_ZIP_URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
TAXI_ZONES_SHP_PATH = 'data/00-raw/taxi_zones/taxi_zones/taxi_zones.shp'


def load_taxi_zones_geodata(shp_path=TAXI_ZONES_SHP_PATH,
                            url=TAXI_ZONES_ZIP_URL):
    """Download (if needed), extract, and return the NYC taxi zones GeoDataFrame."""
    if not os.path.exists(shp_path):
        zip_path = 'data/00-raw/taxi_zones.zip'
        extract_dir = 'data/00-raw/taxi_zones'
        print(f'Downloading taxi zone shapefile from TLC CDN ...')
        urllib.request.urlretrieve(url, zip_path)
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        print(f'Extracted to {extract_dir}')

    gdf = gpd.read_file(shp_path)
    gdf['LocationID'] = gdf['LocationID'].astype(int)
    return gdf


def aggregate_demand_by_zone(df):
    """Total trip count per pickup zone over the entire dataset."""
    zone_demand = (
        df.groupby(['PULocationID', 'Borough', 'Zone'])
        .size()
        .reset_index(name='total_trips')
        .sort_values('total_trips', ascending=False)
    )
    return zone_demand


def aggregate_daily_demand_by_zone(df):
    """Trip count per pickup zone per day."""
    daily = (
        df.groupby(['date', 'PULocationID', 'Borough', 'Zone'])
        .size()
        .reset_index(name='daily_trips')
    )
    return daily


def aggregate_demand_by_borough(df):
    """Total trip count per borough."""
    borough_demand = (
        df.groupby('Borough')
        .size()
        .reset_index(name='total_trips')
        .sort_values('total_trips', ascending=False)
    )
    return borough_demand


def plot_choropleth(gdf, column, title, cmap='YlOrRd', ax=None,
                    figsize=(14, 14), legend=True, log_scale=False):
    """Plot a choropleth map of a GeoDataFrame column."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    plot_col = column
    if log_scale:
        log_col = f'{column}_log'
        gdf = gdf.copy()
        gdf[log_col] = np.log1p(gdf[column])
        plot_col = log_col

    gdf.plot(
        column=plot_col,
        cmap=cmap,
        legend=legend,
        ax=ax,
        edgecolor='black',
        linewidth=0.3,
        missing_kwds={'color': 'lightgrey', 'label': 'No data'},
    )
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_axis_off()
    return ax
