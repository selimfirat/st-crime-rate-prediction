import pandas as pd
import pickle
import os
import numpy as np
from models.dummy_predictors import RandomPredictor, ZerosPredictor, OnesPredictor, AveragePredictor
from evaluator import Evaluator
from models.svr import SVRPredictor
import pyproj
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import verde as vd

class Pipeline:
    
    def __init__(self, config):
        self.config = config
        
    def init_pipeline(self):
        df = self.load_data()
        
        # df = df.round(2)
        
        print("Unique locations", np.unique(df[["lat", "long"]], axis=0).shape)
        print("Min latitude", df["lat"].min())
        print("Max latitude", df["lat"].max())
        print("Min longitude", df["long"].min())
        print("Max longitude", df["long"].max())
        print("Min ts", df["ts"].min())
        print("Max ts", df["ts"].max())
        print("Number of days", (df["ts"].max() - df["ts"].min())//(60*60*24))
        
        projection = pyproj.Proj(proj="merc", lat_ts=df.lat.mean())

        # pyproj doesn't play well with Pandas so we need to convert to numpy arrays
        proj_coords = projection(df.long.values, df.lat.values)
        print(proj_coords)

        spacing = 10 / 60
        reducer = vd.BlockReduce(np.median, spacing=spacing * 111e3)
        filter_coords, filter_bathy = reducer.filter(proj_coords, df.type)
        spline = vd.Spline().fit(filter_coords, filter_bathy)

        region = vd.get_region((df.long, df.lat))
        print("Data region in degrees:", region)
        # Specify the region and spacing in degrees and a projection function
        grid_geo = spline.grid(
            region=region,
            spacing=spacing,
            projection=projection,
            dims=["lat", "long"],
            data_names=["type"],
        )
        
        print("Geographic grid:")
        print(grid_geo)


        
    
    def load_data(self):

        pkl_path = "data/" + self.config.data + ".pkl"
        if os.path.exists(pkl_path):
            df = pickle.load(open(pkl_path, "rb"))
            return df
        
        if self.config.data == "chicago":
            df = pd.concat([
                # pd.read_csv('data/chicago/Chicago_Crimes_2001_to_2004.csv',error_bad_lines=False),
                # pd.read_csv('data/chicago/Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False),
                # pd.read_csv('data/chicago/Chicago_Crimes_2008_to_2011.csv',error_bad_lines=False),
                pd.read_csv('data/chicago/Chicago_Crimes_2012_to_2017.csv',error_bad_lines=False)
            ], ignore_index=False, axis=0)
            df.drop_duplicates(subset=["ID"], keep='first', inplace=True)
            
            df = df[["Date", "Latitude", "Longitude", "Primary Type"]]
            df.columns = ["ts", "lat", "long", "type"]
            df = df.dropna()
            df.ts = pd.to_datetime(df.ts, format='%m/%d/%Y %I:%M:%S %p').astype(np.int64) // 10**9
            df = df.loc[df["ts"] >= self.config.min_ts]
            df = df.loc[df["ts"] <= self.config.max_ts]
            
            df["lat"] = df["lat"].apply(lambda x: float(x))
            df["long"] = df["long"].apply(lambda x: float(x))


        pickle.dump(df, open(pkl_path, "wb"))
        
        return df