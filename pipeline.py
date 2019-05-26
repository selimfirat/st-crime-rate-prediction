import pandas as pd
import pickle
import os
import numpy as np
import pyproj
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import verde as vd

class Pipeline:
    
    def __init__(self, config=None):
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
        

        
    
    def load_data(self):

        pkl_path = "data/chicago.pkl"
        if os.path.exists(pkl_path):
            df = pickle.load(open(pkl_path, "rb"))
            return df

        df = pd.concat([
            # pd.read_csv('data/chicago/Chicago_Crimes_2001_to_2004.csv',error_bad_lines=False),
            # pd.read_csv('data/chicago/Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False),
            # pd.read_csv('data/chicago/Chicago_Crimes_2008_to_2011.csv',error_bad_lines=False),
            pd.read_csv('data/chicago/Chicago_Crimes_2012_to_2017.csv',error_bad_lines=False)
        ], ignore_index=False, axis=0)
        df.drop_duplicates(subset=["ID"], keep='first', inplace=True)

        print("After drop_duplicates", df.shape)
        df = df[["Date", "Latitude", "Longitude", "Primary Type"]]
        df.columns = ["ts", "lat", "long", "type"]
        df = df.dropna()
        df.ts = pd.to_datetime(df.ts, format='%m/%d/%Y %I:%M:%S %p').astype(np.int64) // 10**9
        print("After dropna", df.shape)
        df = df.loc[df["ts"] >= 1420070400]
        df = df.loc[df["ts"] <= 1483228800]

        df["lat"] = df["lat"].apply(lambda x: float(x))
        df["long"] = df["long"].apply(lambda x: float(x))

        print("After minmax ts clip", df.shape)
        chicago = gpd.read_file("data/chicago/city/geo_export_41e84a0e-a158-44f2-abe6-a52f78662352.shp")
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long, df.lat))
        chicago_mask = gdf.geometry.within(chicago.loc[0, 'geometry'])
        chicago_gdf = gdf[chicago_mask]
        print("After chicago clip", chicago_gdf.shape)

        neighborhoods = gpd.read_file("data/chicago/neighborhoods/geo_export_265e79d9-e1f1-47fc-856d-1c753cbbe120.shp")
        neigh_gdf = gpd.sjoin(chicago_gdf, neighborhoods, how="inner", op='within')
        neigh_gdf = neigh_gdf[["ts", "lat", "long", "pri_neigh", "type"]]
        neigh_gdf.columns = ["ts", "lat", "long", "neigh", "type"]

        print("After neighborhood join", neigh_gdf.shape)


        major_types_series = (neigh_gdf.groupby("type").size() > 100)
        major_types = []
        for typ, val in major_types_series.iteritems():
            if val:
                major_types.append(typ)

        neigh_gdf = neigh_gdf[neigh_gdf["type"].isin(major_types)]

        print("After important events clip", neigh_gdf.shape)

        pickle.dump(neigh_gdf, open(pkl_path, "wb"))


        return neigh_gdf
