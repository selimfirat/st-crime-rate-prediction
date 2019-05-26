import pandas as pd
import pickle
import os
import numpy as np
import pyproj
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import verde as vd
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from torch import from_numpy
import utils
import torch
from torch import nn, optim
from evaluator import score_r2
import multiprocessing as mp

class Pipeline:
    
    def __init__(self, config=None):
        self.config = config
        
    def init_pipeline(self):
        df = self.load_data()
        utils.init_seeds()
        utils.init_cuda()
        
        
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

    
    def get_y(self):
        
        gdf = self.load_data()
        
        gdf = gdf.sort_index()
        gdf.ts = pd.to_datetime(gdf.ts, errors='coerce', unit="s")
        gdf["ts_index"] = gdf["ts"]
        gdf = gdf.set_index("ts_index").sort_index()
        bins_dt = pd.date_range(start="2015-01-01", end="2017-01-01", normalize=True, freq="D")
        bins_str = bins_dt.astype(str).values
        labels = ['({}, {})'.format(bins_str[i-1], bins_str[i]) for i in range(1, len(bins_str))]
        gdf['timebin'] = pd.cut(gdf.ts.astype(np.int64)//10**9,
                           bins=bins_dt.astype(np.int64)//10**9,
                           labels=labels,
                           right=False
                           )
        assert not np.any(gdf["timebin"].isna().values) # should be empty

        list_neighborhoods = gdf.neigh.unique().tolist()
        list_types = gdf.type.unique().tolist()
        list_timebins = gdf.timebin.unique().tolist()

        neigh2idx = {}
        idx2neigh = {}
        type2idx = {}
        idx2type = {}
        timebin2idx = {}
        idx2timebin = {}

        for idx, neigh in enumerate(list_neighborhoods):
            neigh2idx[neigh] = idx
            idx2neigh[idx] = neigh

        for idx, typ in enumerate(list_types):
            type2idx[typ] = idx
            idx2type[idx] = typ

        for idx, timebin in enumerate(list_timebins):
            timebin2idx[timebin] = idx
            idx2timebin[idx] = timebin

        num_neighs = len(list_neighborhoods)
        num_types = len(list_types)
        num_timebins = len(list_timebins)

        y = np.zeros((num_timebins, num_neighs, num_types), dtype=np.float64)

        groups = gdf.groupby(by=["timebin", "neigh", "type"]).size()
        keys = groups.keys()
        values = groups.values
        num_values = len(values)

        for i in tqdm(range(num_values)):
            timebin, neigh, typ = keys[i]
            cnt = values[i]
            idx_timebin = timebin2idx[timebin]
            idx_neigh = neigh2idx[neigh]
            idx_type = type2idx[typ]
            y[idx_timebin, idx_neigh, idx_type] = cnt
        
        return y
    
    def get_splits_ar_pr(self, prev_inputs=1, n_splits=3):
        y = self.get_y()
        
        num_regions = y.shape[1]
        num_ts = y.shape[0]

        predict_next = 1


        splits = {}
        region_idxs = range(num_regions)
        for region_idx in tqdm(region_idxs):
            if prev_inputs == 1:
                X_data = y[:-1, region_idx, :]
                y_data = y[1:, region_idx, :]
            else:
                X_data = []
                y_data = []
                for x_ts_start in tqdm(range(0, num_ts - predict_next - prev_inputs, convolve_by)):
                    x_ts_end = x_ts_start + prev_inputs
                    y_ts_start = x_ts_end
                    y_ts_end = y_ts_start + predict_next
                    mx = y[x_ts_start:x_ts_end, region_idx, :]
                    my = y[y_ts_start:y_ts_end, region_idx, :]
                    X_data.append(mx)
                    y_data.append(my)

                X_data = np.array(X_data)
                y_data = np.array(y_data)

            splits[region_idx] = []
            reg_splits = TimeSeriesSplit(n_splits=n_splits)
            for split_idx, (train, test) in enumerate(reg_splits.split(X_data, y_data)):
                train_end = int(len(train)*0.9)
                X_train = from_numpy(X_data[train[:train_end]])
                y_train = from_numpy(y_data[train[:train_end]])
                X_val = from_numpy(X_data[train[train_end:]])
                y_val = from_numpy(y_data[train[train_end:]])
                X_test = from_numpy(X_data[test])
                y_test = from_numpy(y_data[test])

                X_train = X_train.float()
                y_train = y_train.float()
                X_val = X_val.float()
                y_val = y_val.float()
                X_test = X_test.float()
                y_test = y_test.float()
                splits[region_idx].append((X_train, y_train, X_val, y_val, X_test, y_test))

        assert len(splits[region_idxs[0]]) == n_splits

        return splits
    
    def model_per_region(self, splits, model_cls, model_params, cudas, num_epochs=5, early_stop_epochs=100, lr=0.001):
        
        
        writer = utils.get_logger(model_cls.__name__)

        split_idxs = range(len(splits[0]))
        region_idxs = splits.keys()

        score_cv = 0.0

        y_test_shape = splits[0][0][5].shape

        for split_idx in split_idxs:
            print("Split #" + str(split_idx))
            
            all_y_test_pred = np.zeros((y_test_shape[0], len(region_idxs), y_test_shape[1]))
            all_y_test = np.zeros((y_test_shape[0], len(region_idxs), y_test_shape[1]))

            for i, region_idx in tqdm(enumerate(region_idxs)):
                model = model_cls(**model_params)
                model = model.float().to()
                criterion = nn.MSELoss(reduction="mean")
                optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

                X_train, y_train, X_val, y_val, X_test, y_test = splits[region_idx][split_idx]

                X_train, y_train, X_val, y_val, X_test, y_test = X_train.cuda(), y_train.cuda(), X_val.cuda(), y_val.cuda(), X_test.cuda(), y_test.cuda()

                early_stop_cur = 0
                max_val_score = -np.inf
                for num_epoch in range(num_epochs):

                    # Forward
                    y_train_pred = model.forward(X_train)

                    loss = criterion(y_train_pred.squeeze(0), y_train)

                    optimizer.zero_grad()
                    model.zero_grad()

                    # Backward
                    loss.backward()

                    # Update model parameters
                    optimizer.step()

                    with torch.no_grad():

                        y_train_pred = y_train_pred.cpu().detach().numpy().squeeze(0)

                        score_train = score_r2(y_train.cpu(), y_train_pred)

                        y_val_pred = model.forward(X_val)

                        y_val_pred = y_val_pred.cpu().detach().numpy().squeeze(0)
                        score_val = score_r2(y_val.cpu(), y_val_pred)

                        writer.add_scalar("regions/region" + str(region_idx) + "/train_mse", loss.item(), num_epoch+1)
                        writer.add_scalar("regions/region" + str(region_idx) + "/train_r2", score_train, num_epoch+1)
                        writer.add_scalar("regions/region" + str(region_idx) + "/val_r2", score_val, num_epoch+1)

                        early_stop_cur += 1

                        if score_val >= max_val_score:
                            early_stop_cur = 0
                            max_val_score = score_val
                        elif early_stop_cur > early_stop_epochs:
                            writer.add_text("Text", "Region " + str(region_idx) + " Early stopping at Epoch " + str(num_epoch + 1), num_epoch+1)
                            break

                with torch.no_grad():

                    test_pred = model.forward(X_test)
                    all_y_test[:, region_idx, :] = y_test.cpu()
                    all_y_test_pred[:, region_idx, :] = test_pred.cpu().detach().numpy().squeeze(0)

            score_test = score_r2(all_y_test, all_y_test_pred)

            writer.add_scalar("splits/split" + str(split_idx + 1) + "/test_r2", score_test)
            print("Test score for split " + str(split_idx  + 1) + " is " + str(score_test))

            score_cv += score_test

        score_cv = score_cv / len(split_idxs)
        print("Mean of cross validation splits' test score is " + str(score_cv))
        writer.add_scalar("cv/test_r2" , score_cv)

    
    def close(self):
        utils.clear_caches()