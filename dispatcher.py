import argparse
from pipeline import Pipeline
import math

class Dispatcher:
    
    def __init__(self):
        
        config = self.parse_args()
        
        self.run_pipeline(config)
    
    def parse_args(self):
        parser = argparse.ArgumentParser(description='Crime Prediction Framework')
        # parser.add_argument("mode", type=str, default="extract_features")
        parser.add_argument("--data", type=str, default="chicago")
        parser.add_argument("--day_bins", type=int, default=2)
        parser.add_argument("--lat_bins", type=int, default=16)
        parser.add_argument("--long_bins", type=int, default=16)
        parser.add_argument("--min_ts", type=int, default=-math.inf) # 1420070400
        parser.add_argument("--max_ts", type=int, default=math.inf)
        args = parser.parse_args()

        return args

    def run_pipeline(self, config):
        pipeline = Pipeline(config)
        
        pipeline.init_pipeline()