import requests
import pandas as pd
import yaml
import numpy as np
from dask import delayed, compute
from dask.distributed import LocalCluster, Client
import os

def load(path):
    df = pd.read_parquet(path)
    X = df.iloc[:, 1:]
    y = df.iloc[:, :1]

    # set continuous feat missing values to median
    X = X.fillna(X.median(numeric_only=True))

    # One-hot encode categorical features
    X = pd.get_dummies(X)

    # Convert X to numpy array
    X = X.values.astype('float64')

    # One-hot encode y
    y = pd.get_dummies(y.astype('category')).values.astype('float64')
    return X, y

def process_simulated_datasets(in_path, out_path):
    # Read in source data
    dfs = {
        'orthant_train' : pd.read_csv(f"{in_path}Orthant_train.csv", sep=",", header=None),
        'orthant_test' : pd.read_csv(f"{in_path}Orthant_test.csv", sep=",", header=None),
        'trunk_train' : pd.read_csv(f"{in_path}Trunk_train.csv", sep=",", header=None),
        'trunk_test' : pd.read_csv(f"{in_path}Trunk_train.csv", sep=",", header=None),
        'sparse_parity_train' : pd.read_csv(f"{in_path}Sparse_parity_train.csv", sep=",", header=None),
        'sparse_parity_test' : pd.read_csv(f"{in_path}Sparse_parity_test.csv", sep=",", header=None)
    }

    for name, df in dfs.items():
        # Label cols
        df.columns = [f"x{i}" for i in range(len(df.columns) - 1)] + ['y']

        # Reorder data
        df = df[['y'] + df.drop(['y'], axis=1).columns.to_list()]

        # Cast dtypes
        df = df.astype('float32').astype({'y' : 'int'}).astype({'y' : 'category'})

        # Save in parquet format
        df.to_parquet(f"{out_path}{name}.parquet", engine='pyarrow', compression='snappy')

def save_uci_datasets(yaml_path, out_path):
    # Read in the data
    meta = parse_dataset_metadata(yaml_path)

    @delayed
    def process_data(name, out_path, **kwargs):
        # Don't process the data if it already exists
        if os.path.exists(f"{out_path}/{name}.parquet"):
            return None

        # Get the dataset
        df = get_dataset(**kwargs)

        # Write the data locally along with it's metadata to read later
        df.to_parquet(f"{out_path}/{name}.parquet", engine='pyarrow', compression='snappy')

    tasks = [process_data(name=name, out_path=out_path, **info) for name, info in meta.items()]
    _ = compute(tasks)

def parse_dataset_metadata(path):
    # Read yaml
    dfs = yaml.safe_load(open(path, 'rb'))

    # Structure dictionary so we can read this data in later
    out = {}
    for df, meta in dfs.items():
        out[df] = {
            'url' : meta['url'],
            'target' : meta['target'],
            'missing_ind' : meta['missing_ind'],
            'sep' : meta['sep'],
            'drop' : meta['drop'],
            'columns' : list(meta['columns'].keys()),
            'dtype' : meta['columns']
        }

    return out

def get_dataset(url, columns, dtype, missing_ind, target, sep, drop):
    if os.path.exists(url):
        data = open("source_data/shuttle.trn", "r").read()
    else:
        # Grab data from web
        data = requests.get(url).content.decode('utf-8')

    # Decode the data, split on rows, remove leading/trailing whitespace, and then finally split on cols
    rows = [l.strip().split(sep) for l in data.split('\n') if l.strip() != ""]

    # Convert to a pandas dataframe
    dtype_mapping = {
        'ordinal' : 'float32',
        'continuous' : 'float32',
        "category" : "category"
    }
    # Validate we only have valid input types
    input_dtypes = set(dtype.values())
    bad_dtypes = input_dtypes - dtype_mapping.keys()
    if len(bad_dtypes) > 0:
        raise ValueError(f"Bad dtypes found in schema: {bad_dtypes}")
    pd_dtypes = {col : dtype_mapping[t] for col, t in dtype.items()}
    
    # Convert to a pandas dataframe with appropriate schema
    df = pd.DataFrame(rows, columns = columns) \
    .replace(missing_ind, np.nan) \
    .astype(pd_dtypes)

    # Check if any numeric features have zero variation and drop
    variation = df.var(numeric_only=True)
    zero_var = variation[variation == 0].index.to_list()
    drop = set(drop).union(set(zero_var))

    # Check if any categorical cols have only one category
    const_cat = [col for col in df.columns.to_list() if df[col].dtype == 'category']
    const_cat = {col for col in const_cat if len(df[col].cat.categories) == 1}
    drop = drop.union(const_cat)

    # Move the target variable to the first col
    oth_cols = list(set(df.columns.to_list()) - {target} - drop)
    df = df[[target] + oth_cols]
    
    return df

if __name__ == '__main__':
    # Set up a dask cluster so we can process in parallel
    cluster = LocalCluster()
    client = Client(cluster)

    # Process all datasets given the expected yaml file
    save_uci_datasets("datasets.yaml", "data")
    process_simulated_datasets("source_data/simulated/", "data/")