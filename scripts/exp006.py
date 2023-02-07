# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# # EXP005

# +
from __future__ import annotations

import datetime
import gc
import glob
import itertools
import random
from collections import Counter
from pathlib import Path

import cudf
import gensim
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from loguru import logger
from tqdm import tqdm

print("We will use RAPIDS version", cudf.__version__)

cudf.set_option("default_integer_bitwidth", 32)
cudf.set_option("default_float_bitwidth", 32)

np.random.seed(42)
random.seed(42)

VER = 5
DEBUG = False

# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=False

dbg = logger.debug

ROOT = "."
INPUT_DIR = Path(f"{ROOT}/input")
OUTPUT_DIR = Path(f"{ROOT}/output")


class CFG:
    exp_name = "exp006"

    skip_compute_carts_orders_co_visit: bool = True
    skip_compute_buy2buy_co_visit: bool = True
    skip_compute_clicks_co_visit: bool = True

    skip_to_create_item_features: bool = True
    skip_to_create_user_features: bool = True

    skip_validation: bool = True
    skip_train: bool = True


(OUTPUT_DIR / CFG.exp_name).mkdir(parents=True, exist_ok=True)
# -

# ## Utils

# +


def freemem(df: pd.DataFrame) -> None:
    for col in df.columns:
        if df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
        elif df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
    gc.collect()
    return


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def split_train_valid(targets, valid_df):
    uni_user = targets["session"].unique()
    selected_index = np.random.uniform(1, len(uni_user), len(uni_user) // 2).astype("int32")
    dbg(f"{selected_index = }, {selected_index.shape}")
    uni_user_index = uni_user[selected_index]

    first_half_targets = targets[targets.session.isin(uni_user_index)]
    last_half_targets = targets[~targets.session.isin(uni_user_index)]

    first_half_valid_df = valid_df[valid_df.session.isin(uni_user_index)]
    last_half_valid_df = valid_df[~valid_df.session.isin(uni_user_index)]

    logger.info(f"{first_half_targets.shape = }, {last_half_targets.shape = }")

    assert (
        first_half_targets.session.nunique() == first_half_valid_df.session.nunique()
    ), f"{len(first_half_targets)},{len(first_half_valid_df)}"
    assert (
        last_half_targets.session.nunique() == last_half_valid_df.session.nunique()
    ), f"{len(last_half_targets)},{len(last_half_valid_df)}"

    first_half_targets.to_parquet(f"{INPUT_DIR}/first_half_targets.pqt")
    last_half_targets.to_parquet(f"{INPUT_DIR}/last_half_targets.pqt")

    # valid A + valid B
    first_half_valid_df.to_parquet(f"{INPUT_DIR}/first_half_valid_df.pqt")
    last_half_valid_df.to_parquet(f"{INPUT_DIR}/last_half_valid_df.pqt")


def load_train_targets():
    if CFG.skip_validation:
        target = pd.read_parquet(f"{INPUT_DIR}/otto-train-and-test-data-for-local-validation/test_labels.parquet")
        return target
    else:
        target = pd.read_parquet(f"{OUTPUT_DIR}/first_half_targets.pqt")


# NOTE:
# test_labels.parquetは学習データから直近7日分を抜き出してきたもの
# これを前半と後半に分けて、学習用データと検証用データに分ける
# 最終モデル作成の時は、全部使って学習させる
targets = pd.read_parquet(f"{INPUT_DIR}/otto-train-and-test-data-for-local-validation/test_labels.parquet")
test = pd.read_parquet(f"{INPUT_DIR}/otto-train-and-test-data-for-local-validation/test.parquet")

# これで同じデータであることが確認できる
# test[~test.session.isin(targets.session)]

dbg(f"\n{targets}")
split_train_valid(targets=targets, valid_df=test)

del targets, test
gc.collect()


# +
# %%time
# CACHE FUNCTIONS


# CACHE THE DATA ON CPU BEFORE PROCESSING ON GPU
DATA_CACHE: dict[str, str] = {}


def read_file(f):
    return cudf.DataFrame(DATA_CACHE[f])


def read_file_to_cache(f):
    df = pd.read_parquet(f)
    df.ts = (df.ts / 1000).astype("int32")
    df["type"] = df["type"].map(TYPE_LABELS).astype("int8")
    return df


TYPE_LABELS = {"clicks": 0, "carts": 1, "orders": 2}
files = glob.glob(f"{ROOT}/input/otto-chunk-data-in-parquet-format/*_parquet/*")
for f in files:
    DATA_CACHE[f] = read_file_to_cache(f)

ID2LABEL = {v: k for k, v in TYPE_LABELS.items()}
dbg(f"{ID2LABEL}")

# CHUNK PARAMETERS
READ_CT = 5
CHUNK = int(np.ceil(len(files) / 6))
print(f"We will process {len(files)} files, in groups of {READ_CT} and chunks of {CHUNK}.")
dbg(f"{len(DATA_CACHE)}")

# +
# %%time
type_weight = {0: 1, 1: 6, 2: 3}

# USE SMALLEST DISK_PIECES POSSIBLE WITHOUT MEMORY ERROR
DISK_PIECES = 5
SIZE = 1.86e6 / DISK_PIECES

# COMPUTE IN PARTS FOR MEMORY MANGEMENT
if CFG.skip_compute_carts_orders_co_visit:
    tmp = None
    logger.info(" ###### Skip to compute carts orders co-visitation matrix. ")
else:
    for PART in range(DISK_PIECES):
        print()
        print("### DISK PART", PART + 1)

        # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
        # => OUTER CHUNKSloc[targets["type"] == target_type]
        for j in range(6):
            a = j * CHUNK
            b = min((j + 1) * CHUNK, len(files))
            print(f"Processing files {a} thru {b-1} in groups of {READ_CT}...")

            # => INNER CHUNKS
            for k in range(a, b, READ_CT):
                # READ FILE
                df = [read_file(files[k])]
                for i in range(1, READ_CT):
                    if k + i < b:
                        df.append(read_file(files[k + i]))
                df = cudf.concat(df, ignore_index=True, axis=0)
                df = df.sort_values(["session", "ts"], ascending=[True, False])
                # USE TAIL OF SESSION
                df = df.reset_index(drop=True)
                df["n"] = df.groupby("session").cumcount()
                df = df.loc[df.n < 30].drop("n", axis=1)
                # CREATE PAIRS
                df = df.merge(df, on="session")
                df = df.loc[((df.ts_x - df.ts_y).abs() < 24 * 60 * 60) & (df.aid_x != df.aid_y)]
                # MEMORY MANAGEMENT COMPUTE IN PARTS
                df = df.loc[(df.aid_x >= PART * SIZE) & (df.aid_x < (PART + 1) * SIZE)]
                # ASSIGN WEIGHTS
                df = df[["session", "aid_x", "aid_y", "type_y"]].drop_duplicates(["session", "aid_x", "aid_y"])
                df["wgt"] = df.type_y.map(type_weight)
                df = df[["aid_x", "aid_y", "wgt"]]
                df.wgt = df.wgt.astype("float32")
                df = df.groupby(["aid_x", "aid_y"]).wgt.sum()
                # COMBINE INNER CHUNKS
                if k == a:
                    tmp2 = df
                else:
                    tmp2 = tmp2.add(df, fill_value=0)
                print(k, ", ", end="")

                del df
                gc.collect()

            print()
            # COMBINE OUTER CHUNKS
            if a == 0:
                tmp = tmp2
            else:
                tmp = tmp.add(tmp2, fill_value=0)

            del tmp2
            gc.collect()
        # CONVERT MATRIX TO DICTIONARY
        tmp = tmp.reset_index()
        tmp = tmp.sort_values(["aid_x", "wgt"], ascending=[True, False])
        # SAVE TOP 40
        tmp = tmp.reset_index(drop=True)
        tmp["n"] = tmp.groupby("aid_x").aid_y.cumcount()
        tmp = tmp.loc[tmp.n < 20].drop("n", axis=1)
        # SAVE PART TO DISK (convert to pandas first uses less memory)
        tmp.to_pandas().to_parquet(f"{OUTPUT_DIR}/top_20_carts_orders_v{VER}_{PART}.pqt")

# +
# %%time
# USE SMALLEST DISK_PIECES POSSIBLE WITHOUT MEMORY ERROR
DISK_PIECES = 1
SIZE = 1.86e6 / DISK_PIECES

# COMPUTE IN PARTS FOR MEMORY MANGEMENT
if CFG.skip_compute_buy2buy_co_visit:
    tmp = None
    logger.info(" ###### Skip to compute buy2buy co-visitation matrix.")
else:
    for PART in range(DISK_PIECES):
        print()
        print("### DISK PART", PART + 1)

        # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
        # => OUTER CHUNKS
        for j in range(6):
            a = j * CHUNK
            b = min((j + 1) * CHUNK, len(files))
            print(f"Processing files {a} thru {b-1} in groups of {READ_CT}...")

            # => INNER CHUNKS
            for k in range(a, b, READ_CT):
                # READ FILE
                df = [read_file(files[k])]
                for i in range(1, READ_CT):
                    if k + i < b:
                        df.append(read_file(files[k + i]))
                df = cudf.concat(df, ignore_index=True, axis=0)
                df = df.loc[df["type"].isin([1, 2])]  # ONLY WANT CARTS AND ORDERS
                df = df.sort_values(["session", "ts"], ascending=[True, False])
                # USE TAIL OF SESSION
                df = df.reset_index(drop=True)
                df["n"] = df.groupby("session").cumcount()
                df = df.loc[df.n < 30].drop("n", axis=1)
                # CREATE PAIRS
                df = df.merge(df, on="session")
                df = df.loc[((df.ts_x - df.ts_y).abs() < 14 * 24 * 60 * 60) & (df.aid_x != df.aid_y)]  # 14 DAYS
                # MEMORY MANAGEMENT COMPUTE IN PARTS
                df = df.loc[(df.aid_x >= PART * SIZE) & (df.aid_x < (PART + 1) * SIZE)]
                # ASSIGN WEIGHTS
                df = df[["session", "aid_x", "aid_y", "type_y"]].drop_duplicates(["session", "aid_x", "aid_y"])
                df["wgt"] = 1
                df = df[["aid_x", "aid_y", "wgt"]]
                df.wgt = df.wgt.astype("float32")
                df = df.groupby(["aid_x", "aid_y"]).wgt.sum()
                # COMBINE INNER CHUNKS
                if k == a:
                    tmp2 = df
                else:
                    tmp2 = tmp2.add(df, fill_value=0)
                print(k, ", ", end="")
            print()
            # COMBINE OUTER CHUNKS
            if a == 0:
                tmp = tmp2
            else:
                tmp = tmp.add(tmp2, fill_value=0)
            del tmp2, df
            gc.collect()
        # CONVERT MATRIX TO DICTIONARY
        tmp = tmp.reset_index()
        tmp = tmp.sort_values(["aid_x", "wgt"], ascending=[True, False])
        # SAVE TOP 40
        tmp = tmp.reset_index(drop=True)
        tmp["n"] = tmp.groupby("aid_x").aid_y.cumcount()
        tmp = tmp.loc[tmp.n < 20].drop("n", axis=1)
        # SAVE PART TO DISK (convert to pandas first uses less memory)
        tmp.to_pandas().to_parquet(f"{OUTPUT_DIR}/top_20_buy2buy_v{VER}_{PART}.pqt")

# +
# %%time
# USE SMALLEST DISK_PIECES POSSIBLE WITHOUT MEMORY ERROR
DISK_PIECES = 4
SIZE = 1.86e6 / DISK_PIECES

if CFG.skip_compute_clicks_co_visit:
    tmp = None
    logger.info("########### Skip to compute clicks co-vistation matrix.")
else:
    # COMPUTE IN PARTS FOR MEMORY MANGEMENT
    for PART in range(DISK_PIECES):
        print()
        print("### DISK PART", PART + 1)

        # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
        # => OUTER CHUNKS
        for j in range(6):
            a = j * CHUNK
            b = min((j + 1) * CHUNK, len(files))
            print(f"Processing files {a} thru {b-1} in groups of {READ_CT}...")

            # => INNER CHUNKS
            for k in range(a, b, READ_CT):
                # READ FILE
                df = [read_file(files[k])]
                for i in range(1, READ_CT):
                    if k + i < b:
                        df.append(read_file(files[k + i]))
                df = cudf.concat(df, ignore_index=True, axis=0)
                df = df.sort_values(["session", "ts"], ascending=[True, False])
                # USE TAIL OF SESSION
                df = df.reset_index(drop=True)
                df["n"] = df.groupby("session").cumcount()
                df = df.loc[df.n < 30].drop("n", axis=1)
                # CREATE PAIRS
                df = df.merge(df, on="session")
                df = df.loc[((df.ts_x - df.ts_y).abs() < 24 * 60 * 60) & (df.aid_x != df.aid_y)]
                # MEMORY MANAGEMENT COMPUTE IN PARTS
                df = df.loc[(df.aid_x >= PART * SIZE) & (df.aid_x < (PART + 1) * SIZE)]
                # ASSIGN WEIGHTS
                df = df[["session", "aid_x", "aid_y", "ts_x"]].drop_duplicates(["session", "aid_x", "aid_y"])
                df["wgt"] = 1 + 3 * (df.ts_x - 1659304800) / (1662328791 - 1659304800)
                df = df[["aid_x", "aid_y", "wgt"]]
                df.wgt = df.wgt.astype("float32")
                df = df.groupby(["aid_x", "aid_y"]).wgt.sum()
                # COMBINE INNER CHUNKS
                if k == a:
                    tmp2 = df
                else:
                    tmp2 = tmp2.add(df, fill_value=0)
                print(k, ", ", end="")

                del df
                gc.collect()

            print()
            # COMBINE OUTER CHUNKS
            if a == 0:
                tmp = tmp2
            else:
                tmp = tmp.add(tmp2, fill_value=0)

            del tmp2
            gc.collect()
        # CONVERT MATRIX TO DICTIONARY
        tmp = tmp.reset_index()
        tmp = tmp.sort_values(["aid_x", "wgt"], ascending=[True, False])
        # SAVE TOP 40
        tmp = tmp.reset_index(drop=True)
        tmp["n"] = tmp.groupby("aid_x").aid_y.cumcount()
        tmp = tmp.loc[tmp.n < 20].drop("n", axis=1)
        # SAVE PART TO DISK (convert to pandas first uses less memory)
        tmp.to_pandas().to_parquet(f"{OUTPUT_DIR}/top_20_clicks_v{VER}_{PART}.pqt")
# -

# FREE MEMORY
del DATA_CACHE, tmp
_ = gc.collect()


# +
import polars as pl

# type_weight_multipliers = {'clicks': 1, 'carts': 6, 'orders': 3}
type_weight_multipliers = {0: 1, 1: 6, 2: 3}


def suggest_clicks(df: pl.DataFrame, top_20_clicks, top_20_clicks_set, num: int = 20):
    # display(df)
    session = df["session"].unique()[0]

    # USER HISTORY AIDS AND TYPES
    aids = df["aid"].to_list()
    types = df["type"].to_list()
    unique_aids = list(dict.fromkeys(aids[::-1]))

    # top_20_clicks_set = set(top_20_clicks)
    unique_aids_set = set(unique_aids)

    # RERANK CANDIDATES USING WEIGHTS
    if len(unique_aids) >= num:
        weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]
        sorted_aids = [k for k, v in aids_temp.most_common(20)]
        candidates = pl.DataFrame(
            [
                pl.Series("aid", sorted_aids, dtype=pl.UInt32),
                pl.Series("session", [session for _ in range(len(sorted_aids))], dtype=pl.UInt32),
            ]
        )
        return candidates

    # USE "CLICKS" CO-VISITATION MATRIX
    aids2 = list(itertools.chain(*[top_20_clicks[aid] for aid in unique_aids if aid in top_20_clicks_set]))
    # RERANK CANDIDATES
    aids_most_common = Counter(aids2).most_common(num)
    top_aids2 = [aid2 for aid2, cnt in aids_most_common if aid2 not in unique_aids_set]
    result = unique_aids + top_aids2[: num - len(unique_aids)]
    result = result[:num]
    candidates = pl.DataFrame(
        [
            pl.Series("aid", result, dtype=pl.UInt32),
            pl.Series("session", [session for _ in range(len(result))], dtype=pl.UInt32),
        ]
    )
    return candidates


def suggest_buys(df: pl.DataFrame, top_20_buy2buy, top_20_buys, top_20_buy2buy_set, top_20_buys_set, num: int = 20):
    session = df["session"].unique()[0]

    # USER HISTORY AIDS AND TYPES
    aids = df["aid"].to_list()
    types = df["type"].to_list()
    # UNIQUE AIDS AND UNIQUE BUYS
    unique_aids = list(dict.fromkeys(aids[::-1]))
    df = df.filter((df["type"] == 1) | (df["type"] == 2))
    unique_buys = list(dict.fromkeys(df["aid"].to_list()[::-1]))

    # top_20_buys_set = set(top_20_buys)
    # top_20_buy2buy_set = set(top_20_buy2buy)

    # RERANK CANDIDATES USING WEIGHTS
    if len(unique_aids) >= num:
        weights = np.logspace(0.5, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]
        # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(itertools.chain(*[top_20_buy2buy[aid] for aid in unique_buys if aid in top_20_buy2buy_set]))
        for aid in aids3:
            aids_temp[aid] += 0.1
        aids_temp_most_common = aids_temp.most_common(num)
        sorted_aids = [k for k, v in aids_temp_most_common]
        candidates = pl.DataFrame(
            [
                pl.Series("aid", sorted_aids, dtype=pl.UInt32),
                pl.Series("session", [session for _ in range(len(sorted_aids))], dtype=pl.UInt32),
            ]
        )
        return candidates

    # USE "CART ORDER" CO-VISITATION MATRIX
    selected_unique_aids = unique_aids
    aids2 = list(itertools.chain(*[top_20_buys[aid] for aid in selected_unique_aids if aid in top_20_buys_set]))
    # USE "BUY2BUY" CO-VISITATION MATRIX
    selected_unique_buys = unique_buys
    aids3 = list(itertools.chain(*[top_20_buy2buy[aid] for aid in selected_unique_buys if aid in top_20_buy2buy_set]))

    # RERANK CANDIDATES
    unique_aids_set = set(unique_aids)
    aid2_aid_3_most_common = Counter(aids2 + aids3).most_common(num)
    top_aids2 = [aid2 for aid2, cnt in aid2_aid_3_most_common if aid2 not in unique_aids_set]
    result = unique_aids + top_aids2[: num - len(unique_aids)]
    result = result[:num]
    candidates = pl.DataFrame(
        [
            pl.Series("aid", result, dtype=pl.UInt32),
            pl.Series("session", [session for _ in range(len(result))], dtype=pl.UInt32),
        ]
    )
    return candidates


# +
# -- debug
# top_20_clicks, top_20_buys, top_20_buy2buy = load_co_visitation_matrices()
# candidates = pl.from_pandas(load_train(is_all=False).iloc[10000:10500]).select(["session", "aid", "ts", "type"]).sort(["session", "ts"])

# +
# made_candidates = candidates.groupby("session").apply(
#     lambda x: suggest_clicks(x, top_20_clicks=top_20_clicks, num=20)
# )
# made_candidates
# -

# ## Training

# +
from IPython.display import display


def load_train(is_all: bool) -> pd.DataFrame:
    if is_all:
        dfs = []
        for e, chunk_file in enumerate(glob.glob(f"{ROOT}/input/otto-chunk-data-in-parquet-format/train_parquet/*")):
            chunk = pd.read_parquet(chunk_file)
            chunk["ts"] = (chunk.ts / 1000).astype("int32")
            chunk["type"] = chunk.type.map(TYPE_LABELS).astype("int8")
            dfs.append(chunk)
        return pd.concat(dfs).reset_index(drop=True)
    else:
        train = pd.read_parquet(f"{INPUT_DIR}/otto-train-and-test-data-for-local-validation/train.parquet")
        return train


# NOTE:

# このtrain (3week + 1week(test_labels))に対して特徴量を生成する
# train = load_train(is_all=False)
# print(train.shape)
# display(train)

# -

# ### Item Features

from typing import Union

# +
# -- item features
import polars as pl


def create_item_features(df: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    q = (
        df.lazy()
        .with_column((pl.col("ts").cast(pl.Int64) * 1000).cast(pl.Datetime).dt.with_time_unit("ms").alias("timestamp"))
        .with_column(pl.col("timestamp").cast(pl.Time).alias("time"))
        .with_columns(
            [
                pl.col("time")
                .is_between(datetime.time(0, 0, 0), datetime.time(12, 0, 0))
                .alias("am_activity")
                .cast(pl.UInt8),
                pl.col("time")
                .is_between(datetime.time(12, 0, 0), datetime.time(23, 59, 59))
                .alias("pm_activity")
                .cast(pl.UInt8),
                pl.col("aid").cast(pl.UInt32),
            ]
        )
    )
    df = q.collect()

    # display(df)

    item_features = df.groupby("aid").agg(
        [
            pl.col("aid").count().cast(pl.UInt32).alias("item_item_count"),
            pl.col("session").n_unique().cast(pl.UInt32).alias("item_user_count"),
            pl.col("type").mean().cast(pl.Float32).alias("item_buy_ratio"),
            pl.col("am_activity").mean().cast(pl.Float32).alias("item_am_activity_ratio"),
            pl.col("pm_activity").mean().cast(pl.Float32).alias("item_pm_activity_ratio"),
        ]
    )

    return item_features


if CFG.skip_to_create_user_features:
    logger.info("########### Skip to crate item features")
else:
    first_half_df = pd.read_parquet(f"{INPUT_DIR}/first_half_valid_df.pqt")
    train_df_for_item_feature = pd.concat(
        [pd.read_parquet(f"{INPUT_DIR}/otto-full-optimized-memory-footprint/train.parquet"), first_half_df]
    )
    item_features = create_item_features(df=train_df_for_item_feature)
    item_features.to_pandas().to_parquet(f"{INPUT_DIR}/item_features.pqt")
    display(item_features)

    # del item_features
    # gc.collect()


# -

# ### User Features

# +
# -- user features
def create_user_features(df: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    q = (
        df.lazy()
        .with_column((pl.col("ts").cast(pl.Int64) * 1000).cast(pl.Datetime).dt.with_time_unit("ms").alias("timestamp"))
        .with_column(pl.col("timestamp").cast(pl.Time).alias("time"))
        .with_columns(
            [
                pl.col("session").cast(pl.UInt32),
                pl.col("session").count().over("session").alias("user_session_length").cast(pl.UInt16),
                pl.col("time")
                .is_between(datetime.time(0, 0, 0), datetime.time(12, 0, 0))
                .alias("am_activity")
                .cast(pl.UInt8),
                pl.col("time")
                .is_between(datetime.time(12, 0, 0), datetime.time(23, 59, 59))
                .alias("pm_activity")
                .cast(pl.UInt8),
            ]
        )
    )
    df = q.collect().sort(by=["session", "ts"])
    # display(df)

    first_session_ts = df.groupby("session").first().select(["session", "user_session_length", "ts"]).sort("session")
    last_session_ts = df.groupby("session").last().select(["session", "user_session_length", "ts"]).sort("session")
    session_sec_df = (
        last_session_ts.with_columns(
            [pl.col("session").cast(pl.UInt32), pl.col("ts").cast(pl.Int64).alias("sec_last_session")]
        )
        .with_column(first_session_ts["ts"].cast(pl.Int64).alias("sec_first_session"))
        .with_column((pl.col("sec_last_session") - pl.col("sec_first_session")).alias("sec_to_end_session"))
        .select(["session", "sec_to_end_session", "user_session_length"])
    )

    # display(session_sec_df)

    user_features = df.groupby("session").agg(
        [
            pl.col("session").count().alias("user_user_count").cast(pl.UInt32),
            pl.col("aid").n_unique().alias("user_item_count").cast(pl.UInt32),
            pl.col("type").mean().alias("user_buy_ratio").cast(pl.Float32),
            pl.col("am_activity").mean().alias("user_am_activity_ratio").cast(pl.Float32),
            pl.col("pm_activity").mean().alias("user_pm_activity_ratio").cast(pl.Float32),
        ]
    )

    user_features = user_features.join(session_sec_df, on="session", how="left")

    return user_features


if CFG.skip_to_create_user_features:
    logger.info("######## Skip to create user feature")
else:
    valid_a = pd.read_parquet(f"{INPUT_DIR}/first_half_valid_df.pqt")
    display(valid_a)
    user_features = create_user_features(df=valid_a)
    display(user_features)
    user_features.to_pandas().to_parquet(f"{INPUT_DIR}/user_features.pqt")

    del user_features
    gc.collect()


# +


def create_features(df: pl.DataFrame) -> pl.DataFrame:
    pre_columns = df.columns

    # session毎に何個目のアクションかの逆順
    df = df.with_column(pl.col("session").cumcount().reverse().over("session").alias("action_num_reverse_chrono"))

    # 各sessionのlogの長さ
    df = df.with_column(pl.col("session").count().over("session").alias("session_length").cast(pl.UInt32))

    # logのrecency score
    linear_interpolation = 0.1 + ((1 - 0.1) / (df["session_length"] - 1)) * (
        df["session_length"] - df["action_num_reverse_chrono"] - 1
    )
    df = df.with_column((2**linear_interpolation).alias("log_recency_score").cast(pl.Float32).fill_nan(1.0))

    # type毎に重み付けしたlogのrecency score
    type_weights = {0: 1, 1: 6, 2: 3}
    type_weighted_log_recency_score = (
        df["type"].apply(lambda x: type_weights[TYPE_LABELS[x]] if isinstance(x, str) else type_weights[x])
        * df["log_recency_score"]
    )
    df = df.with_column(type_weighted_log_recency_score.alias("type_weighted_log_recency_score").cast(pl.Float32))
    df = df.select(
        [
            *pre_columns,
            pl.col("log_recency_score"),
            pl.col("action_num_reverse_chrono"),
            pl.col("type_weighted_log_recency_score"),
            pl.col("session_length"),
        ]
    ).with_columns([pl.col("^log_recency_score|type_weighted_log_recency_score$").cumcount().over(["session", "aid"])])

    display(df)

    return df


# -

# ### Create Candidates

# +
from typing import NamedTuple


class CoVisitMat(NamedTuple):
    """topk=20"""

    top_clicks: dict[str, dict[str, int]]
    top_buys: dict[str, dict[str, int]]
    top_buy2buy: dict[str, dict[str, int]]


def pqt_to_dict(df):
    return df.groupby("aid_x").aid_y.apply(list).to_dict()


def load_co_visitation_matrices():
    # LOAD THREE CO-VISITATION MATRICES
    top_20_clicks = pqt_to_dict(pd.read_parquet(f"{OUTPUT_DIR}/top_20_clicks_v{VER}_0.pqt"))
    for k in range(1, DISK_PIECES):
        top_20_clicks.update(pqt_to_dict(pd.read_parquet(f"{OUTPUT_DIR}/top_20_clicks_v{VER}_{k}.pqt")))

    top_20_buys = pqt_to_dict(pd.read_parquet(f"{OUTPUT_DIR}/top_20_carts_orders_v{VER}_0.pqt"))
    for k in range(1, DISK_PIECES):
        top_20_buys.update(pqt_to_dict(pd.read_parquet(f"{OUTPUT_DIR}/top_20_carts_orders_v{VER}_{k}.pqt")))

    top_20_buy2buy = pqt_to_dict(pd.read_parquet(f"{OUTPUT_DIR}/top_20_buy2buy_v{VER}_0.pqt"))

    return CoVisitMat(top_clicks=top_20_clicks, top_buys=top_20_buys, top_buy2buy=top_20_buy2buy)


if not CFG.skip_train:
    top_20_clicks, top_20_buys, top_20_buy2buy = load_co_visitation_matrices()


# +
# %%time

# -- make label
def make_target(target_df: pl.DataFrame, target_type: str) -> pl.DataFrame:
    if target_type not in {"clicks", "carts", "orders"}:
        raise ValueError

    q = (
        target_df.lazy()
        .filter(pl.col("type") == target_type)
        .select(["session", "ground_truth"])
        .explode("ground_truth")
        .rename({"session": "user", "ground_truth": "item"})
        .with_columns(
            [
                pl.col("user").cast(pl.UInt32),
                pl.col("item").cast(pl.UInt32),
                pl.lit(1).alias(target_type).cast(pl.UInt8),
            ]
        )
    )
    targets = q.collect()
    return targets


def sampling_candidates(candidates: Union[pl.DataFrame, pd.DataFrame], label, sampling_rate=0.5) -> pl.DataFrame:
    # -- sampling
    if isinstance(candidates, pd.DataFrame):
        logger.info("type: pd.DataFrame")
        positive = candidates.loc[candidates[label] == 1].sample(frac=sampling_rate, random_state=42)
        # pos:neg = 1:20
        negative_num = 20 * len(positive)
        negative = candidates.loc[candidates[label] == 0].sample(negative_num, random_state=42)
        candidates = pd.concat([positive, negative], axis=0, ignore_index=True)
        logger.info(
            f"{len(positive)=}:{len(negative)} = {len(positive)/len(candidates)=}:{len(negative)/len(candidates)=}"
        )
        return candidates
    else:
        # -- positive user sampling
        np.random.seed(42)
        positive_uni_user = candidates.filter(pl.col(label) == 1)["user"].unique()
        positive_num = (
            min(
                round(sampling_rate * len(candidates.filter(pl.col(label) == 1))),
                len(candidates.filter(pl.col(label) == 1)),
            )
            - 1
        )
        selected_index = np.random.uniform(1, len(positive_uni_user), positive_num).astype("int32") - 1
        positive_user = positive_uni_user[selected_index]
        dbg(f"{positive_user.shape = }")
        positive = candidates.filter((pl.col("type") == TYPE_LABELS[label]) & (pl.col("user").is_in(positive_user)))

        # -- negative user sampling
        np.random.seed(42)
        negative_uni_user = candidates.filter(pl.col(label) == 0)["user"].unique()
        negative_num = min(20 * len(positive), len(negative_uni_user))
        selected_index = np.random.uniform(1, len(negative_uni_user), negative_num).astype("int32") - 1
        negative_user = negative_uni_user[selected_index]
        dbg(f"{negative_user.shape = }")
        negative = candidates.filter((pl.col("type") == TYPE_LABELS[label]) & (pl.col("user").is_in(negative_user)))

        candidates = pl.concat([positive, negative], how="vertical").sort(by="user")
        logger.info(
            f"{len(positive)=}:{len(negative)} = {len(positive)/len(candidates)=}:{len(negative)/len(candidates)=}"
        )
        logger.info(f"{candidates['type'].unique() = }")
        return candidates


def sampling_candidates_by_session(df: pl.DataFrame, sampling_rate: float = 1.0) -> pl.DataFrame:
    uni_user = df["session"].unique()
    select_num = min(round(sampling_rate * len(uni_user)), len(uni_user))
    np.random.seed(42)
    selected_index = np.random.uniform(1, len(uni_user), select_num).astype("int32")
    selected_uni_users = uni_user[selected_index]
    logger.info(f"before select. {df.shape}")
    df = df.filter(pl.col("session").is_in(selected_uni_users))
    logger.info(f"after select. {df.shape}")
    return df


# +
# c = pl.read_parquet(f"{ROOT}/output/candidate_with_features_p0.pqt")
# uni_user = c.filter(pl.col("session") >= 11111111)["session"].unique()
# c.filter(pl.col("session").is_in(uni_user))

# +
def reduce_polars_mem_usage(df):
    for col_name in df.columns:
        if df[col_name].dtype.string_repr() == "i64":
            df = df.with_column(pl.col(col_name).cast(pl.Int32))
        if df[col_name].dtype.string_repr() == "f64":
            df = df.with_column(pl.col(col_name).cast(pl.Float32))
    return df


# d = reduce_polars_mem_usage(c)
# d


# -

# ### Rerank

# +
import xgboost as xgb
from sklearn.model_selection import GroupKFold


def _train_xgb_per_fold(candidates: pl.DataFrame, label: str, fold: str, item_features, df: pl.DataFrame) -> None:
    train_idx = candidates[fold].to_pandas() == 0
    valid_idx = np.where(candidates[fold] == 1)[0]

    # negative_idx = make_negative_idx()
    train_negative_labels = candidates[label].to_pandas().loc[train_idx & (candidates[label].to_pandas() == 0)]
    negative_idx = train_negative_labels.sample(frac=0.1, random_state=42).index.to_numpy()

    train_idx = np.hstack((np.where(train_idx & (candidates[label].to_pandas() == 1))[0], negative_idx))
    train_idx.sort()

    query_train = np.unique(candidates[train_idx, "user"], return_counts=True)[1].astype(np.uint16)
    query_valid = np.unique(candidates[valid_idx, "user"], return_counts=True)[1].astype(np.uint16)

    train_user_features = create_user_features(
        df=df.filter(pl.col("session").is_in(candidates[train_idx]["user"].unique().to_list()))
    )
    train_candidates = make_features(
        candidates=candidates[train_idx], item_features=item_features, user_features=train_user_features
    )
    valid_user_features = create_user_features(
        df=df.filter(pl.col("session").is_in(candidates[valid_idx]["user"].unique().to_list()))
    )
    valid_candidates = make_features(
        candidates=candidates[valid_idx], item_features=item_features, user_features=valid_user_features
    )

    del train_user_features, valid_user_features
    gc.collect()

    feature_cols = train_candidates.columns
    feature_cols = [
        col
        for col in feature_cols
        if col
        not in {
            "index",
            "user",
            "clicks",
            "carts",
            "orders",
            "type",
            "item",
            "session",
            "aid",
            "fold0",
            "fold1",
            "fold2",
            "fold3",
            "fold4",
            "fold5",
        }
    ]
    global FEATURES
    FEATURES = feature_cols
    logger.warning(f"[global] {FEATURES = }")

    dtrain = xgb.DMatrix(
        data=train_candidates[:, FEATURES].to_pandas(),
        label=train_candidates[:, label].to_pandas(),
        group=query_train,
        nthread=-1,
    )
    dvalid = xgb.DMatrix(
        data=valid_candidates[:, FEATURES].to_pandas(),
        label=valid_candidates[:, label].to_pandas(),
        group=query_valid,
        nthread=-1,
    )

    logger.info("train")
    display(train_candidates.select(FEATURES))
    logger.info(f"{train_candidates[label].n_unique() = }")
    logger.info("valid")
    display(valid_candidates.select(FEATURES))
    logger.info(f"{valid_candidates[label].n_unique() = }")

    del query_train, query_valid
    gc.collect()

    xgb_parms = {
        "objective": "rank:pairwise",
        "tree_method": "gpu_hist",
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "booster": "dart",
        "random_state": 42,
    }
    num_boost_round = 100
    model = xgb.train(
        params=xgb_parms,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        num_boost_round=num_boost_round,
        verbose_eval=num_boost_round // 5,
    )
    model.save_model(f"{OUTPUT_DIR}/{CFG.exp_name}/XGB_fold{fold}_{label}.xgb")


def train_model(candidates: pl.DataFrame, label: str, item_features: pl.DataFrame, df: pl.DataFrame) -> None:
    if label not in {"clicks", "carts", "orders"}:
        raise ValueError

    # Reference
    # 1. https://www.kaggle.com/competitions/otto-recommender-system/discussion/379952
    candidates = candidates.sort("user")
    gkf = GroupKFold(n_splits=5)
    for fold, (train_idx, valid_idx) in enumerate(gkf.split(X=candidates, groups=candidates["user"]), start=1):
        candidates = candidates.with_column(pl.lit(0).alias(f"fold{fold}").cast(pl.UInt8))
        candidates[valid_idx, f"fold{fold}"] = 1

    folds = [column for column in candidates.columns if column.startswith("fold")]
    for fold in folds:
        _train_xgb_per_fold(candidates=candidates, label=label, fold=fold, item_features=item_features, df=df)


def create_candidates(
    df: pl.DataFrame,
    top_20_clicks: dict,
    top_20_buy2buy: dict,
    top_20_buys: dict,
    w2vec: gensim.models.Word2Vec,
    clicks_num: int = 20,
    buys_num: int = 20,
) -> pl.DataFrame:

    top_20_clicks_set = set(top_20_clicks)
    top_20_buys_set = set(top_20_buys)
    top_20_buy2buy_set = set(top_20_buy2buy)

    logger.info(f"{df['session'].n_unique() = }")

    # candidates_clicks = df.groupby(["session"]).apply(
    #     lambda x: suggest_clicks(
    #         x,
    #         top_20_clicks=top_20_clicks,
    #         top_20_clicks_set=top_20_clicks_set,
    #         num=20
    #     )
    # )

    candidates_clicks = pl.concat(
        [
            suggest_clicks(x, top_20_clicks=top_20_clicks, top_20_clicks_set=top_20_clicks_set, num=clicks_num)
            for x in tqdm(df.groupby("session"), total=df["session"].n_unique())
        ],
        how="vertical",
    )

    # candidates_buys = df.groupby(["session"]).apply(
    #     lambda x: suggest_buys(
    #         x,
    #         top_20_buy2buy=top_20_buy2buy,
    #         top_20_buys=top_20_buys,
    #         num=20,
    #         top_20_buy2buy_set=top_20_buy2buy_set,
    #         top_20_buys_set=top_20_buys_set
    #     )
    # )
    candidates_buys = pl.concat(
        [
            suggest_buys(
                x,
                top_20_buy2buy=top_20_buy2buy,
                top_20_buys=top_20_buys,
                num=buys_num,
                top_20_buy2buy_set=top_20_buy2buy_set,
                top_20_buys_set=top_20_buys_set,
            )
            for x in tqdm(df.groupby("session"), total=df["session"].n_unique())
        ],
        how="vertical",
    )

    top_k = 5
    top_clicks_candidates = (
        (
            df["session"]
            .unique()
            .to_frame()
            .with_columns(
                [
                    pl.Series(
                        name="aid",
                        values=[df.filter(pl.col("type") == 0)["aid"].value_counts(sort=True)[:top_k]["aid"].to_list()],
                        dtype=pl.List(pl.UInt32),
                    ),
                    pl.col("session").cast(pl.UInt32),
                ]
            )
        )
        .explode("aid")
        .with_column(pl.lit("clicks").alias("type"))
    )
    top_buys_candidates = (
        df["session"]
        .unique()
        .to_frame()
        .with_columns(
            [
                pl.Series(
                    name="aid",
                    values=[
                        df.filter((pl.col("type") == 1) | (pl.col("type") == 2))["aid"]
                        .value_counts(sort=True)[:top_k]["aid"]
                        .to_list()
                    ],
                    dtype=pl.List(pl.UInt32),
                ),
                pl.col("session").cast(pl.UInt32),
            ]
        )
    ).explode("aid")
    top_orders_candidates = top_buys_candidates.clone().with_column(pl.lit("orders").alias("type"))
    top_carts_candidates = top_buys_candidates.clone().with_column(pl.lit("carts").alias("type"))

    del top_buys_candidates

    display(candidates_clicks)
    display(candidates_buys)

    clicks_candidates = candidates_clicks.select(["session", "aid"]).with_column(pl.lit("clicks").alias("type"))
    orders_candidates = candidates_buys.select(["session", "aid"]).clone().with_column(pl.lit("orders").alias("type"))
    carts_candidates = candidates_buys.select(["session", "aid"]).clone().with_column(pl.lit("carts").alias("type"))

    del candidates_clicks, candidates_buys, top_20_clicks, top_20_buys, top_20_buy2buy
    gc.collect()

    def _create_candidates_using_w2vec(w2vec, df):
        session_aids = df.to_pandas().reset_index(drop=True).groupby("session")["aid"].apply(list)
        index = AnnoyIndex(f=32, metric="euclidean")
        aid2idx = {aid: i for i, aid in enumerate(w2vec.wv.index_to_key)}
        labels = []
        for aids in session_aids:
            aids = list(dict.fromkeys(aids[::-1]))
            most_recent_aid = aids[0]
            nns = [w2vec.wv.index_to_key[i] for i in index.get_nns_by_item(aid2idx[most_recent_aid], 11)[1:]]
            labels.append(nns)

        candidates = pl.DataFrame({"session": session_aids.index, "aid": pl.Series("labels", labels)}).explode("labels")
        return candidates

    candidates = pl.concat(
        (
            clicks_candidates,
            orders_candidates,
            carts_candidates,
            top_clicks_candidates,
            top_orders_candidates,
            top_carts_candidates,
            _create_candidates_using_w2vec(w2vec=w2vec, df=df),
        )
    )

    candidates = candidates.with_columns(
        [pl.col("session").cast(pl.UInt32), pl.col("type").apply(lambda x: TYPE_LABELS[x])]
    )
    # candidates["type"] = candidates["type"].apply(lambda x: TYPE_LABELS[x] if isinstance(x, str) else x)
    # candidates = candidates.explode("aid")

    candidates = reduce_polars_mem_usage(candidates)
    return candidates


# features global cache
FEATURES = None
CHUNKS = 10


def save_candidates_with_features(candidates: pl.DataFrame) -> None:
    if not isinstance(candidates, pd.DataFrame):
        raise TypeError

    item_features = pd.read_parquet(f"{INPUT_DIR}/item_features.pqt")
    user_features = pd.read_parquet(f"{INPUT_DIR}/user_features.pqt")

    # candidates = candidates.merge(item_features, left_on="aid", right_index=True, how="left").fillna(-1)
    # candidates = candidates.merge(user_features, left_on="session", right_index=True, how="left").fillna(-1)
    chunk_size = np.ceil(len(candidates) / CHUNKS).astype("int32")
    for k in range(CHUNKS):
        df = candidates.iloc[k * chunk_size : (k + 1) * chunk_size].copy()
        df = df.merge(item_features, left_on="aid", right_index=True, how="left").fillna(-1)
        df = df.merge(user_features, left_on="session", right_index=True, how="left").fillna(-1)
        df.to_parquet(f"{OUTPUT_DIR}/candidate_with_features_p{k}.pqt")


def make_features(
    candidates: pl.DataFrame, item_features: pl.DataFrame | None = None, user_features: pl.DataFrame | None = None
) -> pl.DataFrame:
    if item_features is None:
        item_features = pd.read_parquet(f"{INPUT_DIR}/item_features.pqt")
    if user_features is None:
        user_features = pd.read_parquet(f"{INPUT_DIR}/user_features.pqt")

    # logger.info("item features")
    # display(item_features)
    # logger.info("user features")
    # display(user_features)

    candidates = (
        candidates.join(item_features.rename({"aid": "item"}), on="item", how="left")
        .join(user_features.rename({"session": "user"}), on="user", how="left")
        .fill_null(-1)
    )
    candidates = create_features(candidates.rename({"user": "session"})).rename({"session": "user"})
    return candidates


def make_label(candidates: pl.DataFrame | None = None) -> pl.DataFrame:
    # 実際のデータを読み込んでくる
    train_targets = pl.read_parquet(f"{INPUT_DIR}/otto-train-and-test-data-for-local-validation/test_labels.parquet")
    click_target = make_target(train_targets, target_type="clicks")
    cart_target = make_target(train_targets, target_type="carts")
    order_target = make_target(train_targets, target_type="orders")

    dbg("clicks_target")
    display(click_target)

    # 実際にtargetのeventがあった行は1にそれ以外はNaN=0になっている
    if candidates is None:
        candidates = pl.concat(
            [pl.read_parquet(f"{OUTPUT_DIR}/candidate_with_features_p{k}.pqt") for k in range(CHUNKS)]
        )

    q = (
        candidates.lazy()
        .rename({"session": "user", "aid": "item"})
        .with_columns(
            [
                pl.col("user").cast(pl.UInt32),
                pl.col("item").cast(pl.UInt32),
            ]
        )
        .join(click_target.lazy(), on=["user", "item"], how="left")
        .with_column(pl.col("clicks").fill_null(0))
        .join(cart_target.lazy(), on=["user", "item"], how="left")
        .with_column(pl.col("carts").fill_null(0))
        .join(order_target.lazy(), on=["user", "item"], how="left")
        .with_column(pl.col("orders").fill_null(0))
    )
    candidates = q.collect()
    return candidates


def load_test() -> pd.DataFrame:
    dfs = []
    for e, chunk_file in enumerate(glob.glob(f"{ROOT}/input/otto-chunk-data-in-parquet-format/test_parquet/*")):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts / 1000).astype("int32")
        chunk["type"] = chunk["type"].map(TYPE_LABELS).astype("int8")
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True)  # .astype({"ts": "datetime64[ms]"})


def get_data_for_all_train() -> tuple[pd.DataFrame, pl.DataFrame, pl.DataFrame]:
    # 直近1週間のuser
    sorted_df = pd.read_parquet(f"{INPUT_DIR}/otto-train-and-test-data-for-local-validation/test.parquet").sort_values(
        ["session", "ts"]
    )
    df_for_item_features = pd.concat([load_train(is_all=True).sort_values(["session", "ts"]), load_test()])
    item_features = create_item_features(df=df_for_item_features)
    user_features = create_user_features(df=sorted_df)
    return sorted_df, item_features, user_features


def train(
    train_df: pd.DataFrame | pl.DataFrame | None = None,
    item_features: pl.DataFrame | None = None,
    user_features: pl.DataFrame | None = None,
    sampling_rate: float = 1.0,
    is_all: bool = False,
    use_cache: bool = False,
) -> None:
    def _internal_create_candidates_df(train_df: pl.DataFrame, is_all: bool, use_cache: bool, sampling_rate: float):
        if is_all:
            if train_df is not None:
                logger.warning("is_all=True but, train_df is not None.")
            train_df, item_features, user_features = get_data_for_all_train()

        sorted_df = pl.from_pandas(train_df.sort_values(["session", "ts"]))
        sorted_df = sampling_candidates_by_session(df=sorted_df, sampling_rate=sampling_rate)
        sorted_df = sorted_df.sort(["session", "ts"])

        logger.info("Start to create candidates.")

        def _create_candidates_or_use_cache(
            sorted_df: pl.DataFrame, use_cache: bool, cache_path: Path = Path(f"{ROOT}/output/candidates_cache.pickle")
        ) -> pl.DataFrame:
            top_20_clicks, top_20_buys, top_20_buy2buy = load_co_visitation_matrices()
            w2vec = gensim.models.Word2Vec.load(f"{OUTPUT_DIR}/w2vec_train/w2vec")

            if use_cache:
                logger.warning("########### use Candidates cache ############")
                candidates = pl.from_pandas(pd.read_pickle(cache_path))
                candidates = reduce_polars_mem_usage(candidates)
                candidates = sampling_candidates_by_session(candidates, sampling_rate=sampling_rate)
                logger.info("cached candidates")
                display(candidates)
            else:
                candidates = create_candidates(
                    df=sorted_df,
                    top_20_clicks=top_20_clicks,
                    top_20_buy2buy=top_20_buy2buy,
                    top_20_buys=top_20_buys,
                    clicks_num=20,
                    buys_num=20,
                    w2vec=w2vec,
                )
                candidates.to_pandas().to_pickle(cache_path)
            candidates = candidates.sort(["session"])
            return candidates

        candidates = _create_candidates_or_use_cache(sorted_df=sorted_df, use_cache=use_cache)

        # save_candidates_with_features(candidates)
        # logger.info("Start to make features.")

        logger.info("Start to make labels.")
        candidates = make_label(candidates)
        candidates = reduce_polars_mem_usage(candidates)

        logger.info("candidates")
        display(candidates)

        # feature_cols = candidates.columns[2:-3]
        # feature_cols = [
        #     col
        #     for col in feature_cols
        #     if col not in {"index", "user", "clicks", "carts", "orders", "type", "item", "session", "aid"}
        # ]
        # global FEATURES
        # FEATURES = feature_cols
        # logger.warning(f"[global] {FEATURES = }")

        # clicks_candidates = sampling_candidates(candidates, label="clicks", sampling_rate=1.0)
        # carts_candidates = sampling_candidates(candidates, label="carts", sampling_rate=1.0)
        # orders_candidates = sampling_candidates(candidates, label="orders", sampling_rate=1.0)
        # del candidates
        # gc.collect()
        # return clicks_candidates, carts_candidates, orders_candidates
        candidates = candidates.sort(["user"])
        return candidates, item_features, sorted_df

    # ---------
    # main part
    # ---------
    # clicks_candidates, carts_candidates, orders_candidates = _internal_create_candidates_df(
    #     train_df=train_df, is_all=is_all, use_cache=use_cache, sampling_rate=sampling_rate
    # )
    candidates, item_features, sorted_df = _internal_create_candidates_df(
        train_df=train_df, is_all=is_all, use_cache=use_cache, sampling_rate=sampling_rate
    )

    logger.info("###### Start to train clicks ######")
    train_model(candidates=candidates, label="clicks", item_features=item_features, df=sorted_df)

    logger.info("###### Start to train carts ######")
    train_model(candidates=candidates, label="carts", item_features=item_features, df=sorted_df)

    logger.info("###### Start to train orders ######")
    train_model(candidates=candidates, label="orders", item_features=item_features, df=sorted_df)


if not CFG.skip_train:
    train_df = pd.read_parquet(f"{INPUT_DIR}/first_half_valid_df.pqt")
    train(train_df)


# +
def load_valid() -> pd.DataFrame:
    valid = pd.read_parquet(f"{INPUT_DIR}/last_half_valid_df.pqt")
    return valid


# -


def infer_each_target(test_candidates: pl.DataFrame, target_type: str) -> pd.DataFrame:
    if target_type not in {"clicks", "carts", "orders"}:
        raise ValueError

    def _internal_infer(test_candidates: pl.DataFrame) -> np.ndarray:
        test_candidates = test_candidates.sort("user")
        test_group: np.ndarray = (
            test_candidates.to_pandas().groupby("user").size().to_frame("size")["size"].to_numpy().astype("uint16")
        )
        # test_group = [50] * (len(test_candidates) // 50)

        preds = np.zeros(len(test_candidates), dtype="f4")
        for fold in range(5):
            model = xgb.Booster()
            model.load_model(f"{OUTPUT_DIR}/{CFG.exp_name}/XGB_fold{fold}_{target_type}.xgb")
            model.set_param({"predictor": "gpu_predictor"})
            dtest = xgb.DMatrix(data=test_candidates.to_pandas().loc[:, FEATURES], group=test_group)
            preds += model.predict(dtest) / 5
        return preds

    preds = _internal_infer(test_candidates=test_candidates)
    predictions = test_candidates.select(["user", "item"]).clone()
    predictions = predictions.with_column(pl.Series("score", preds, dtype=pl.Int32))
    del test_candidates, preds
    gc.collect()

    predictions_q = (
        predictions.lazy()
        .sort("score", reverse=True)
        .unique(subset=["user", "item"], keep="last")
        .sort(["user", "score"], reverse=[False, True])
        .groupby("user")
        .agg([pl.col("item").limit(20).cast(pl.Int32).list()])
        .with_column(pl.col("user").cast(pl.Int32))
    )
    predictions = predictions_q.collect()

    logger.info("predictions")
    display(predictions)

    sub = predictions.with_columns(
        [
            pl.col("item").apply(lambda x: " ".join(map(str, x))).alias("labels"),
            (pl.col("user").cast(pl.Utf8) + f"_{target_type}").alias("session_type"),
        ]
    ).select(["session_type", "labels"])
    display(sub)
    return sub


# +


def computed_metric(
    submission_df: pd.DataFrame, gt_df: pd.DataFrame
) -> tuple[float, dict[str, float], dict[str, pd.DataFrame]]:
    logger.info(" ----------- start to computation of metrics ------------ \n")
    if "session_type" not in submission_df.columns or "labels" not in submission_df.columns:
        raise ValueError(f"invalid columns in submission_df: {submission_df.columns}")

    # if "aid" not in gt_df.columns:
    #     raise ValueError(f"invalid columns in gt_df: {gt_df.columns}")
    metrics_per_type = {}
    validation_df_per_type = {}
    score = 0
    weights = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
    for t in weights.keys():
        sub = submission_df.loc[submission_df["session_type"].str.contains(t)].copy()
        sub["session"] = sub["session_type"].apply(lambda x: int(x.split("_")[0]))
        sub["labels"] = sub["labels"].apply(lambda x: [int(i) for i in x.split()[:20]])

        # -- gt label
        gt_df["type"] = gt_df.type.map(lambda x: x if isinstance(x, str) else IDS2LABEL[x])
        test_labels = gt_df.loc[gt_df["type"] == t]

        test_labels = test_labels.merge(sub, how="left", on=["session"])
        dbg("test_labels")
        display(test_labels)

        test_labels["hits"] = test_labels.apply(
            lambda df: len(set(df["ground_truth"]).intersection(set(df["labels"]))), axis=1
        )

        test_labels["gt_count"] = test_labels["ground_truth"].str.len().clip(0, 20)
        recall = test_labels["hits"].sum() / test_labels["gt_count"].sum()
        metrics_per_type[t] = recall
        validation_df_per_type[t] = test_labels
        score += weights[t] * recall
        logger.info(f"{t} recall = {recall}")

    assert 0.0 <= score <= 1.0, f"invalid score: {score}"
    return score, metrics_per_type, validation_df_per_type


# -

# ## Validation

# +
# -- create valid candidates
def valid(valid_df: pl.DataFrame, use_cache: bool) -> None:
    # sorted_df = valid_df.sort_values(["session", "ts"])
    sorted_df = valid_df.sort(["session", "ts"])
    # sorted_df = cudf.from_pandas(sorted_df)

    # pandarallelはメモリ不足でカーネル落ちる
    # 確実にresource開放したい
    logger.info("Start to create candidates.")

    def _create_candidates(
        use_cache: bool, cache_path: Path = Path(f"{ROOT}/output/candidates_cache.pickle")
    ) -> pl.DataFrame:
        top_20_clicks, top_20_buys, top_20_buy2buy = load_co_visitation_matrices()
        if use_cache:
            candidates = pl.from_pandas(pd.read_pickle(cache_path))
            logger.info("########### use Candidates cache")
            display(candidates)
        else:
            candidates = create_candidates(
                df=sorted_df, top_20_clicks=top_20_clicks, top_20_buy2buy=top_20_buy2buy, top_20_buys=top_20_buys
            )
            candidates.to_pandas().to_pickle(cache_path)
        return candidates

    candidates = _create_candidates(use_cache=use_cache)
    display(candidates)
    candidates = candidates.sort(["user"])

    # save_candidates_with_features(candidates)

    # -- 特徴量生成
    logger.info("Start to make features.")

    def _create_item_user_features() -> tuple[pl.DataFrame, pl.DataFrame]:
        first_half_df = pd.read_parquet(f"{OUTPUT_DIR}/first_half_valid_df.pqt")
        train_df = pd.concat([load_train(is_all=False), first_half_df])
        item_features = create_item_features(df=train_df)
        user_features = create_user_features(df=valid_df)
        return item_features, user_features

    item_features, user_features = _create_item_user_features()
    candidates = make_features(candidates=candidates, item_features=item_features, user_features=user_features)
    del item_features, user_features
    gc.collect()

    logger.info("Start to make labels.")
    candidates = make_label(candidates)

    logger.info("candidates")
    display(candidates)

    # feature_cols = candidates.columns[2:-3]
    # global FEATURES
    # FEATURES = feature_cols

    def _make_preds_per_label(label: str) -> pd.DataFrame:
        label_candidates = candidates[candidates["type"] == TYPE_LABELS[label]]
        label_preds = infer_each_target(label_candidates, target_type=label)
        return label_preds

    preds = pd.concat(
        [
            _make_preds_per_label("clicks"),
            _make_preds_per_label("carts"),
            _make_preds_per_label("orders"),
        ]
    )

    # メトリック計算
    valid_gt = pd.read_parquet(f"{INPUT_DIR}/last_half_targets.pqt")
    dbg("valid_gt")
    display("valid_gt")
    score, metrics, oof = computed_metric(submission_df=preds, gt_df=valid_gt)
    logger.info(f"{score = }")
    logger.info(f"{metrics = }")


if not CFG.skip_validation:
    valid(valid_df=pl.read_parquet(f"{INPUT_DIR}/last_half_valid_df.pqt"))
# -

# ## Inference

# ### all training

train(is_all=True, sampling_rate=1.0, use_cache=False)
gc.collect()


# ### Test

# +
# %%time


def test():
    def _test_internal_create_candidates() -> pd.DataFrame:
        test_df = load_test()
        display(test_df)

        top_20_clicks, top_20_buys, top_20_buy2buy = load_co_visitation_matrices()
        print("Here are size of our 3 co-visitation matrices:")
        print(len(top_20_clicks), len(top_20_buy2buy), len(top_20_buys))

        candidates = create_candidates(
            df=pl.from_pandas(test_df),
            top_20_clicks=top_20_clicks,
            top_20_buy2buy=top_20_buy2buy,
            top_20_buys=top_20_buys,
            clicks_num=20,
            buys_num=20,
        )

        df_for_item_features = pd.concat([load_train(is_all=True), test_df])
        item_features = create_item_features(df_for_item_features)
        user_features = create_user_features(test_df)
        candidates = make_features(candidates=candidates, item_features=item_features, user_features=user_features)
        candidates = reduce_mem_usage(candidates.to_pandas())
        return candidates

    candidates = _test_internal_create_candidates()
    display(candidates)

    feature_cols = candidates.columns[2:-3]
    feature_cols = [
        col
        for col in feature_cols
        if col
        not in {
            "index",
            "user",
            "clicks",
            "carts",
            "orders",
            "type",
            "item",
            "session",
            "aid",
            "fold0",
            "fold1",
            "fold2",
            "fold3",
            "fold4",
            "fold5",
        }
    ]
    global FEATURES
    FEATURES = feature_cols
    logger.warning(f"[global] {FEATURES = }")

    TEST_CHUNKS = 3

    def _save_candidates_with_feature_to_pieces(candidates: pd.DataFrame) -> None:
        chunk_size = int(np.ceil(len(candidates.user.unique()) / TEST_CHUNKS))
        users = candidates["user"].unique()
        cnt = 0
        for k in tqdm(range(TEST_CHUNKS), desc="make candidates with features"):
            sessions = users[int(k * chunk_size) : int((k + 1) * chunk_size)]
            df = candidates[candidates["user"].isin(sessions)]
            cnt += len(df)
            df.to_pickle(f"{OUTPUT_DIR}/test_candidate_with_features_p{k}.pickle")

    _save_candidates_with_feature_to_pieces(candidates=candidates)

    def _internal_make_predict_per_piece(k: int) -> pd.DataFrame:
        candidates_with_feature = pl.from_pandas(
            pd.read_pickle(f"{OUTPUT_DIR}/test_candidate_with_features_p{k}.pickle")
        )
        candidates_with_feature = reduce_polars_mem_usage(candidates_with_feature)
        # candidates_with_feature["type"] = candidates_with_feature["type"].astype("uint8")
        display(candidates_with_feature)

        # ----------------
        # predict clicks
        # ----------------
        clicks_candidates = candidates_with_feature.filter(pl.col("type") == TYPE_LABELS["clicks"])
        click_preds = infer_each_target(clicks_candidates, target_type="clicks")
        del clicks_candidates
        gc.collect()

        # ----------------
        # predict carts
        # ----------------
        carts_candidates = candidates_with_feature.filter(pl.col("type") == TYPE_LABELS["carts"])
        carts_preds = infer_each_target(carts_candidates, target_type="carts")
        del carts_candidates
        gc.collect()

        # ----------------
        # predict orders
        # ----------------
        orders_candidates = candidates_with_feature.filter(pl.col("type") == TYPE_LABELS["orders"])
        orders_preds = infer_each_target(orders_candidates, target_type="orders")
        del orders_candidates
        gc.collect()

        sub = pl.concat([click_preds, carts_preds, orders_preds])
        return sub

    final_sub = pl.concat([_internal_make_predict_per_piece(k=k) for k in tqdm(range(TEST_CHUNKS))], how="vertical")
    final_sub.write_csv(f"{OUTPUT_DIR}/{CFG.exp_name}/submission.csv")
    logger.info(f"sub \n {final_sub}")
    display(final_sub)
    logger.info(f"{final_sub.shape = }")


# -

test()
