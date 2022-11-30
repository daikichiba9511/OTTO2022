# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Co-visitation matrix - simplified, impoved logic
#
# - [1]のNotebookを参考. baseは[2]でそれに[3]でいわれてるtest dataのleakと新しいロジックを追加したもの
# - datasetは最適化したものを使ってる[4]
# - validationの追加[5]
# - exp001をベースにReference[1]の新しいバージョンのロジックを追加する
#   - co-visitation matrixを求めるときにconcatしてから求める
#   - aidのtypeによっての重み付け
#
# ## Reference
#
# 1. [co-visitation matrix - simplified, imprvd logic 🔥, @radek1](https://www.kaggle.com/code/radek1/co-visitation-matrix-simplified-imprvd-logic)
#
# 2. [Co-visitation Matrix, @vslaykovsky](https://www.kaggle.com/code/vslaykovsky/co-visitation-matrix)
#
# 3. [Test Data Leak - LB Boost, @cdeotte](https://www.kaggle.com/code/cdeotte/test-data-leak-lb-boost)
#
# 4. [Full dataset processed to CSV/parquet files with optimized memory footprint](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843)
#
# 5. [local validation tracks public LB perfecty -- here is the setup](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991)
#
# 6. [Compute Validation Score - [CV 563] ](https://www.kaggle.com/code/cdeotte/compute-validation-score-cv-563)


# %%
from __future__ import annotations

import csv
import gc
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Final

import cudf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beartype import beartype
from cudf.core.dataframe import itertools
from loguru import logger
from tqdm import tqdm

tqdm.pandas()


class UtilsModule:
    @staticmethod
    def timeit(func):
        """
        Reference
        1. https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
        """

        @wraps(func)
        def timeit_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            logger.info(f"Function [ {func.__name__} ] Took {total_time:.4f} seconds")
            return result

        return timeit_wrapper


@dataclass(frozen=True)
class Config:
    """
    Args:
        exp_name:
        debug: debug機能をonにするか, 起動時に'python -O <file path>' でFalse
        seed:
        do_validation: validation関数を実行するか
        do_re_compute: 共起行列を再度計算するか, True => する(do) False => しない(do not)
    """

    exp_name: str = Path(__file__).stem
    debug: bool = __debug__ if __debug__ is not None else True
    seed: int = 42
    do_validation: bool = True
    do_re_compute: bool = True


@dataclass(frozen=True)
class Data:
    """
    Args:
        train_df:
        valid_df:
        test_df:
        test_labels:
        id2type:
        type2id:
        sample_sub:
    """

    train_df: pd.DataFrame | None
    valid_df: pd.DataFrame | None
    test_df: pd.DataFrame | None
    test_labels: pd.DataFrame | None
    id2type: dict[int, str]
    type2id: dict[str, int]
    sample_sub: pd.DataFrame


LABEL2IDS = {
    "clicks": 0,
    "carts": 1,
    "orders": 2,
}

IDS2LABEL = {
    0: "clicks",
    1: "carts",
    2: "orders",
}

# this cache variable is used to reduce file I/O in DataModule._read_file
DATA_CACHE = {}


class DataModule:

    ver: Final[int] = 5

    @staticmethod
    @beartype
    def save_data_to_global_cache(files: list[Path]) -> None:
        """load data from parquet file and then save to global cache variable: DATA_CACHE"""
        logger.warning(" ####### WARNING: save to global cache to reduce file I/O")
        for file_path in files:
            df = pd.read_parquet(file_path)
            DATA_CACHE[file_path] = df

    @staticmethod
    @beartype
    def _read_file(file_path: Path, use_data_cache: bool = True) -> cudf.DataFrame:
        if use_data_cache and len(DATA_CACHE) > 0:
            df = DATA_CACHE.get(file_path)
            if df is None:
                logger.info(f"missed cache: {file_path = }")
                df = pd.read_parquet(file_path)
            df = cudf.DataFrame(df)
            logger.warning(" ####### WARNING: using global cache for dataframe.")
        else:
            df = cudf.read_parquet(file_path)
        if df is None:
            raise ValueError

        df["ts"] = (df["ts"] / 1000).astype("int32")
        # df["session"] = df["session"].astype("int32")
        # df["aid"] = df["aid"].astype("int32")
        df["type"] = df["type"].map(LABEL2IDS).astype("int8")
        return df

    @staticmethod
    @beartype
    def _load_train_for_validation(root: Path = Path("./")) -> pd.DataFrame:
        train_dfs = []
        for e, chunk_file in enumerate(Path(f"{root}/input/otto-validation/train_parquet").glob("*")):
            chunk = pd.read_parquet(chunk_file)
            chunk["type"] = chunk["type"].map(lambda x: LABEL2IDS[x])
            train_dfs.append(chunk)
        # return pd.concat(train_dfs).reset_index(drop=True).astype({"ts": "datetime64[ms]"})
        return pd.concat(train_dfs).reset_index(drop=True)

    @staticmethod
    @beartype
    def _load_valid(root: Path = Path("./")) -> pd.DataFrame:
        valid_dfs = []
        for e, chunk_file in enumerate(Path(f"{root}/input/otto-validation/test_parquet").glob("*")):
            chunk = pd.read_parquet(chunk_file)
            chunk["type"] = chunk["type"].map(lambda x: LABEL2IDS[x])
            valid_dfs.append(chunk)
        # return pd.concat(valid_dfs).reset_index(drop=True).astype({"ts": "datetime64[ms]"})
        return pd.concat(valid_dfs).reset_index(drop=True)

    @staticmethod
    @beartype
    def _load_test(root: Path = Path("./")) -> pd.DataFrame:
        # this 'ts' data was devided by 1000
        # test = pd.read_parquet(f"{root}/input/otto-full-optimized-memory-footprint/test.parquet")
        # test = test.reset_index(drop=True)
        # test["ts"] = test["ts"] / 1000
        # test["type"] = test["type"].map(lambda x: IDS2LABEL[x])
        # test = test.astype({"ts": "datetime64[ms]", "type": "int64"})
        # test = test.astype({"type": "int64"})
        dfs = []
        for e, chunk_file in enumerate(
            Path(f"{root}/input/otto-chunk-data-in-parquet-format/test_parquet").glob("*.parquet")
        ):
            chunk = pd.read_parquet(chunk_file)
            chunk["ts"] = (chunk["ts"] / 1000).astype("int32")
            chunk["type"] = chunk["type"].map(lambda x: LABEL2IDS[x]).astype("int8")
            dfs.append(chunk)

        test = pd.concat(dfs).reset_index(drop=True)
        return test

    @staticmethod
    @beartype
    def _load_test_labels(root: Path = Path("./")) -> pd.DataFrame:
        """validationの評価用のdata"""
        test_labels = pd.read_parquet(f"{root}/input/otto-validation/test_labels.parquet")
        return test_labels

    @staticmethod
    @beartype
    def _pqt_to_dict(df: pd.DataFrame) -> dict:
        return df.groupby("aid_x")["aid_y"].apply(list).to_dict()

    @staticmethod
    @beartype
    def load_co_visitation_matrix_parquet(
        ver: int, root: Path = Path("./"), click_dick_pieces: int = 4, buys_disk_pieces: int = 4
    ) -> dict[str, dict[str, np.ndarray]]:
        logger.info(" ######### loading co-visitation matrix... ########### ")
        logger.info(f" --- {ver = }, {click_dick_pieces = }")

        # -- load clicks co-visitation-matrix
        top_20_click = DataModule._pqt_to_dict(pd.read_parquet(f"{root}/output/top_20_clicks_0_ver{ver}.parquet"))
        for click_part in range(1, click_dick_pieces):
            top_20_click.update(
                DataModule._pqt_to_dict(
                    pd.read_parquet(f"{root}/output/top_15_carts_orders_{click_part}_ver{ver}.parquet")
                )
            )

        # -- load buy2buy co-visitation-matrix
        top_15_buys = DataModule._pqt_to_dict(pd.read_parquet(f"{root}/output/top_15_carts_orders_0_ver{ver}.parquet"))
        for buys_part in range(1, buys_disk_pieces):
            top_15_buys.update(
                DataModule._pqt_to_dict(
                    pd.read_parquet(f"{root}/output/top_15_carts_orders_{buys_part}_ver{ver}.parquet")
                )
            )

        # -- load buy2buy co-visitation-matrix
        top_15_buy2buy = DataModule._pqt_to_dict(pd.read_parquet(f"{root}/output/top_15_buy2buy_0_ver{ver}.parquet"))

        return {"top20_clicks": top_20_click, "top15_buys": top_15_buys, "top15_buy2buy": top_15_buy2buy}

    @staticmethod
    @beartype
    def load_data(root: Path = Path("./"), debug: bool = False, is_for_valid: bool = False) -> Data:
        # dataset only for validation from https://www.kaggle.com/datasets/radek1/otto-train-and-test-data-for-local-validation
        # 'ts' columns data were devided by 1000, because the creator thinks 'ts' columns is enough orders of seconds.
        # So, If you need, must restore 'ts' columns data by multiplying 1000
        # Ref: 5.
        # TODO: when I make final submit, change this directory to original dataset / otto-full-optimized-memory-footprint

        # train = pd.read_parquet(f"{root}/input/otto-validation/train.parquet")  # type: ignore
        # valid = pd.read_parquet(f"{root}/input/otto-validation/test.parquet")

        train = None
        valid = None
        valid_labels = None

        if is_for_valid:
            train = DataModule._load_train_for_validation()
            valid = DataModule._load_valid()
            valid_labels = DataModule._load_test_labels()

        test = DataModule._load_test()

        sample_sub = pd.read_csv(f"{root}/input/otto-recommender-system/sample_submission.csv")

        assert train is None or isinstance(train, pd.DataFrame)
        assert test is None or isinstance(test, pd.DataFrame)
        return Data(
            train_df=train,
            valid_df=valid,
            test_labels=valid_labels,
            test_df=test,
            id2type=IDS2LABEL,
            type2id=LABEL2IDS,
            sample_sub=sample_sub,
        )


# %%


# %%
class PreprocessModule:
    @staticmethod
    @beartype
    def optimize_train_df(train: pd.DataFrame) -> None:
        train.loc[:, "session"] = train.loc[:, "session"].astype(np.int32)
        train.loc[:, "aid"] = train.loc[:, "aid"].astype(np.int32)
        # train.loc[:, "ts"] = train.loc[:, "ts"].astype(np.int32)

    @staticmethod
    @beartype
    def _sampled_df_by_session(df: pd.DataFrame, sampling_ratio: float, seed: int = 42) -> pd.DataFrame:
        if sampling_ratio < 0.0 or sampling_ratio > 1.0:
            raise ValueError(f"Invalid Value: {sampling_ratio}")

        dropped_duplicated_df = df.drop_duplicates(["session"])
        if dropped_duplicated_df is None:
            return df

        remained_df_session = dropped_duplicated_df.sample(frac=sampling_ratio, random_state=seed)["session"]
        sub_df = df.loc[df.loc[:, "session"].isin(remained_df_session), :]
        return sub_df

    @staticmethod
    @beartype
    def preprocess(df: pd.DataFrame, fraction_of_sessions_to_use: float) -> pd.DataFrame:
        if fraction_of_sessions_to_use < 0.0 or fraction_of_sessions_to_use > 1.0:
            raise ValueError("invalid range")

        subset_of_df = PreprocessModule._sampled_df_by_session(df=df, sampling_ratio=fraction_of_sessions_to_use)
        subset_of_df.index = pd.MultiIndex.from_frame(subset_of_df[["session"]])
        return subset_of_df


class GenerateModule:
    @staticmethod
    @UtilsModule.timeit
    @beartype
    def make_covisitation_matrix(subsets: pd.DataFrame, chunk_size: int = 30_000) -> defaultdict[int, Counter]:
        """compute co-visitation matrix

        Args:
            subsets: subsets dataframe which has multiple index of 'sessions'
            chunk_size:

        Return:
            next_aids: co-visitation count dict {'aid_x': {'aid_y': count}}
        """

        print(" ------ make co-visitation matrix -------- ")
        # {"aid_x": {"aid_c": count}}
        next_aids = defaultdict(Counter)

        # NOTE:
        # only 'test' is not same data source, so 'ts' column was devided by 1000. so, you need to multiply 1000
        # <-> but, you don't need to do train and valid
        # Referrence
        # 1. https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991#2023514
        # 2. https://www.kaggle.com/code/radek1/a-robust-local-validation-framework
        #
        # Procedure.
        # 1. chunkにわける
        # 2. session毎に最後の30個の履歴を最後尾から取ってきてindexをreset
        # 3. それらをsession毎にmerge
        # 4. aid_x != aid_yのものを取ってくる
        # 5. ts_yとts_xが0日以上1日以下の履歴を抽出する

        uni_sessions = subsets["session"].unique()
        print(f"{len(uni_sessions) = }")
        for i in tqdm(range(0, len(uni_sessions), chunk_size)):
            current_chunk = subsets.loc[
                uni_sessions[i] : uni_sessions[min(len(uni_sessions) - 1, i + chunk_size - 1)]
            ].reset_index(drop=True)
            current_chunk = (
                current_chunk.groupby("session", as_index=False).nth(list(range(-30, 0))).reset_index(drop=True)
            )
            consecuitive_aids = current_chunk.merge(current_chunk, on="session")
            consecuitive_aids = consecuitive_aids[consecuitive_aids["aid_x"] != consecuitive_aids["aid_y"]]
            # consecuitive_aids["days_elapsed"] = (consecuitive_aids["ts_y"] - consecuitive_aids["ts_x"]) / (24 * 60 * 60)
            consecuitive_aids = consecuitive_aids[
                (consecuitive_aids["ts_y"] - consecuitive_aids["ts_x"]).abs() <= 24 * 60 * 60 * 1000
            ]
            for aid_x, aid_y in zip(consecuitive_aids["aid_x"], consecuitive_aids["aid_y"]):
                next_aids[aid_x][aid_y] += 1

        return next_aids

    @staticmethod
    @beartype
    def _comupute_carts_orders_co_visitation_matrix(
        files: list[Path], root: Path = Path("./"), use_data_cache: bool = True
    ) -> None:
        # train = pd.read_csv(f"{root}/input/otto-full-optimized-memory-footprint/train.parquet")
        # return train
        chunk = int(np.ceil(len(files)))
        read_ct = 5
        type_weight = {0: 1, 1: 6, 2: 3}

        disk_pieces = 4
        size = 1.86e6 / disk_pieces

        for part in range(disk_pieces):
            print(f" --- {part = }")
            for j in range(6):
                start = j * chunk
                end = min((j + 1) * chunk, len(files))
                for k in tqdm(range(start, end, read_ct)):
                    df = [DataModule._read_file(files[k], use_data_cache=use_data_cache)]
                    for i in range(1, read_ct):
                        if k + i >= end:
                            continue
                        df.append(DataModule._read_file(files[k + i], use_data_cache=use_data_cache))
                    df = cudf.concat(df, ignore_index=True, axis=0)
                    df = df.sort_values(["session", "ts"], ascending=[True, False])

                    # -- use tail of session
                    df = df.reset_index(drop=True)
                    df["n"] = df.groupby("session").cumcount()
                    df = df.loc[df["n"] < 30].drop("n", axis=1)

                    # -- create pairs
                    # sessionで同一dataframeをinner joinすることでペアを作る
                    # {ts_x1, ts_x2, ts_x3, ... ts_xn}, {ts_y1, ts_y2, ts_y3, ..., ts_yn}
                    # => (ts_x1, ts_y2), (ts_x1, ts_y3), (ts_x1, ts_y4), ...
                    # 同一session(=同一user)内で、ts_xiのイベントが起きた時に起こった他のイベントのペアを作成してる
                    df = df.merge(df, on="session")
                    df = df.loc[((df["ts_x"] - df["ts_y"]).abs() < 24 * 60 * 60) & (df["aid_x"] != df["aid_y"])]

                    # -- memory management compute in parts
                    df = df.loc[(df["aid_x"] >= part * size) & (df["aid_x"] < (part + 1) * size)]

                    # -- assign weights
                    df = df[["session", "aid_x", "aid_y", "type_y"]].drop_duplicates(["session", "aid_x", "aid_y"])
                    df["wgt"] = df["type_y"].map(type_weight)
                    df = df[["aid_x", "aid_y", "wgt"]]
                    df["wgt"] = df["wgt"].astype("float32")
                    df = df.groupby(["aid_x", "aid_y"])["wgt"].sum()

                    # -- combine inner chunks
                    if k == start:
                        tmp2 = df
                    else:
                        # fill NaN by fiil_value
                        tmp2 = tmp2.add(df, fill_value=0)

                    del df
                    gc.collect()

                # -- combine outer chunks
                if start == 0:
                    tmp = tmp2
                else:
                    tmp = tmp.add(tmp2, fill_value=0)

            # -- convert matrix to dictionary
            # 重みは降順
            # aid_xの中で重み順で並んでる -> 重い方がaid_xが起きた時にaid_yが起きやすい
            tmp = tmp.reset_index()
            tmp = tmp.sort_values(["aid_x", "wgt"], ascending=[True, False])

            # -- save top 15
            tmp = tmp.reset_index(drop=True)
            tmp["n"] = tmp.groupby("aid_x")["aid_y"].cumcount()
            tmp = tmp.loc[tmp["n"] < 15].drop("n", axis=1)

            # -- save part to disk
            tmp.to_pandas().to_parquet(f"{root}/output/top_15_carts_orders_{part}_ver{DataModule.ver}.parquet")

    @staticmethod
    @beartype
    def _compute_buy2buy_co_visitation_matrix(
        files: list[Path], root: Path = Path("./"), use_data_cache: bool = True
    ) -> None:

        chunk = int(np.ceil(len(files)))
        disk_pieces = 1
        read_ct = 5
        type_weight = {0: 1, 1: 6, 2: 3}
        size = 1.86e6 / disk_pieces
        for part in range(disk_pieces):
            print(f" -------- dist part {part} ----------- ")
            for j in range(6):
                start = j * chunk
                end = min((j + 1) * chunk, len(files))
                print(f" --- Processing files {start} through {end} in groups of {read_ct}")
                for k in range(start, end, read_ct):
                    # -- read file
                    df = [DataModule._read_file(files[k], use_data_cache=use_data_cache)]
                    for i in range(1, read_ct):
                        if k + i >= end:
                            continue
                        df.append(DataModule._read_file(files[k + i], use_data_cache=use_data_cache))
                    df = cudf.concat(df, ignore_index=True, axis=0)
                    df = df.loc[df["type"].isin([1, 2])]
                    df = df.sort_values(["session", "ts"], ascending=[True, False])

                    # -- use tail of session
                    df = df.reset_index(drop=True)
                    df["n"] = df.groupby("session").cumcount()
                    df = df.loc[df["n"] < 30].drop("n", axis=1)

                    # -- create pairs
                    # select 14 days
                    # 14日以内に再購買があったペアを抽出する
                    df = df.merge(df, on="session")
                    df = df.loc[((df["ts_y"] - df["ts_x"]).abs() < 14 * 24 * 60 * 60) & (df["aid_x"] != df["aid_y"])]

                    # -- memory management compute in pairs
                    df = df.loc[(df["aid_x"] >= part * size) & (df["aid_x"] <= (part + 1) * size)]

                    # -- assign weights
                    # 購買の共起をcount upしてる
                    df = df[["session", "aid_x", "aid_y", "type_y"]].drop_duplicates(["session", "aid_x", "aid_y"])
                    df["wgt"] = 1
                    df = df[["aid_x", "aid_y", "wgt"]]
                    df["wgt"] = df["wgt"].astype("float32")
                    df = df.groupby(["aid_x", "aid_y"])["wgt"].sum()

                    # -- combine inner chunks
                    if k == start:
                        tmp2: cudf.DataFrame = df
                    else:
                        tmp2 = tmp2.add(df, fill_value=0)

                    del df
                    gc.collect()

                # -- combine outer chunks
                if start == 0:
                    tmp = tmp2
                else:
                    tmp = tmp.add(tmp2, fill_value=0)

            # -- convert matrix to dictionary
            tmp = tmp.reset_index()
            tmp = tmp.sort_values(["aid_x", "wgt"], ascending=[True, False])

            # -- save top 15
            tmp = tmp.reset_index(drop=True)
            tmp["n"] = tmp.groupby("aid_x")["aid_y"].cumcount()
            tmp = tmp.loc[tmp["n"] < 15].drop("n", axis=1)
            tmp.to_pandas().to_parquet(f"{root}/output/top_15_buy2buy_{part}_ver{DataModule.ver}.parquet")

    @staticmethod
    @beartype
    def _compute_clicks_co_visitation_matrix(
        files: list[Path], root: Path = Path("./"), use_data_cache: bool = True
    ) -> None:
        chunk = int(np.ceil(len(files)))
        disk_pieces = 4
        read_ct = 5
        type_weight = {0: 1, 1: 6, 2: 3}
        size = 1.86e6 / disk_pieces
        for part in range(disk_pieces):
            print(f" -------- dist part {part} ----------- ")
            for j in range(6):
                start = j * chunk
                end = min((j + 1) * chunk, len(files))
                print(f" --- Processing files {start} through {end} in groups of {read_ct}")

                for k in range(start, end, read_ct):
                    # -- read file
                    df = [DataModule._read_file(files[k], use_data_cache=use_data_cache)]
                    for i in range(1, read_ct):
                        if k + i >= end:
                            continue
                        df.append(DataModule._read_file(files[k + i], use_data_cache=use_data_cache))
                    df = cudf.concat(df, ignore_index=True, axis=0)
                    df = df.sort_values(["session", "ts"], ascending=[True, False])

                    # -- use tail of session
                    df = df.reset_index(drop=True)
                    df["n"] = df.groupby("session").cumcount()
                    df = df.loc[df["n"] < 30].drop("n", axis=1)

                    # -- create pairs
                    df = df.merge(df, on="session")
                    df = df.loc[((df["ts_y"] - df["ts_x"]).abs() < 24 * 60 * 60) & (df["aid_x"] != df["aid_y"])]

                    # -- memory management compute in parts
                    df = df.loc[(df["aid_x"] >= part * size) & (df["aid_x"] <= (part + 1) * size)]

                    # -- assign weights
                    # 1659304800 is minimum value of ts
                    # 1662328791  is maximum value of ts
                    # ほしいweightの最大値は4, 最小値は1, この範囲は任意の値で、cvのスコアが一番いい値だったらしい
                    # point slope: (y_2 - y_1) = m * (x2 - x1)
                    # where:
                    # x2 = 1662328791
                    # x1 = 1659304800
                    # y2 = 4
                    # y1 = 1
                    #
                    # Reference 1.
                    # > There are 5 weeks of data to train our models (4 train 1 test).
                    # > The max value determines how much emphasis we give to recent data.
                    # > If we make the maximum 6 for example, then pairs of items during the last week of test data (most recent) will have weights between 5 and 6.
                    # > Then second to last week will have weights 4 to 5.
                    # > And then 3 to 4, 2 to 3, and 1 to 2.
                    # > (i.e. the most recent week is approx 3.7x more important than the oldest week in determining the pairs in our co-visitation matrix).
                    #
                    # ## Reference:
                    # 1. https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575/comments#2043151
                    df = df[["session", "aid_x", "aid_y", "ts_x"]].drop_duplicates(["session", "aid_x", "aid_y"])
                    df["wgt"] = 1 + 3 * (df["ts_x"] - 1659304800) / (1662328791 - 1659304800)
                    df = df[["aid_x", "aid_y", "wgt"]]
                    df["wgt"] = df["wgt"].astype("float32")
                    df = df.groupby(["aid_x", "aid_y"])["wgt"].sum()

                    # -- combine inner chunks
                    if k == start:
                        tmp2: cudf.DataFrame = df
                    else:
                        tmp2 = tmp2.add(df, fill_value=0)

                    del df
                    gc.collect()

                # -- combine outer chunks
                if start == 0:
                    tmp = tmp2
                else:
                    tmp = tmp.add(tmp2, fill_value=0)

            # -- convert matrix to dictionary
            tmp = tmp.reset_index()
            tmp = tmp.sort_values(["aid_x", "wgt"], ascending=[True, False])

            # -- save top 15
            tmp = tmp.reset_index(drop=True)
            tmp["n"] = tmp.groupby("aid_x")["aid_y"].cumcount()
            tmp = tmp.loc[tmp["n"] < 20].drop("n", axis=1)
            tmp.to_pandas().to_parquet(f"{root}/output/top_20_clicks_{part}_ver{DataModule.ver}.parquet")

    @staticmethod
    @beartype
    def compute_co_vistation_matrix(root: Path = Path("./"), use_data_cache: bool = True) -> None:
        """compute co-visitation matrix and save"""
        logger.info(" ------ compute co-visitation matrix and save to pickle/parquet.")
        files = list(Path(f"{root}/input/otto-chunk-data-in-parquet-format/").glob("*_parquet/*"))
        logger.info(f"{len(files)=}")
        if use_data_cache:
            DataModule.save_data_to_global_cache(files=files)

        GenerateModule._comupute_carts_orders_co_visitation_matrix(files, use_data_cache=use_data_cache)
        GenerateModule._compute_buy2buy_co_visitation_matrix(files, use_data_cache=use_data_cache)
        GenerateModule._compute_clicks_co_visitation_matrix(files, use_data_cache=use_data_cache)


class SuggestModule:
    @staticmethod
    @UtilsModule.timeit
    @beartype
    def suggest_items(session_aids: pd.Series, session_types, next_aids: dict[np.int32, Counter]) -> list[list[int]]:
        """suggest items"""

        labels = []
        no_data = 0
        no_data_all_aids = 0
        type_weight_multipliers = {0: 1, 1: 6, 2: 3}
        for aids, types in zip(session_aids, session_types):
            if len(aids) >= 20:

                weights = np.logspace(0.1, 1.0, len(aids), base=2, endpoint=True) - 1
                aids_temp = defaultdict(lambda: 0)
                for aid, weight, type in zip(aids, weights, types):
                    aids_temp[aid] += weight * type_weight_multipliers[type]

                sorted_items = [k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
                labels.append(sorted_items[:20])
            else:
                unique_preserved_order_aids: list[np.int32] = list(dict.fromkeys(aids[::-1]))
                aids_len_start = len(unique_preserved_order_aids)

                candidates = []
                for aid in unique_preserved_order_aids:
                    if aid in next_aids:
                        candidates += [aid for aid, count in next_aids[aid].most_common(20)]

                unique_preserved_order_aids += [
                    aid for aid, count in Counter(candidates).most_common(40) if aid not in unique_preserved_order_aids
                ]

                labels.append(unique_preserved_order_aids[:20])

                if not candidates:
                    no_data += 1
                if aids_len_start == len(unique_preserved_order_aids):
                    no_data_all_aids += 1

        print(f"Test sessions that we did not manage to extend based on the co-visitation matrix: {no_data_all_aids}")
        return labels

    @staticmethod
    @beartype
    def suggest_clicks(df: pd.DataFrame, top_20_clicks: dict, top_clicks: np.ndarray) -> list[int]:
        type_weight_multipliers = {0: 1, 1: 6, 2: 3}
        # -- user history aids and types
        aids = df["aid"].tolist()
        types = df["type"].tolist()
        unique_aids = list(dict.fromkeys(aids[::-1]))

        # -- rerank candidates using weights
        if len(unique_aids) >= 20:
            weights = np.logspace(0.1, 1.0, len(aids), base=2, endpoint=True) - 1
            aids_temp = Counter()

            # -- rerank based on repeat items and type of items
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * type_weight_multipliers[t]

            sorted_aids = [k for k, _ in aids_temp.most_common(20)]
            return sorted_aids

        else:
            # -- use 'cliks' co-visitation matrix
            aids2 = list(itertools.chain(*[top_20_clicks[aid] for aid in unique_aids if aid in top_20_clicks]))

            # -- rerank candidates
            top_aids2 = [aid2 for aid2, _ in Counter(aids2).most_common(20) if aid2 not in unique_aids]

            result = unique_aids + top_aids2[: 20 - len(unique_aids)]
            return result + list(top_clicks)[: 20 - len(result)]

    @staticmethod
    @beartype
    def suggest_buys(df: pd.DataFrame, top_15_buy2buy: dict, top_15_buys: dict, top_orders: np.ndarray) -> list[int]:
        type_weight_multipliers = {0: 1, 1: 6, 2: 3}
        # -- use history aids and types
        aids = df["aid"].tolist()
        types = df["type"].tolist()

        # -- unique aids and unique buys
        unique_aids = list(dict.fromkeys(aids[::-1]))
        df = df.loc[(df["type"] == 1) | (df["type"] == 2)]
        unique_buys = list(dict.fromkeys(df["aid"].tolist()[::-1]))

        # -- rerank candidates using weights
        if len(unique_aids) >= 20:
            weights = np.logspace(0.5, 1.0, len(aids), base=2, endpoint=True) - 1
            aids_temp = Counter()

            # -- rerank based on repeat items and type of items
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * type_weight_multipliers[t]

            # -- rerank candidates using 'buy2buy' co-visitation-matrix
            aids3 = list(itertools.chain(*[top_15_buy2buy[aid] for aid in unique_buys if aid in top_15_buy2buy]))
            for aid in aids3:
                aids_temp[aid] += 0.1

            sorted_aids = [k for k, _ in aids_temp.most_common(20)]
            return sorted_aids
        else:
            # -- use 'cart order' co-visitation matrix
            aids2 = list(itertools.chain(*[top_15_buys[aid] for aid in unique_aids if aid in top_15_buys]))

            # -- use 'buy2buy' co-visitation matrix
            aids3 = list(itertools.chain(*[top_15_buy2buy[aid] for aid in unique_buys if aid in top_15_buy2buy]))

            # -- rerank candidate
            top_aid2 = [aid2 for aid2, _ in Counter(aids2 + aids3).most_common(20) if aid2 not in unique_aids]
            result = unique_aids + top_aid2[: 20 - len(unique_aids)]
            return result + list(top_orders)[: 20 - len(result)]


# %%
class MetricsModule:
    @staticmethod
    @UtilsModule.timeit
    @beartype
    def computed_metric(
        submission_df: pd.DataFrame, gt_df: pd.DataFrame
    ) -> tuple[float, dict[str, float], dict[str, pd.DataFrame]]:
        print(" ----------- start to computation of metrics ------------ \n")
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
            test_labels = gt_df.loc[gt_df["type"] == t]

            test_labels = test_labels.merge(sub, how="left", on=["session"])
            test_labels["hits"] = test_labels.apply(
                lambda df: len(set(df["ground_truth"]).intersection(set(df["labels"]))), axis=1
            )
            test_labels["gt_count"] = test_labels["ground_truth"].str.len().clip(0, 20)
            recall = test_labels["hits"].sum() / test_labels["gt_count"].sum()
            metrics_per_type[t] = recall
            validation_df_per_type[t] = test_labels
            score += weights[t] * recall
            print(f"{t} recall = {recall}")

        assert 0.0 <= score <= 1.0, f"invalid score: {score}"
        return score, metrics_per_type, validation_df_per_type


@UtilsModule.timeit
def validate() -> None:
    data_for_valid = DataModule.load_data(is_for_valid=True)
    co_visitation_matrices = DataModule.load_co_visitation_matrix_parquet(ver=5)
    valid = data_for_valid.valid_df
    valid_gt = data_for_valid.test_labels
    if valid is None:
        raise ValueError

    pred_df = made_predictions(test=valid)
    score, metrics_per_type, validation_df_per_type = MetricsModule.computed_metric(
        submission_df=pred_df, gt_df=valid_gt
    )


# %%


@UtilsModule.timeit
@beartype
def made_predictions(test: pd.DataFrame) -> pd.DataFrame:
    co_visitation_matrices = DataModule.load_co_visitation_matrix_parquet(ver=5)
    logger.info("Here are size of our 3 co-visitation matrices:")
    logger.info(
        f"{len(co_visitation_matrices['top20_clicks']) = }, \n"
        + f"{len(co_visitation_matrices['top15_buys']) = }, \n"
        + f"{len(co_visitation_matrices['top15_buy2buy']) = }"
    )

    top_clicks = test.loc[test["type"] == "clicks", "aid"].value_counts().index.values[:20]
    top_orders = test.loc[test["type"] == "orders", "aid"].value_counts().index.values[:20]

    # -- create submission
    start = time.time()
    logger.info(" ---- start to make predicitons --- ")
    pred_df_clicks = (
        test.sort_values(["session", "ts"])
        .groupby(["session"])
        .progress_apply(
            lambda x: SuggestModule.suggest_clicks(
                df=x, top_20_clicks=co_visitation_matrices["top20_clicks"], top_clicks=top_clicks
            )
        )
    )

    pred_df_buys = (
        test.sort_values(["session", "ts"])
        .groupby(["session"])
        .progress_apply(
            lambda x: SuggestModule.suggest_buys(
                df=x,
                top_15_buy2buy=co_visitation_matrices["top15_buy2buy"],
                top_15_buys=co_visitation_matrices["top15_buys"],
                top_orders=top_orders,
            )
        )
    )
    duration = time.time() - start
    logger.info(f"Predict duration : {duraition} [s] = {duration/60:.3f} [m]")

    clicks_pred_df = pd.DataFrame(pred_df_clicks.add_suffix("_clicks"), columns=["labels"]).reset_index()
    orders_pred_df = pd.DataFrame(pred_df_buys.add_suffix("_orders"), columns=["labels"]).reset_index()
    carts_pred_df = pd.DataFrame(pred_df_buys.add_suffix("_carts"), columns=["labels"]).reset_index()
    pred_df = pd.concat([clicks_pred_df, orders_pred_df, carts_pred_df])

    logger.debug(pred_df.head())

    pred_df.columns = ["session_type", "labels"]
    pred_df["labels"] = pred_df["labels"].apply(lambda x: " ".join(map(str, x)))
    return pred_df


def main():
    logger.info(f"\n {'#'*20} {Config.exp_name}: debug mode: {Config.debug}: {Config.do_validation = }{'#'*20} \n")
    if Config.do_validation:
        validate()

    data = DataModule.load_data(debug=Config.debug)
    # train = data.train_df
    # valid = data.valid_df
    # id2type = data.id2type
    # type2id = data.type2id
    # sample_sub = data.sample_sub

    ########################
    # make inference
    ########################
    test = data.test_df
    if Config.do_re_compute:
        GenerateModule.compute_co_vistation_matrix()
    else:
        logger.info("\n ####### WARNING #########: load cache from DISK \n")
    pred_df = made_predictions(test=test)
    sub_save_path = Path("./output") / Config.exp_name / "submission.csv"
    pred_df.to_csv(sub_save_path, index=False)
    logger.info(pred_df.head())
    logger.info("\n ##################### END ###################### \n")


if __name__ == "__main__":
    main()
