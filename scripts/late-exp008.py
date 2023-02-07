from __future__ import annotations

import datetime
import gc
import itertools
from collections import Counter
from pathlib import Path

import cudf
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from src.constants import INPUT_DIR, OUT_DIR, ROOT, TYPE_LABELS

VER = 5


class CFG:
    exp_name = "exp008_late_sub"
    make_covis_matrix: bool = False
    debug: bool = False
    carts2orders_disk_pieces: int = 5
    buys2buys_disk_pieces: int = 1
    clicks2clicks_disk_pieces: int = 4


# CACHE THE DATA ON CPU BEFORE PROCESSING ON GPU
DATA_CACHE: dict[str, str] = {}


def load_all_train_data() -> pl.DataFrame:
    # dfs = [
    #     pl.read_parquet(chunk_file).with_columns(
    #         [
    #             (pl.col("ts") / 1000).cast(pl.UInt32).alias("ts"),
    #             pl.col("type").map(lambda x: TYPE_LABELS[x]).alias("type"),
    #         ]
    #     )
    #     for chunk_file in (INPUT_DIR / "otto-chunk-data-in-parquet-format/train_parquet").glob("*")
    # ]
    dfs = []
    for chunk_file in (INPUT_DIR / "otto-chunk-data-in-parquet-format/train_parquet").glob("*"):
        df = pl.read_parquet(chunk_file).with_columns(
            [
                (pl.col("ts") / 1000).cast(pl.UInt32).alias("ts"),
                pl.col("type").apply(lambda x: TYPE_LABELS[x]).alias("type"),
            ]
        )
        dfs.append(df)
    return pl.concat(dfs)


def load_all_test_data() -> pl.DataFrame:
    dfs = []
    for chunk_file in (INPUT_DIR / "otto-chunk-data-in-parquet-format/test_parquet").glob("*"):
        df = pl.read_parquet(chunk_file).with_columns(
            [
                (pl.col("ts") / 1000).cast(pl.UInt32).alias("ts"),
                pl.col("type").apply(lambda x: TYPE_LABELS[x]).alias("type"),
            ]
        )
        dfs.append(df)
    return pl.concat(dfs)


def load_train_data() -> pl.DataFrame:
    return pl.read_parquet(f"{INPUT_DIR}/otto-train-and-test-data-for-local-validation/train.parquet")


def read_file(f: Path) -> cudf.DataFrame:
    return cudf.DataFrame(DATA_CACHE[f])


def read_file_to_cache(f: Path) -> pd.DataFrame:
    df = pd.read_parquet(f)
    df.ts = (df.ts / 1000).astype("int32")
    df["type"] = df["type"].map(TYPE_LABELS).astype("int8")
    return df


def load_data_to_cache() -> None:
    files = list((INPUT_DIR / "otto-chunk-data-in-parquet-format").glob("*_parquet/*"))
    for f in files:
        DATA_CACHE[f] = read_file_to_cache(f)


def load_co_visitation_matrices() -> tuple[dict, dict, dict]:
    def pqt_to_dict(df: pd.DataFrame) -> dict:
        return df.groupby("aid_x").aid_y.apply(list).to_dict()

    # LOAD THREE CO-VISITATION MATRICES
    top_20_clicks = pqt_to_dict(pd.read_parquet(f"{OUT_DIR}/top_20_clicks_v{VER}_0.pqt"))
    for k in range(1, CFG.clicks2clicks_disk_pieces):
        top_20_clicks.update(pqt_to_dict(pd.read_parquet(f"{OUT_DIR}/top_20_clicks_v{VER}_{k}.pqt")))

    top_20_buys = pqt_to_dict(pd.read_parquet(f"{OUT_DIR}/top_20_carts_orders_v{VER}_0.pqt"))
    for k in range(1, CFG.carts2orders_disk_pieces):
        top_20_buys.update(pqt_to_dict(pd.read_parquet(f"{OUT_DIR}/top_20_carts_orders_v{VER}_{k}.pqt")))

    top_20_buy2buy = pqt_to_dict(pd.read_parquet(f"{OUT_DIR}/top_20_buy2buy_v{VER}_0.pqt"))

    return top_20_clicks, top_20_buys, top_20_buy2buy


def make_covis_matrix_carts2orders() -> None:
    DISK_PIECES = CFG.carts2orders_disk_pieces
    SIZE = 1.86e6 / DISK_PIECES
    READ_CT = 5
    assert len(DATA_CACHE) > 0
    files = DATA_CACHE.keys()
    CHUNK = int(np.ceil(len(files) / 6))
    type_weight = {0: 1, 1: 6, 2: 3}
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
        tmp.to_pandas().to_parquet(f"{OUT_DIR}/top_20_carts_orders_v{VER}_{PART}.pqt")


def make_covis_matrix_buy2buy() -> None:
    DISK_PIECES = CFG.buys2buys_disk_pieces
    SIZE = 1.86e6 / DISK_PIECES
    READ_CT = 5
    assert len(DATA_CACHE) > 0
    files = DATA_CACHE.keys()
    CHUNK = int(np.ceil(len(files) / 6))
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
        tmp.to_pandas().to_parquet(f"{OUT_DIR}/top_20_buy2buy_v{VER}_{PART}.pqt")


def make_covis_matrix_clicks2clicks() -> None:
    DISK_PIECES = CFG.clicks2clicks_disk_pieces
    SIZE = 1.86e6 / DISK_PIECES
    READ_CT = 5
    assert len(DATA_CACHE) > 0
    files = DATA_CACHE.keys()
    CHUNK = int(np.ceil(len(files) / 6))
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
        tmp.to_pandas().to_parquet(f"{OUT_DIR}/top_20_clicks_v{VER}_{PART}.pqt")


def create_item_features(df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
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


def create_user_features(df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
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

    # first_session_ts = df.groupby("session").first().select(["session", "user_session_length", "ts"]).sort("session")
    # last_session_ts = df.groupby("session").last().select(["session", "user_session_length", "ts"]).sort("session")
    # session_sec_df = (
    #     last_session_ts.with_columns(
    #         [pl.col("session").cast(pl.UInt32), pl.col("ts").cast(pl.Int64).alias("sec_last_session")]
    #     )
    #     .with_column(first_session_ts["ts"].cast(pl.Int64).alias("sec_first_session"))
    #     .with_column((pl.col("sec_last_session") - pl.col("sec_first_session")).alias("sec_to_end_session"))
    #     .select(["session", "sec_to_end_session", "user_session_length"])
    # )

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

    # user_features = user_features.join(session_sec_df, on="session", how="left")

    return user_features


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
    id2label = {v: k for k, v in TYPE_LABELS.items()}
    for t in weights.keys():
        sub = submission_df.loc[submission_df["session_type"].str.contains(t)].copy()
        sub["session"] = sub["session_type"].apply(lambda x: int(x.split("_")[0]))
        sub["labels"] = sub["labels"].apply(lambda x: [int(i) for i in x.split()[:20]])

        # -- gt label
        gt_df["type"] = gt_df.type.map(lambda x: x if isinstance(x, str) else id2label[x])
        test_labels = gt_df.loc[gt_df["type"] == t]

        test_labels = test_labels.merge(sub, how="left", on=["session"])
        # assert len(test_labels) == sub_ln, f"mismath length: {sub_ln = }, {len(test_labels) = }"
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


def suggest_clicks(df: pl.DataFrame, top_20_clicks: dict, top_20_clicks_set: dict, num: int = 20) -> pl.DataFrame:
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}
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


def suggest_buys(
    df: pl.DataFrame,
    top_20_buy2buy: dict,
    top_20_buys: dict,
    top_20_buy2buy_set: set,
    top_20_buys_set: set,
    num: int = 20,
) -> pl.DataFrame:
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}
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


def create_candidates(
    df: pl.DataFrame,
    top_20_clicks: dict,
    top_20_buy2buy: dict,
    top_20_buys: dict,
    clicks_num: int = 20,
    buys_num: int = 20,
) -> pl.DataFrame:

    top_20_clicks_set = set(top_20_clicks)
    top_20_buys_set = set(top_20_buys)
    top_20_buy2buy_set = set(top_20_buy2buy)

    print(f"{df['session'].n_unique() = }")

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

    clicks_candidates = candidates_clicks.select(["session", "aid"]).with_column(pl.lit("clicks").alias("type"))
    orders_candidates = candidates_buys.select(["session", "aid"]).clone().with_column(pl.lit("orders").alias("type"))
    carts_candidates = candidates_buys.select(["session", "aid"]).clone().with_column(pl.lit("carts").alias("type"))

    del candidates_clicks, candidates_buys, top_20_clicks, top_20_buys, top_20_buy2buy
    gc.collect()

    candidates = pl.concat(
        (
            clicks_candidates,
            orders_candidates,
            carts_candidates,
            top_clicks_candidates,
            top_orders_candidates,
            top_carts_candidates,
        )
    )

    candidates = candidates.with_columns(
        [
            pl.col("session").cast(pl.UInt32),
            pl.col("type").apply(lambda x: TYPE_LABELS[x]),
        ]
    )
    return candidates


def make_label(candidates: pl.DataFrame | None = None) -> pl.DataFrame:
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

    # 実際のデータを読み込んでくる
    train_targets = pl.read_parquet(f"{INPUT_DIR}/otto-train-and-test-data-for-local-validation/test_labels.parquet")
    click_target = make_target(train_targets, target_type="clicks")
    cart_target = make_target(train_targets, target_type="carts")
    order_target = make_target(train_targets, target_type="orders")

    # 実際にtargetのeventがあった行は1にそれ以外はNaN=0になっている
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


FEATURES = None


def train_per_fold(candidates: pl.DataFrame, fold: str, label: str, df: pl.DataFrame) -> xgb.Booster:
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
    ).rename({"session": "user"})
    valid_user_features = create_user_features(
        df=df.filter(pl.col("session").is_in(candidates[valid_idx]["user"].unique().to_list()))
    ).rename({"session": "user"})

    train_candidates = candidates[train_idx].join(train_user_features, on="user").fill_null(-1)
    valid_candidates = candidates[valid_idx].join(valid_user_features, on="user").fill_null(-1)

    assert not check_same_value_exist(train_candidates, valid_candidates, key="user")

    del train_user_features, valid_user_features
    gc.collect()

    skip_cols = {
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
    feature_cols = [col for col in train_candidates.columns if col not in skip_cols]
    global FEATURES
    FEATURES = feature_cols
    print(f"[global] {FEATURES = }")

    print(f"{train_candidates[:, FEATURES] = }")
    print(f"{valid_candidates[:, FEATURES] = }")

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

    del query_train, query_valid
    gc.collect()

    xgb_parms = {
        "objective": "rank:pairwise",
        "tree_method": "gpu_hist",
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
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
    return model


def predict(candidates: pl.DataFrame, target: str, model: xgb.Booster | None = None) -> pl.DataFrame:

    candidates = candidates.sort("user")
    group: np.ndarray = (
        candidates.to_pandas().groupby("user").size().to_frame("size")["size"].to_numpy().astype("uint16")
    )
    # test_group = [50] * (len(test_candidates) // 50)

    preds = np.zeros(len(candidates), dtype="f4")
    for fold in range(1, 6):
        if model is None:
            model = xgb.Booster()
            model.load_model(f"{OUT_DIR}/{CFG.exp_name}/XGB_fold{fold}_{target}.xgb")
        model.set_param({"predictor": "gpu_predictor"})
        dtest = xgb.DMatrix(data=candidates.to_pandas().loc[:, FEATURES], group=group)
        preds += model.predict(dtest) / 5

    predictions = candidates.select(["user", "item"]).clone().with_column(pl.Series("score", preds, dtype=pl.Int32))

    del candidates, preds
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

    print(predictions)

    predictions = predictions.with_columns(
        [
            pl.col("item").apply(lambda x: " ".join(map(str, x))).alias("labels"),
            (pl.col("user").cast(pl.Utf8) + f"_{target}").alias("session_type"),
        ]
    ).select(["session_type", "labels"])
    return predictions


def valid_per_fold(model_per_label: dict[str, xgb.Booster], valid_candidates: pl.DataFrame, valid_gt: pl.DataFrame):
    preds = []
    for label in model_per_label.keys():
        preds.append(
            predict(
                candidates=valid_candidates.filter(pl.col("type") == TYPE_LABELS[label]),
                target=label,
                model=model_per_label[label],
            )
        )
    preds = pl.concat(preds)

    score, metrics_per_type, validation_df_per_type = computed_metric(
        submission_df=preds.to_pandas(), gt_df=valid_gt.to_pandas()
    )
    print(f"{score = }")
    print(f"{metrics_per_type = }")
    scores_path = OUT_DIR / CFG.exp_name / "score.txt"
    metrics_path = OUT_DIR / CFG.exp_name / "metrics.txt"
    if CFG.debug:
        with scores_path.open("w") as fp:
            fp.write(f"{score}")
        with metrics_path.open("w") as fp:
            fp.write(f"{metrics_per_type}")


def train(train_df: pl.DataFrame, valid_df: pl.DataFrame | None) -> None:
    top_20_clicks, top_20_buys, top_20_buy2buy = load_co_visitation_matrices()
    print("Here are size of our 3 co-visitation matrices:")
    print(len(top_20_clicks), len(top_20_buy2buy), len(top_20_buys))

    candidates = create_candidates(
        df=train_df,
        top_20_clicks=top_20_clicks,
        top_20_buy2buy=top_20_buy2buy,
        top_20_buys=top_20_buys,
        clicks_num=20,
        buys_num=20,
    )
    candidates = make_label(candidates)

    do_validation = valid_df is not None
    if do_validation:
        item_features = create_item_features(df=pl.concat([load_train_data(), train_df]))
        # valid_gt = (
        #     valid_df.select(["session", "aid", "type"])
        #     .groupby(["session", "type"])
        #     .agg_list()
        #     .rename({"aid": "ground_truth"})
        # )
        valid_gt = pl.read_parquet(f"{INPUT_DIR}/last_half_targets.pqt")
        # valid_gt = pl.read_parquet(INPUT_DIR / "otto-validation" / "test_labels.parquet").with_column(
        #     pl.col("ground_truth").cast(pl.List(pl.Utf8))
        # )
        valid_gt = valid_gt.filter(pl.col("session").is_in(valid_df["session"].unique()))
        valid_candidates = (
            create_candidates(
                df=valid_df,
                top_20_clicks=top_20_clicks,
                top_20_buy2buy=top_20_buy2buy,
                top_20_buys=top_20_buys,
                clicks_num=20,
                buys_num=20,
            )
            .rename({"session": "user", "aid": "item"})
            .join(create_user_features(df=valid_df).rename({"session": "user"}), on="user")
            .join(item_features.rename({"aid": "item"}), on="item")
            .fill_null(-1.0)
        )
    else:
        item_features = create_item_features(df=pl.concat([load_all_train_data(), load_all_test_data()]))

    # Reference
    # 1. https://www.kaggle.com/competitions/otto-recommender-system/discussion/379952
    candidates = candidates.join(item_features.rename({"aid": "item"}), on="item").sort("user")
    gkf = GroupKFold(n_splits=5)
    for fold, (train_idx, valid_idx) in enumerate(gkf.split(X=candidates, groups=candidates["user"]), start=1):
        candidates = candidates.with_column(pl.lit(0).alias(f"fold{fold}").cast(pl.UInt8))
        candidates[valid_idx, f"fold{fold}"] = 1

    folds = [column for column in candidates.columns if column.startswith("fold")]
    for fold in folds:
        print("##################\n" + f"## Start {fold}\n" + "##################\n")
        model_per_label = {}
        for label in ["clicks", "carts", "orders"]:
            model = train_per_fold(candidates=candidates, label=label, fold=fold, df=train_df)
            model_per_label[label] = model
            if not do_validation:
                model.save_model(f"{OUT_DIR}/{CFG.exp_name}/XGB_{fold}_{label}.xgb")

        if do_validation:
            valid_per_fold(model_per_label, valid_candidates, valid_gt)


def test() -> None:
    test_df = load_all_test_data()
    if CFG.debug:
        debug_user_num = 100
        debug_user = test_df["session"].unique()[:debug_user_num]
        test_df = test_df.filter(pl.col("session").is_in(debug_user))

    print(f"{test_df = }")

    top_20_clicks, top_20_buys, top_20_buy2buy = load_co_visitation_matrices()
    print("Here are size of our 3 co-visitation matrices:")
    print(len(top_20_clicks), len(top_20_buy2buy), len(top_20_buys))

    candidates = create_candidates(
        df=test_df,
        top_20_clicks=top_20_clicks,
        top_20_buy2buy=top_20_buy2buy,
        top_20_buys=top_20_buys,
        clicks_num=20,
        buys_num=20,
    )

    df_for_item_features = pl.concat([load_all_train_data(), test_df])
    item_features = create_item_features(df_for_item_features)
    user_features = create_user_features(test_df)

    candidates = (
        candidates.join(item_features, on="aid")
        .join(user_features, on="session")
        .fill_null(-1.0)
        .rename({"session": "user", "aid": "item"})
    )

    TEST_CHUNKS = 3

    def _save_candidates_with_feature_to_pieces(candidates: pd.DataFrame) -> None:
        chunk_size = int(np.ceil(len(candidates.user.unique()) / TEST_CHUNKS))
        users = candidates["user"].unique()
        cnt = 0
        for k in tqdm(range(TEST_CHUNKS), desc="make candidates with features"):
            sessions = users[int(k * chunk_size) : int((k + 1) * chunk_size)]
            df = candidates[candidates["user"].isin(sessions)]
            cnt += len(df)
            df.to_pickle(f"{OUT_DIR}/test_candidate_with_features_p{k}.pickle")

    _save_candidates_with_feature_to_pieces(candidates=candidates.to_pandas())

    def _internal_make_predict_per_piece(k: int) -> pd.DataFrame:
        candidates_with_feature = pl.from_pandas(pd.read_pickle(f"{OUT_DIR}/test_candidate_with_features_p{k}.pickle"))
        print(candidates_with_feature)

        preds = []
        for label in ["clicks", "carts", "orders"]:
            labeled_candindates = candidates_with_feature.filter(pl.col("type") == TYPE_LABELS[label])
            preds.append(predict(labeled_candindates, target=label))
        preds = pl.concat(preds)
        return preds

    final_sub = pl.concat([_internal_make_predict_per_piece(k=k) for k in tqdm(range(TEST_CHUNKS))], how="vertical")
    final_sub.write_csv(f"{OUT_DIR}/{CFG.exp_name}/submission.csv")
    print(f"{final_sub = }")


def check_same_value_exist(df_i: pl.DataFrame, df_j: pl.DataFrame, key="session") -> bool:
    return len(set(df_i[key]).intersection(set(df_j[key]))) > 0


def main():
    if CFG.make_covis_matrix:
        load_data_to_cache()
        make_covis_matrix_carts2orders()
        make_covis_matrix_buy2buy()
        make_covis_matrix_clicks2clicks()

    Path(OUT_DIR / CFG.exp_name).mkdir(parents=True, exist_ok=True)

    # train_df = load_all_train_data()
    train_df = pl.read_parquet(INPUT_DIR / "first_half_valid_df.pqt").select(["session", "aid", "ts", "type"])
    valid_df = pl.read_parquet(INPUT_DIR / "last_half_valid_df.pqt").select(["session", "aid", "ts", "type"])
    assert not check_same_value_exist(
        train_df, valid_df
    ), f"{set(train_df['session']).intersection(set(valid_df['session'])) = }"
    print(train_df)
    if CFG.debug:
        debug_user_num = 100
        debug_train_user = train_df["session"].unique()[:debug_user_num]
        debug_valid_user = valid_df["session"].unique()[:debug_user_num]
        train_df = train_df.filter(pl.col("session").is_in(debug_train_user))
        valid_df = valid_df.filter(pl.col("session").is_in(debug_valid_user))
    print(f"{train_df.shape = }, {valid_df.shape = }")
    train(train_df=train_df, valid_df=valid_df)

    # ---------------
    # 全データ使って学習
    # ---------------
    print("##################\n" + "## training using all data \n" + "##################\n")
    train(train_df=pl.concat([train_df, valid_df]), valid_df=None)

    # ---------------
    # submissionを作る
    # ---------------
    test()


if __name__ == "__main__":
    main()
