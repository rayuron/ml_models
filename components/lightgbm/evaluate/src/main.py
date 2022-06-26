import pickle
from pathlib import Path

import fire
import lightgbm as lgb
import pandas as pd


def main(
    train_data_dir_path,
    valid_data_dir_path,
    data_file_format,
    target_col,
    objective,
    num_class,
    valid_metrics,
    learning_rate,
    model_path,
):
    if data_file_format == "csv":
        train = pd.concat((pd.read_csv(f) for f in Path(train_data_dir_path).glob("*")))
        valid = pd.concat((pd.read_csv(f) for f in Path(valid_data_dir_path).glob("*")))
    elif data_file_format == "parquet":
        train = pd.concat((pd.read_parquet(f) for f in Path(train_data_dir_path).glob("*")))
        valid = pd.concat((pd.read_parquet(f) for f in Path(valid_data_dir_path).glob("*")))
    elif data_file_format == "pickle":
        train = pd.concat((pd.read_pickle(f) for f in Path(valid_data_dir_path).glob("*")))
        valid = pd.concat((pd.read_pickle(f) for f in Path(valid_data_dir_path).glob("*")))
    else:
        pass

    lgb_params = {
        "learning_rate": learning_rate,
        "max_depth": 2**7 - 1,
        "num_leaves": 7,
        "random_state": 42,
        "verbose": -1,
        "metric": valid_metrics,
    }

    if objective == "multiclass":
        lgb_params["objective"] = objective
        lgb_params["num_class"] = num_class
    else:
        lgb_params["objective"] = objective

    model = lgb.train(
        params=lgb_params,
        train_set=lgb.Dataset(data=train.drop(target_col, axis=1), label=train[target_col]),
        valid_names=["valid_sets"],
        valid_sets=[lgb.Dataset(data=valid.drop(target_col, axis=1), label=valid[target_col])],
        num_boost_round=10000,
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True), lgb.log_evaluation(1)],
    )

    # Dump the models
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    fire.Fire(main)
