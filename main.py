import json
import mlflow
import tempfile
import os
import wandb
import hydra
import subprocess
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
             _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
            "main",
            parameters={
                "input_artifact": "sample.csv:latest",
                "output_artifact": "clean_sample.csv",
                "output_type": "clean_sample",
                "output_description": "Data with outliers and null values removed",
                "min_price": config['etl']['min_price'],
                "max_price": config['etl']['max_price']
            },
        )

        if "data_check" in active_steps:
            _ = mlflow.run(
                    os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                    "main",
                    parameters={
                        "csv": "clean_sample.csv:latest",
                        "ref": "clean_sample.csv:reference",
                        "min_price": config["etl"]["min_price"],
                        "max_price": config["etl"]["max_price"],
                        "kl_threshold": config["data_check"]["kl_threshold"]
                    },
                )

        if "train_val_test_split" in active_steps:
            _ = mlflow.run(
                f"{config.main.components_repository}/train_val_test_split",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": str(config.modeling.test_size),
                    "random_seed": str(config.modeling.random_seed),
                    "stratify_by": config.modeling.stratify_by,
                },
            )

        if "train_random_forest" in active_steps:
            # Flatten rf_config
            rf_config = dict(config["modeling"]["random_forest"].items())

        

            # Serialize RandomForest configuration to rf_config.json
            rf_config_path = os.path.abspath("rf_config.json")
            with open(rf_config_path, "w+") as fp:
                json.dump(rf_config, fp)

            # Execute train_random_forest/run.py using subprocess
            subprocess.run([
                "python", os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest", "run.py"),
                "--trainval_artifact", "clean_sample.csv:latest",
                "--val_size", str(config.modeling.test_size),
                "--random_seed", str(config.modeling.random_seed),
                "--stratify_by", config.modeling.stratify_by,
                "--rf_config", rf_config_path,
                "--max_tfidf_features", str(config.modeling.max_tfidf_features),
                "--output_artifact", "random_forest_export"
            ])
        if "test_regression_model" in active_steps:
            _ = mlflow.run(
                f"{config.main.components_repository}/test_regression_model",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset": "test_data.csv:latest"
                },
            )


if __name__ == "__main__":
    go()
