#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    run = wandb.init(job_type="job_t_cleaning")
    run.config.update(args)



    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info(f"Downloaded input artifact to {artifact_local_path}")

    # Read the input data
    df = pd.read_csv(artifact_local_path)

    # Basic data cleaning: Remove outliers
    min_price = args.min_price
    max_price = args.max_price

    logger.info(f"Removing outliers outside the range {min_price} to {max_price}")
    df = df[df['price'].between(min_price, max_price)].copy()

    # Drop rows with missing values
    logger.info("Dropping rows with missing values")
    df.dropna(inplace=True)

    # Save cleaned data to a CSV file
    cleaned_data_path = "clean_sample.csv"
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    df.to_csv(cleaned_data_path, index=False)
    logger.info(f"Cleaned data saved to {cleaned_data_path}")

    # Log the cleaned data to Weights & Biases
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(cleaned_data_path)
    run.log_artifact(artifact)
    logger.info("Cleaned data artifact logged to Weights & Biases")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to consider",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to consider",
        required=True
    )


    args = parser.parse_args()

    go(args)
