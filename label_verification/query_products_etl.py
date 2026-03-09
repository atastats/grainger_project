import logging

import omegaconf
import pandas as pd

log = logging.getLogger(__name__)


def query_products_etl(config: omegaconf.dictconfig.DictConfig) -> pd.DataFrame:
    """
    1. Loads the shopping queries dataset
    2. Performs the necessary joins to get the df_examples_products dataframe
    3. Filters the dataframe to only include the queries of interest specified in the config
    4. Returns the resulting dataframe for use in the label verification step.

    Parameters
    ----------
    config: omegaconf.dictconfig.DictConfig
        The configuration object containing the paths to the datasets and the list of queries of interest.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the joined and filtered data for use in the label verification step.
    """

    # Load the datasets
    log.info("Loading datasets...")
    df_examples = pd.read_parquet(config.data.examples_path)
    df_products = pd.read_parquet(config.data.products_path)
    log.info(f"Loaded {len(df_examples)} examples and {len(df_products)} products.")

    # Perform the joins detailed in the repo
    log.info("Performing joins...")
    df_examples_products = pd.merge(
        df_examples,
        df_products,
        how="left",
        left_on=["product_locale", "product_id"],
        right_on=["product_locale", "product_id"],
    )
    log.info(f"Joined dataframe has {len(df_examples_products)} rows.")

    # Filter to only the queries of interest as specified in the config
    log.info("Filtering to queries of interest...")
    df_examples_products = df_examples_products[
        (df_examples_products["query"].isin(config.data.queries_of_interest))
        & (df_examples_products["esci_label"] == "E")
    ]
    log.info(
        f"Found {len(df_examples_products)} rows for {len(config.data.queries_of_interest)} queries of interest."
    )

    # Confirm the filtering didnt remove all data
    if df_examples_products.empty:
        raise ValueError(
            f"No results found for queries of interest: {list(config.data.queries_of_interest)}"
        )

    return df_examples_products
