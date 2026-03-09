import logging
import os
import sys

import hydra

from .query_products_etl import query_products_etl
from .verify_labels import verify_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-6s [%(filename)s:%(lineno)d] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# Disable pager so fire help output is printed to stdout instead of opening a pager
os.environ["PAGER"] = "cat"


def main():
    log.info("Starting label verification process...")
    # Initialize Hydra and load the config
    with hydra.initialize(config_path=".", version_base="1.1"):
        config = hydra.compose(config_name="config")

        # Do the ETL to get the joined and filtered data for use in the label verification step
        df_examples_products = query_products_etl(config)

        # Run the label verification step to get the dataframe of examples that should be flagged for LLM analysis
        verify_labels(config, df_examples_products)


if __name__ == "__main__":
    main()
