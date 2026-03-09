import logging

import omegaconf
import pandas as pd

from .infer_ollama import query_ollama
from .pre_filters import fuzzy_token_set_pre_filter, negation_pre_filter
from .prompt_templating import label_verification_prompt_templating

log = logging.getLogger(__name__)


def verify_row(
    row: pd.Series,
    config: omegaconf.dictconfig.DictConfig,
    label_verification_prompt: str,
) -> pd.Series:
    """
    Verify a single row label and return the verification result using a single prompt.

    Parameters
    ----------
    row : pd.Series
        The row of the DataFrame to be verified.
    config : omegaconf.dictconfig.DictConfig
        The configuration object containing model and prompt settings.
    label_verification_prompt : str
        The label verification prompt template.

    Returns
    -------
    pd.Series
        A Series containing the verification results for the row.
    """
    # Fast path - fuzzy match is strong enough, no LLM needed
    if not negation_pre_filter(row["query"]) and fuzzy_token_set_pre_filter(
        row["query"], row["product_title"]
    ):
        log.info(
            f"Fuzzy match passed for query: '{row['query']}' "
            f"and product: '{row['product_title']}'. Skipping LLM."
        )
        return pd.Series(
            {
                "query_id": row["query_id"],
                "product_id": row["product_id"],
                "query": row["query"],
                "product_title": row["product_title"],
                "product_description": row["product_description"],
                "product_bullet_point": row["product_bullet_point"],
                "accurate_match": True,
                "reason": "Fuzzy match threshold met",
                "reformulated_query": None,
            }
        )

    # LLM verification needed (single prompt)
    templated_verification_prompt = label_verification_prompt_templating(
        config, label_verification_prompt, row
    )
    ollama_resp = query_ollama(config, templated_verification_prompt)

    # Parse keys from LLM response
    accurate = ollama_resp.get("is_accurate", False)
    reason = ollama_resp.get("conflict", "No conflict explanation provided")
    reformulated_query = ollama_resp.get("reformulated_query", None)

    if isinstance(accurate, str):
        accurate = accurate.lower() == "true"

    return pd.Series(
        {
            "query_id": row["query_id"],
            "product_id": row["product_id"],
            "query": row["query"],
            "product_title": row["product_title"],
            "product_description": row["product_description"],
            "product_bullet_point": row["product_bullet_point"],
            "accurate_match": accurate,
            "reason": reason,
            "reformulated_query": reformulated_query,
        }
    )


def verify_labels(
    config: omegaconf.dictconfig.DictConfig, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Verify labels for a DataFrame of queries and products.
    Parameters
    ----------
    config : omegaconf.dictconfig.DictConfig
        The configuration object containing model and prompt settings.
    df : pd.DataFrame
        The DataFrame containing the queries and products to be verified.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the verification results for each row.
    """
    log.info("Loading prompts...")
    with open(config.prompt.label_verification_prompt_path, "r", encoding="utf-8") as f:
        label_verification_prompt = f.read()

    # Apply the verification function to each row in the DataFrame
    results = df.apply(
        lambda row: verify_row(row, config, label_verification_prompt),
        axis=1,
    ).reset_index(drop=True)

    log.info(
        f"Label verification completed. "
        f"{len(results[results['reason'] == 'Fuzzy match threshold met'])} fuzzy confirmed, "
        f"{len(results[results['accurate_match'] == True])} accurate total, "
        f"{len(results[results['reformulated_query'].notna()])} reformulated."
    )

    # Save results to CSV
    output_path = config.data.output_path
    results.to_csv(output_path, index=False)
    log.info(f"Results saved to {output_path}")

    return results
