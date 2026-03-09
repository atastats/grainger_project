import omegaconf
import pandas as pd


def build_product_desc(
    config: omegaconf.dictconfig.DictConfig, query_row: pd.Series
) -> str:
    """
    Build the product description string from the configured fields based on the
    prioritization in the prompt configuration.

    Parameters
    ----------
    config : omegaconf.dictconfig.DictConfig
        The configuration object containing the list of relevant product context fields.
    query_row : pd.Series
        The row of the DataFrame containing the product information.

    Returns
    -------
    str
        The formatted product description string.
    """
    # First, for brand and color prepend/append to title if not present.
    product_desc = query_row.get("product_title", "")
    # Bring brand into title if requested and not already present
    if product_desc and "product_brand" in config.prompt.product_context:
        brand = query_row.get("product_brand")
        # Check if na
        if pd.notna(brand):
            brand = str(brand).strip()
            if brand and brand.lower() not in product_desc.lower():
                # Prepend brand to title
                product_desc = f"{brand} {product_desc}"

    # Optionally append color in parentheses
    if product_desc and "product_color" in config.prompt.product_context:
        color = query_row.get("product_color")
        # Check if not na
        if pd.notna(color):
            color = str(color).strip()
            if color and color.lower() not in product_desc.lower():
                # Append color in parentheses
                product_desc = f"{product_desc} ({color})"

    product_desc = (
        "Product Title: " + product_desc
    )  # Label the title field for clarity in the prompt
    # Now we deal with all other context fields based on their ordering
    for key in config.prompt.product_context:
        if key in ["product_title", "product_color", "product_brand"]:
            continue  # We've already handled these fields
        value = query_row.get(key)
        if value is not None and pd.notna(value) and value != "":
            formatted_key = key.replace("_", " ").title()
            addition = f"{formatted_key}: {value}"
            # Check if adding this would exceed the max length
            if (
                len(product_desc) + len(addition)
                >= config.prompt.product_context_max_length
            ):
                # Append only the portion of addition that fits within the limit
                product_desc += f"\n{addition[:config.prompt.product_context_max_length - len(product_desc)]}"
                break  # Stop adding more context if we would exceed the limit
            product_desc += f"\n{addition}"
    return product_desc


def label_verification_prompt_templating(
    config: omegaconf.dictconfig.DictConfig,
    label_verification_prompt: str,
    query_row: pd.Series,
) -> str:
    """
    Template the label verification prompt with the query and product description.

    Parameters
    ----------
    config : omegaconf.dictconfig.DictConfig
        The configuration object containing the list of relevant product context fields.
    label_verification_prompt : str
        The label verification prompt template string with placeholders for templating.
    query_row : pd.Series
        The row of the DataFrame containing the query and product information for templating.

    Returns
    -------
    str
        The templated prompt string ready for Ollama LLM inference.
    """
    return label_verification_prompt.format(
        query=query_row["query"], product_desc=build_product_desc(config, query_row)
    )
