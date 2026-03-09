import json
import logging
import os
from typing import List, Optional

import omegaconf
import requests
from pydantic import BaseModel

log = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


class VerificationResult(BaseModel):
    query_specs: Optional[List[str]]
    product_specs: Optional[List[str]]
    conflict: Optional[str]
    is_accurate: bool
    reformulated_query: Optional[str]


def query_ollama(config: omegaconf.dictconfig.DictConfig, prompt: str) -> dict:
    """
    Send a prompt to the Ollama server and return the response.

    Parameters
    ----------
    config : omegaconf.dictconfig.DictConfig
        The configuration object containing model and request settings.
    prompt: str
        The prompt to send to the model.

    Returns
    -------
    dict
        The model's response as a dictionary.

    Raises
    ------
    requests.ConnectionError
        If the Ollama server is not reachable.
    requests.HTTPError
        If the server returns an error response.
    ValueError
        If the model returns an empty response.
    pydantic.ValidationError
        If the model response does not match the expected schema.
    json.JSONDecodeError
        If the model response cannot be parsed as JSON.
    """
    # Build schema object from verification model and pass directly
    schema = VerificationResult.model_json_schema()
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": config.model.name,
            "prompt": prompt,
            "stream": False,
            "format": schema,  # Schema restricts output token decoding
            "options": {
                "num_predict": config.model.max_tokens,
                "temperature": config.model.temperature,
                "seed": config.model.seed,
            },
        },
        timeout=300,
    )
    response.raise_for_status()

    payload = response.json()
    response_text = payload.get("response") or payload.get("thinking", "")

    if not response_text:
        raise ValueError("Empty response from Ollama")

    # The ollama server should already respect our schema, but validate anyway
    try:
        vr = VerificationResult.model_validate_json(response_text)
    except Exception as exc:  # ValidationError, JSONDecodeError, etc.
        # Log the exception plus the entire text so you can inspect what the model
        # actually returned (use repr to make quotes/newlines visible).
        log.warning(
            f"Ollama response did not match schema ({exc}); "
            f"raw response:\n{repr(response_text)}"
        )
        # try to return a dict if possible, otherwise fall back to a marker
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as jexc:
            log.error(f"Unable to parse raw response as JSON: {jexc}")
            raise ValueError(
                "Ollama response did not match schema and could not be parsed as JSON"
            ) from exc

    # successful validation
    return vr.model_dump()
