import json
import logging
import os

import omegaconf
import requests

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def query_ollama(config: omegaconf.dictconfig.DictConfig, prompt: str) -> dict:
    """
    Send a prompt to the Ollama server and return the response.

    Parameters
    ----------
    prompt: str
        The prompt to send to the model.

    Returns
    -------
    str
        The model's response as a string.

    Raises
    ------
    requests.ConnectionError
        If the Ollama server is not reachable.
    requests.HTTPError
        If the server returns an error response.
    ValueError
        If the model returns an empty response.
    json.JSONDecodeError
        If the model response cannot be parsed as JSON.
    """
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": config.model.name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
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

    return json.loads(response_text)
