# ESCI Label Verification

## Approach

### Embeddings
Embeddings were my first instinct, but they fail to capture negation and subtle mismatches (e.g., "with" vs. "without" pillow shams).

### Named Entity Recognition (NER)
NER would allow explainable comparison of query and product specs. I struggled to find a strong domain model, and output formats are inconsistent without fine-tuning.

### Cross-encoder and NLI-style models
These models are better than embeddings for detecting conflicts (product type, brand, quantity, negation) because they jointly encode query and product text. However, they struggle with longer input strings and can't handle query reformulation without a generative step.

### LLM-based Verification and Reformulation
LLMs offer flexibility and step-by-step reasoning, allowing for accurate conflict detection and query reformulation. The model is prompted to reason through specifications, identify conflicts, and decide accuracy, minimizing hallucinations and improving reliability.

> **Note:** Fine-tuning is out of scope here due to time constraints.

## Prompt
```
You are a product relevance auditor. Your task is to check for explicit conflicts between a customer query and a product.

Instructions:
- Mark a product as inaccurate only if there is a direct contradiction between the query and product information.
- Missing product details are not conflicts.
- Extra product details or included items are not conflicts.
- Pay special attention to negations in the query (e.g., "without batteries"): if the product includes what the query excludes, that's a conflict.

For each example, follow these steps:
1. List the key specifications from the query as a list.
2. List the key specifications from the product as a list.
3. Identify any direct conflicts.
4. Decide if the match is accurate (true/false).
5. If inaccurate, reformulate the query so it accurately describes the conflicting product, not the original query.

Respond only in this JSON format:
{{
    "query_specs": [...],
    "product_specs": [...],
    "conflict": "brief explanation or null",
    "is_accurate": true or false,
    "reformulated_query": "new query for the conflicting product or null"
}}

Now apply the same rules to the following:
Query: {query}
{product_desc}
```

## Prompt Configuration
Based on some analysis of the data, 8,796 chars is the largest full product context length (all fields) in the dataset. For the subset of data we are concerned with, this maximum drops to 3,726 chars. I will implement truncation at 5,000 chars (~1200 tokens). We can reduce this to increase inference speed at the risk of losing some product context. 

The product information includes five keys:
- "product_title"   # Highest priority, will never be truncated (always < 400 chars)
- "product_bullet_point" # Second highest priority, but may be truncated
- "product_description"  # Third highest priority, but may be truncated
- "product_brand"  # Prepended to title if not already present in title
- "product_color"  # Appended to title if not already present in title

## Pre-Filtering
With the model running locally on CPU, every LLM call is expensive latency-wise. Before hitting the LLM, I apply two lightweight filters:

1. **Negation check:** If the query contains a negation word (`not`, `without`, `no`, `except`, `never`, `none`), skip filtering entirely and send straight to the LLM.
2. **Fuzzy token match:** Stop words and punctuation are removed from both the query and product title before computing `token_set_ratio`. Hyphens and underscores are replaced with spaces (e.g., `dewalt-8v` → `dewalt 8v`), and other punctuation is stripped. If the score is 100 (every query term is present in the product title), label as "E" without an LLM call.

This filters out **25% of records** with no model inference at all. The pre-filter is intentionally conservative — it only confirms obvious positive matches to avoid false positives.

## Pipeline
```
Query-Product Pair
       ↓
1. Query contains negation? → Send to LLM
       ↓
2. Fuzzy Token Set Ratio == 100? → Label "E" ✓
       ↓
3. Everything else → LLM verification call
```

## Design Decisions
- **Inference backend:** Ollama was chosen for ease of local deployment. The code is structured to allow easy swapping of the inference backend with minimal changes.
- **Model selection:** `Qwen3-4B` was selected based on various benchmark scores (particularly around instruction following — the primary requirement for reliable structured JSON output in this pipeline).
- **Serial inference:** LLM calls are made serially. In this Dockerized local Ollama setup, parallel requests offer limited throughput gains while increasing memory pressure. In a production environment with an optimized serving stack (e.g., vLLM, TensorRT, Bedrock), requests could be parallelized using `asyncio` + `httpx`.
- **Docker Compose:** The Ollama model runs in a separate container from the application, allowing for modularity, easy debugging, and model persistence across runs via a named volume.

## Prerequisites
- Docker Desktop configured with at least **8GB RAM** 
  - Tested and running on a 16GB M1 Pro with 8GB allocated to Docker
- A local copy of the [ESCI dataset](https://github.com/amazon-science/esci-data)

## Setup
1. Copy `.env.example` to `.env`
2. Fill in your local path to the esci-data directory (`DATA_DIR`)
3. Run `docker compose up --abort-on-container-exit`

## Runtime
The full pipeline takes approximately 10–30 minutes when run locally in Docker with Ollama on consumer hardware. Runtime is dominated by serial LLM inference on CPU.

## GPU Acceleration (Optional)
If you have an NVIDIA GPU, add the following to the `ollama` service in `docker-compose.yaml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## Output
Results are saved to `results.csv` in the project root directory after the pipeline completes.

## Results

The results table below will be updated with the final markdown output from pandas once the pipeline is run: