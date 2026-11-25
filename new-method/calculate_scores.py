"""
This is the final method used in the thesis to score masked sentences for the level
of humanness of the masked token.

The method uses "distilbert/distilbert-base-uncased" as the masked language model,
and sentence-transformers/all-MiniLM-L6-v2 as the embedding model, and topk=50
as the number of most probable tokens to calculate humanness similarity per masked token for.

Claude (GUI version, free tier) has been utilized in writing the method.
"""

# ── 0. Imports ─────────────────────────────────────────────────────────────────
import os
os.environ["HF_HOME"] = "/projappl/project_xxxxxxx/hf_home"

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from transformers import pipeline, Pipeline, AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
from typing import Iterable, overload, Literal
from collections import OrderedDict
from pathlib import Path
import re
import csv
import gc

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s (line %(lineno)d) | %(levelname)s: %(message)s"
    )

# ── 1. Constants ────────────────────────────────────────────────────────────────

_prototype_cache = {}

ANIMATE_DESCRIPTIONS = [
    "a living person, human being, or individual who can act and think",
    "a human who makes decisions and has intentions",
    "someone who thinks, feels, and can take action",
    "a name of a person or a personal pronoun referring to a human",
    "a person with a professional role such as a doctor, scientist, or engineer",
    "a human subject or participant in a study or experiment",
    "an individual who experiences emotions and has subjective awareness",
    "a named historical or contemporary figure such as an author or inventor",
    "a collective group of humans such as a team, committee, or community",
    "a pronoun referring to one or more people such as I, you, he, she, or we",
    "a human agent who performs actions and bears responsibility",
    "a living creature capable of perception and voluntary movement",
    "an animate being with a social role or identity",
    "a person referred to by their job title, role, or name"
]

INANIMATE_DESCRIPTIONS = [
    "a non-living machine, device, tool, or system",
    "a piece of software or automated system without consciousness",
    "an object or artifact that cannot think or feel",
    "a technical system or piece of equipment",
    "an abstract concept or a thing that does not have a physical form",
    "a physical object with no capacity for thought or intention",
    "an automated process or algorithm that executes without awareness",
    "a product, platform, or application built by humans",
    "a natural or man-made structure such as a building or infrastructure",
    "a measurement, metric, or quantitative value",
    "a document, report, or piece of written output",
    "an institution or organization treated as a non-human entity",
    "a chemical substance, material, or physical phenomenon",
    "a vehicle, vessel, or means of transport",
    "something that is acted upon rather than acting of its own will",
    "a pronoun referring to an inanimate object such as it"
]

CLASS_DESCRIPTIONS = [ANIMATE_DESCRIPTIONS, INANIMATE_DESCRIPTIONS]
CLASS_NAMES_DEFAULT = ["animate", "inanimate"]

# Cache the static embeddings in format {(model_id, prefix, template, token): np.ndarray}
_static_emb_cache: dict[tuple[str, str, str, str], np.ndarray] = {}

MAX_CACHED_MODELS = 5
# Cache the models to avoid redownloads
_model_cache = OrderedDict()

MLM_ID = "distilbert/distilbert-base-uncased"
EMB_ID = "sentence-transformers/all-MiniLM-L6-v2"


# ── 2. Model registry ───────────────────────────────────────────────────────────
# Pooling strategy and any required input prefix, sourced from official model cards.

EMB_MODEL_CONFIGS = {
    # mean pool — no prefix
    "sentence-transformers/all-MiniLM-L6-v2":  {"pooling": "mean", "query_prefix": "", "doc_prefix": "", "static_template": "What is the definition of [PLACEHOLDER]?"},
    # mean pool - prefix
    "nomic-ai/modernbert-embed-base":          {"pooling": "mean", "query_prefix": "search_query: ", "doc_prefix": "search_document: ", "static_template": "What is the most accurate description of [PLACEHOLDER]?"},
    # CLS pool — no prefix
    "BAAI/bge-base-en-v1.5":                   {"pooling": "cls",  "query_prefix": " ", "doc_prefix": "", "static_template": "Can you define [PLACEHOLDER]?"},
    # last token pool - prefix
    "jinaai/jina-embeddings-v5-text-nano":     {"pooling": "last", "query_prefix": "Query: ", "doc_prefix": "Document: ", "static_template": "Can you define [PLACEHOLDER]?"},
}

def get_emb_config(model_id: str) -> dict:
    if model_id in EMB_MODEL_CONFIGS:
        return EMB_MODEL_CONFIGS[model_id]
    # Fallback: warn and use mean pooling with no prefix nor template
    logger.warning(f"""  ⚠ '{model_id}' not in registry — defaulting to mean pooling, no prefix, no template.
                       Check the model card and add it to EMB_MODEL_CONFIGS if needed.""")
    return {"pooling": "mean", "query_prefix": "", "doc_prefix": "", "static_template": ""}


# ── 3. Config dataclass ──────────────────────────────────────────────────────

@dataclass
class Config:
    # MLM
    mlm_pipe:             Pipeline  # HF fill-mask pipeline — to automatically handle model-specific settings
    mlm_model_id:         str

    # Embedding model
    emb_tokenizer:        PreTrainedTokenizer
    emb_model:            PreTrainedModel
    emb_model_id:         str
    emb_config:           dict  # {"pooling": "mean"|"cls"|"last", "query_prefix": str, "doc_prefix": str, "static_template": str}

    # Prototype setup
    class_descriptions:   list[list[str]]
    class_names:          list[str]
    class_protos:         np.ndarray = field(init=False, default_factory=lambda: np.empty(0)) # (n_classes, hidden_dim)

    mask_token:           str = field(init=False, default="") # [MASK] for BERTs, <mask> for RoBERTa
    template_placeholder: str = "[PLACEHOLDER]"

    def build_prototype_vectors(self):
        """
        Embed each class description list, average per class, re-normalize.
        Returns (class_protos, class_names).
        """

        if not self.emb_model_id in _prototype_cache:

            logger.info("Building prototype vectors…")
            protos = []
            for descriptions in self.class_descriptions:
                embs  = embed_texts_batch(descriptions, self, "doc_prefix") # (n_desc, hidden_dim)
                proto = embs.mean(axis=0)
                proto = proto / (np.linalg.norm(proto) + 1e-12) # re-normalize back to unit length
                protos.append(proto)

            _prototype_cache[self.emb_model_id] = np.array(protos)
            logger.info("Prototype vectors cached.")

        return _prototype_cache[self.emb_model_id] # (n_classes, hidden_dim)

    def __post_init__(self):
        self.class_protos = self.build_prototype_vectors()
        logger.info("Prototype vectors loaded.")

        self.mask_token = self.mlm_pipe.tokenizer.mask_token
        logger.info(f"Model uses {self.mask_token} as mask_token.")


# ── 4. Pooling functions ───────────────────────────────────────────────────────

def normalize_and_move(pooled: torch.Tensor) -> np.ndarray:
    normalized = F.normalize(pooled)
    return normalized.detach().cpu().numpy()

def pool_batch(hidden: torch.Tensor,
               attention_mask: torch.Tensor,
               strategy: str) -> np.ndarray:
    """
    Pool a batch of hidden states to one vector per sequence.
    hidden: (batch, seq_len, hidden_dim)
    Returns: (batch, hidden_dim) numpy array, L2-normalized per row.
    """
    if strategy == "cls":
        pooled = hidden[:, 0, :]                               # (batch, seq_len, hidden_dim)
    elif strategy == "mean":
        mask   = attention_mask.unsqueeze(-1).float()          # (batch, seq_len, 1)
        summed = (hidden * mask).sum(dim=1)                    # (batch, hidden_dim)
        counts = mask.sum(dim=1).clamp(min=1e-12)              # (batch, 1)
        pooled = summed / counts                               # (batch, hidden_dim)
    elif strategy == "last":
        sequence_lengths = attention_mask.sum(dim=1) - 1       # (batch,)
        batch_idx        = torch.arange(hidden.size(0), device=hidden.device)
        pooled           = hidden[batch_idx, sequence_lengths] # (batch, hidden_dim)
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy!r}")

    return normalize_and_move(pooled)

def embed_texts_batch(texts: list[str], config: Config, prefix_key: str) -> np.ndarray:
    """
    Embed a list of texts in one batched forward pass.
    Applies model-specific prefix before encoding.
    Returns: (n_texts, hidden_dim) numpy array.
    """
    prefix        = config.emb_config[prefix_key]
    strategy      = config.emb_config["pooling"]
    prefixed      = [prefix + t for t in texts] if len(prefix) > 0 else texts

    device        = next(config.emb_model.parameters()).device
    inputs        = config.emb_tokenizer(
                        prefixed,
                        return_tensors="pt",
                        truncation=True,
                        max_length=config.emb_model.config.max_position_embeddings,
                        padding=True
                    )
    inputs        = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        hidden    = config.emb_model(**inputs).last_hidden_state

    return pool_batch(hidden, inputs["attention_mask"], strategy)

def embed_texts_batch_cached(tokens: list[str], config: Config, prefix_key: str, template: str = "") -> np.ndarray:
    """
    Embed tokens statically, using cache for tokens seen before.
    Returns: (n_tokens, hidden_dim)
    """
    cache_key            = config.emb_model_id
    prefix               = config.emb_config[prefix_key]
    template_placeholder = config.template_placeholder
    uncached = [t for t in tokens if (cache_key, prefix, template, t) not in _static_emb_cache]

    if uncached:
        if len(template) > 0:
            embs = embed_texts_batch([template.replace(template_placeholder, t) for t in uncached], config, prefix_key) # one batch for new tokens
        else:
            embs = embed_texts_batch(uncached, config, prefix_key)
        for token, emb in zip(uncached, embs):
            _static_emb_cache[(cache_key, prefix, template, token)] = emb

    return np.array([_static_emb_cache[(cache_key, prefix, template, t)] for t in tokens])


# ── 5. Initialization ──────────────────────────────────────────────────────────

def load_pipeline(mlm_model_id:  str,
                  emb_model_id:  str,
                  class_descriptions: list[list[str]] = CLASS_DESCRIPTIONS,
                  class_names:   list[str] = CLASS_NAMES_DEFAULT,
                  device: str = "cuda") -> Config:
    """
    Load both models once, build prototype vectors, return a Config object
    that stores all the necessary information.
    """

    @overload
    def get_or_load(model_id: str, model_type: Literal["mlm"]) -> Pipeline: ...

    @overload
    def get_or_load(model_id: str, model_type: Literal["emb"]) -> tuple[PreTrainedTokenizer, PreTrainedModel]: ...

    def get_or_load(model_id: str, model_type: str) -> Pipeline | tuple[PreTrainedTokenizer, PreTrainedModel]:
        if model_id in _model_cache:
            _model_cache.move_to_end(model_id)  # mark as recently used
            return _model_cache[model_id]

        if len(_model_cache) >= MAX_CACHED_MODELS:
            evicted_id, _ = _model_cache.popitem(last=False)  # remove oldest
            logger.info(f"Evicting {evicted_id} from cache")

        if model_type == "mlm":
          logger.info(f"Loading MLM: {mlm_model_id}")
          _model_cache[model_id] = pipeline(
                                    "fill-mask",
                                    model=model_id,
                                    device=device,
                                    tokenizer_kwargs={
                                        "truncation": True
                                    }
                                  )
          return _model_cache[model_id]

        elif model_type == "emb":
            logger.info(f"Loading embedding model: {emb_model_id}")
            emb_tokenizer = AutoTokenizer.from_pretrained(emb_model_id,
                                                          local_files_only=True)
            emb_model     = AutoModel.from_pretrained(emb_model_id,
                                                      local_files_only=True,
                                                      attn_implementation="sdpa")
            emb_model.eval()
            if device == "cuda" and torch.cuda.is_available():
                emb_model = emb_model.cuda()
            _model_cache[emb_model_id] = (emb_tokenizer, emb_model)
            return (emb_tokenizer, emb_model)

        else: raise ValueError(model_type)


    mlm_pipe                 = get_or_load(mlm_model_id, "mlm")
    emb_config               = get_emb_config(emb_model_id)
    emb_tokenizer, emb_model = get_or_load(emb_model_id, "emb")

    config = Config(
        mlm_pipe=mlm_pipe,
        mlm_model_id=mlm_model_id,
        emb_tokenizer=emb_tokenizer,
        emb_model=emb_model,
        emb_model_id=emb_model_id,
        emb_config=emb_config,
        class_descriptions=class_descriptions,
        class_names=class_names
    )
    logger.info("Models loaded.\n")
    return config


# ── 6. MLM prediction ──────────────────────────────────────────────────────────

def predict_mask(sentences: Iterable[str],
                 config: Config,
                 top_k: int = 50,
                 batch_size = 128):
    """
    Return top-k candidate tokens and their MLM probabilities.
    Uses HF fill-mask pipeline in order to handle mask token and
    tokenization correctly for any supported MLM (BERT, RoBERTa, ModernBERT, etc.).
    """

    if isinstance(sentences, str):
      sentences = [sentences]

    def center_on_mask(text, max_length=config.mlm_pipe.model.config.max_position_embeddings):
        """
        Returns a possibly-truncated string, re-centered around the mask token,
        using character offsets.
        """
        encoding = config.mlm_pipe.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,  # only consider content tokens for centering
            truncation=False,
        )
        input_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        if config.mlm_pipe.tokenizer.mask_token_id not in input_ids:
            raise ValueError("No mask token found in text")
        
        # Reserve space for special tokens ([CLS], [SEP]) the pipeline will add
        budget = max_length - 2

        if len(input_ids) < budget:
            return text

        mask_idx = input_ids.index(config.mlm_pipe.tokenizer.mask_token_id)

        half = budget // 2
        start_tok = max(0, mask_idx - half)
        end_tok = min(len(input_ids), start_tok + budget)

        # Map token window back to character offsets in the original string
        start_char = offsets[start_tok][0]
        end_char = offsets[end_tok - 1][1]

        return text[start_char:end_char]

    def preprocess_stream(sentences):
        for sentence in sentences:
            yield center_on_mask(sentence, max_length=config.mlm_pipe.model.config.max_position_embeddings)

    results = config.mlm_pipe(preprocess_stream(sentences), top_k=top_k, batch_size=batch_size)

    for result in results:
        if result and isinstance(result[0], list):
            logger.warning("Multiple mask tokens detected in an input; skipping.")
            yield None
            continue
        tokens    = [r["token_str"].strip() for r in result]
        mlm_probs = np.array([r["score"] for r in result])
        yield {"tokens": tokens, "mlm_probs": mlm_probs}


# ── 7. Scoring ─────────────────────────────────────────────────────────────────

def score_candidates(embeddings: np.ndarray,
                     config: Config) -> np.ndarray:
    """
    Compute animacy class similarities for each candidate embedding.
    Get the difference between the similarities by subtracting the similarity
    to the inanimate prototype from the similarity to the animate prototype.
    Positive values indicate the embedding is closer to the animate prototype,
    and negative values indicate the embedding is closer to the inanimate prototype.
    embeddings: (n_candidates, hidden_dim)
    """
    # (n_candidates, hidden_dim) @ (hidden_dim, n_classes) → (n_candidates, n_classes)
    sims  = embeddings @ config.class_protos.T
    sim_diff = sims[:, 0] - sims[:, 1]

    return sim_diff

def scores_from(sim_diff: np.ndarray,
                mlm_probs: np.ndarray,
                top_k: int) -> float:
    sim_diff   = sim_diff[:top_k]
    probs      = mlm_probs[:top_k]
    total_prob = probs.sum()

    return float(np.dot(sim_diff, probs) / total_prob) \
            if total_prob > 0 else 0.0

# ── 9. Full classify_mask ──────────────────────────────────────────────────────

def classify_mask(sentences: Iterable[str],
                  config: Config,
                  top_k: int = 50,
                  batch_size: int = 128):

    for result in predict_mask(sentences, config, top_k=top_k, batch_size=batch_size):
        if not result:
            yield float("nan")
            continue

        embeddings = embed_texts_batch_cached(result["tokens"], config, "query_prefix")
        sim_diff = score_candidates(embeddings, config)
        score = scores_from(sim_diff, result["mlm_probs"], top_k)

        yield score

# ── 10. Example usage ──────────────────────────────────────────────────────────

def get_short_name(model_id: str) -> str:
  return model_id.rsplit("/", 1)[-1]

EXAMPLES = [
    "[MASK] reached optimum proficiency in tasks after just one repetition.",
    "[MASK] invented the telephone.",
    "[MASK] was installed on the server last night.",
    "[MASK] decided to escalate the issue to management.",
    "[MASK] crashed unexpectedly during the experiment.",
    "[MASK] wrote a letter to the editor.",
    "After the update, [MASK] didn't work as it used to.",
    "She felt seen while talking to [MASK].",
    "A deceptive [MASK] tried to fool the human.",
    "We think the output of [MASK] is hallucinated.",
    "We think [MASK] hallucinates.",
    "[MASK] told the doctor about the hallucinations they had.",
    "Hallucination is a know quirk in [MASK].",
    "I am using [MASK] as a tutor.",
    "I am using an [MASK] tutor."
  ]

def run_examples(
  examples: list[str] = EXAMPLES,
  mlm_model_id = MLM_ID,
  emb_model_id = EMB_ID,
  return_outputs: bool = False
):
    test_config = load_pipeline(
        mlm_model_id = mlm_model_id,
        emb_model_id = emb_model_id
    )

    if "roberta" in test_config.mlm_model_id:
        examples = [ex.replace("[MASK]", "<mask>") for ex in examples]

    outputs = []
    for example, result in zip(examples, classify_mask(examples, test_config)):
      outputs.append(result)
      logger.info(example)
      logger.info(result)

    if return_outputs:
      return outputs

# ---------------------------------------------

SENTENCE_DIR = "/scratch/project_xxxxxxx/results-tests/csv-data/masked-sentences/xml-fulltexts/"
SCORES_DIR = "/scratch/project_xxxxxxx/results-tests/scores/new-method/xml-fulltexts/"

SHARD_NUM_PATTERN = re.compile(r"shard_(\d{3})")

REPORT_INTERVAL = 1000
BATCH_SIZE = 128

def get_filepaths(dir: str, glob_pattern: str):
    return sorted(Path(dir).glob(glob_pattern))

def get_unprocessed(
          csv_dir: str,
          scores_dir: str,
          csv_glob_pattern: str = "masked-sentences-xml-shard_*.csv",
          scores_glob_pattern: str = "scores_new_method_xml-shard_*.csv"
    ) -> list[tuple[Path, Path, dict | None]]:

    def get_last_row(filepath, id_fields: list[str] = ["i", "paper_id", "sent_id", "match_id"]):
        last_row = {}
        i = -1
        with open(filepath) as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Due to an earlier bug, a hacky fix for checking that current row is not a column name row
                # Do not increment i if it is
                if row.get("id", "") == "id" and row["sent_id"] == "sent_id":
                    continue
                i += 1
                last_row = row
                last_row["i"] = i
        return {key: last_row.get(key) for key in id_fields}
    
    def get_progress_str(last_csv: dict, last_scores: dict):
        done = last_scores["i"]
        total = last_csv["i"]
        return f"{done}/{total} ({done/total:.2%}) sentences processed"
    
    # Collect the names of the existing score files
    score_files = get_filepaths(scores_dir, scores_glob_pattern)
    progress = {}
    for score_filepath in score_files:
        scores_last_row = get_last_row(score_filepath) 
        match = SHARD_NUM_PATTERN.search(score_filepath.name)
        if match:
            progress[match.group(1)] = scores_last_row
    
    # Collect the names of the csv files that are missing scores
    csv_files = get_filepaths(csv_dir, csv_glob_pattern)
    unprocessed = []
    for csv_filepath in csv_files:
        match = SHARD_NUM_PATTERN.search(csv_filepath.name)
        if match:
            shard_num = match.group(1)
            # If matching shard is not found, add the file to the list of unprocessed
            if shard_num not in progress:
                score_filepath = Path(scores_dir, f"{scores_glob_pattern.replace('*', shard_num)}")
                unprocessed.append((csv_filepath, score_filepath, None))
                logger.info(f"Added {csv_filepath} to the unprocessed list")

            # If matching shard is found, check if the last row has matching identifications
            else:
                csv_last_row = get_last_row(csv_filepath)
                # If the last rows do not match, add to the list of unprocessed
                if progress[shard_num] != csv_last_row:
                    score_filepath = Path(scores_dir, f"{scores_glob_pattern.replace('*', shard_num)}")
                    unprocessed.append((csv_filepath, score_filepath, progress[shard_num]))
                    logger.info(f"Added {progress[shard_num]} to the unprocessed list - {get_progress_str(csv_last_row, progress[shard_num])}")
                # Else, no need to add the file to the list
                else:
                    logger.info(f"Results for {csv_filepath} have been finished")
    
    # Return the filepaths as tuples of (read_from, write_to, last_row_ids)
    return unprocessed

def get_rows_and_sentences(reader: csv.DictReader, last_row: dict | None, column_name: str = "masked_sentence_minimal"):
    start_found = False
    if not last_row:
        start_found = True
    else:
        last_row_tuple = (last_row["paper_id"], last_row["sent_id"], last_row["match_id"])

    rows = []
    sentences = []
    
    for row in reader:
        if not start_found:
            current_row_tuple = (row["paper_id"], row["sent_id"], row["match_id"])
            if last_row_tuple == current_row_tuple:
                start_found = True
                continue # Continue from the next row as this one is already processed
        else:
            rows.append(row)
            sentences.append(row[column_name])
    if not start_found:
        raise ValueError(f"Couldn't find the start position")
    
    assert len(rows) == len(sentences), f"{len(rows)} rows vs {len(sentences)} sentences"
    return rows, sentences

def get_scores(sentence_filename: str | Path, score_filename: str | Path, last_row: dict | None, config: Config):
    
    with open(sentence_filename, newline="") as infile:

        if last_row:
            logger.info(f"Continuing from {last_row}")

        reader = csv.DictReader(infile)
        column_names = reader.fieldnames
        if not column_names:
           raise ValueError(f"Couldn't read header row from {sentence_filename}")
        column_names = list(column_names) + ["score"]

        rows, sentences = get_rows_and_sentences(reader, last_row)

        # Open scores file in append mode in case the progress is being returned to
        with open(score_filename, "a", newline="") as outfile:
 
            writer = csv.DictWriter(outfile, fieldnames=column_names)
            # Only write the header if not resuming
            if not last_row:
                writer.writeheader()

            for i, (row, score) in enumerate(zip(rows, classify_mask(sentences, config, batch_size=BATCH_SIZE)), start=1):
                
                row["score"] = score
                writer.writerow(row)
            
                if i % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                if i % REPORT_INTERVAL == 0:
                    outfile.flush()
                    logger.info(f"Scores from {i} sentences logged")
                
# ==================== MAIN ====================

def main():

    unprocessed = get_unprocessed(SENTENCE_DIR, SCORES_DIR)
    logger.info(f"Found {len(unprocessed)} CSV files with incomplete progress")

    config = load_pipeline(
        mlm_model_id = MLM_ID,
        emb_model_id = EMB_ID
    )

    for i, (read_from, write_to, last_row) in enumerate(unprocessed, start=1):       

        logger.info(f"Processing file {read_from.name} ({i}/{len(unprocessed)})")

        try:
            logger.info("Starting to calculate scores for remaining sentences...")
            get_scores(read_from, write_to, last_row, config)

            logger.info(f"Scores for {read_from.name} written to {write_to.name}!")

        except Exception as e:
            logger.exception(f"{type(e).__name__}: {e}")
    
    logger.info("All done!")

if __name__ == "__main__":
    main()
