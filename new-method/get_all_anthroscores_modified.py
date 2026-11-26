# SPDX-License-Identifier: BSD 2-Clause

# Copyright (c) Myra Cheng 2023, AnthroScore
# Copyright (c) 2026, Siiri Käpyaho (modifications)
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# ----------------------------------------

# This file includes a modified version of AnthroScore (https://github.com/myracheng/anthroscore).

# Changes include making the mechanism more robust by decreasing repetition and
# saving the progress incrementally at every 1000th item. This version only
# calculates scores for pre-masked sentences instead of sentence masking being
# a part of the process.

# Checkpoint logic is not part of the original implementation.

"""
Scores masked sentences according to how likely the masked phrase is to be replaced
either with human pronouns or non-human pronouns.
"""

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s (line %(lineno)d) | %(levelname)s: %(message)s"
    )

from dataclasses import dataclass, asdict, field
import re
import pandas as pd
from pathlib import Path
import json
import csv
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
import numpy as np
import scipy
import gc

# ========== LOAD MODEL ==========

model = RobertaForMaskedLM.from_pretrained('roberta-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # type: ignore[arg-type]
logger.info(f"BERT model loaded on {device}")
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# ========== SET CONSTANTS ==========

TERMS = ['he', 'she', 'her', 'him', 'He', 'She', 'Her', 'it', 'its', 'It', 'Its']
# index    0      1      2      3     4      5      6     7      8     9     10
FIRST_NONHUMAN_INDEX = TERMS.index("it")
TARGET_INDS = [tokenizer.get_vocab()[x] for x in TERMS]

def get_filepaths(dir: str, glob_pattern: str):
    return sorted(Path(dir).glob(glob_pattern))

SENTENCE_DIR = "/scratch/project_xxxxxxx/results-tests/csv-data/"
ALL_SENTENCE_FILES = get_filepaths(SENTENCE_DIR, "sentences-shard_*.csv")
SCORES_DIR = "/scratch/project_xxxxxxx/results-tests/scores/"
CHECKPOINT_FILE = "/scratch/project_xxxxxxx/results-tests/checkpoint/anthroscore-checkpoint.json"

SHARD_NUM_PATTERN = re.compile(r"shard_(\d{3})")

# ========== PREPARE CHECKPOINT LOGIC =========

def get_shard_info(filepaths: list[Path]) -> list[tuple[str, str]]:
    shards = []
    for filepath in filepaths:
        match = SHARD_NUM_PATTERN.search(filepath.name)
        if match:
            shards.append((str(filepath), match.group(1)))
    return shards

def get_line_count_info(filename: str | Path) -> int:
    with open(filename) as file:
        return sum(1 for line in file if line.strip())
    
def get_file_info(dir: str, glob_pattern: str) -> dict[str, dict[str, str | int]]:
    filepaths = get_filepaths(dir, glob_pattern)
    shards = get_shard_info(filepaths)
    full_info = {shard_num: {"filepath": str(filepath), "line_count": get_line_count_info(filepath)} for filepath, shard_num in shards}
    return full_info

@dataclass
class Checkpoint:
    processed: list[str] = field(default_factory=list)
    interrupted: list[list] = field(default_factory=list)
    current_file: str = ""
    current_row_index: int = 0

    def save(self):
        with open(CHECKPOINT_FILE, "w") as file:
            json.dump(asdict(self), file, indent=2)
    
    def mark_as_processed(self):
        self.processed.append(self.current_file)
        # Remove from interrupted in case the file was started from there
        self.interrupted = [item for item in self.interrupted if item[0] != self.current_file]
        self.current_file = ""
        self.current_row_index = 0
        self.save()
    
    def mark_as_interrupted(self):
        # Remove the old entry of the same file if present
        self.interrupted = [item for item in self.interrupted if item[0] != self.current_file]
        self.interrupted.append([self.current_file, self.current_row_index])
        self.current_file = ""
        self.current_row_index = 0
        self.save()

    def sync_progress(self, filepath: str, index: int):
        self.current_file = filepath
        self.current_row_index = index


def initialize_checkpoint() -> Checkpoint:
    sentence_info = get_file_info(SENTENCE_DIR, "sentences-shard_*.csv")
    sentence_scores_info = get_file_info(SCORES_DIR, "sentence_scores-shard_*.csv")

    processed = []
    interrupted = []
    for shard_num, info in sentence_info.items():
        if shard_num in sentence_scores_info:
            if info["line_count"] != sentence_scores_info[shard_num]["line_count"]:
                interrupted.append((info["filepath"], sentence_scores_info[shard_num]["line_count"]))
            else:
                processed.append(info["filepath"])
    checkpoint = Checkpoint(processed=processed, interrupted=interrupted)
    checkpoint.save()
    return checkpoint

def load_checkpoint(filename: str | Path) -> Checkpoint:
    if Path(filename).exists():
        with open(filename) as file:
            data = json.load(file)
            return Checkpoint(**data)
    else:
        return initialize_checkpoint()

def get_triple(filename: str) -> tuple[Path, Path, Path]:
    sentence_file_template = "sentences-shard_*.csv"
    sentence_scores_file_template = "sentence_scores-shard_*.csv"
    doc_scores_file_template = "anthroscores-shard_*.csv"

    match = SHARD_NUM_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Couldn't find a shard number in filename {filename}")

    shard_num = match.group(1)
    return (
        Path(SENTENCE_DIR, f"{sentence_file_template.replace('*', shard_num)}"),
        Path(SCORES_DIR, f"{sentence_scores_file_template.replace('*', shard_num)}"),
        Path(SCORES_DIR, f"{doc_scores_file_template.replace('*', shard_num)}")
    )

def build_queue(checkpoint: Checkpoint) -> list[tuple[str, int]]:
    # Skip files that are either processed or in interrupted state
    exclude = set(checkpoint.processed) | {filepath for filepath, _ in checkpoint.interrupted}

    # Queue the interrupted files
    queue = [(filepath, row_index) for filepath, row_index in checkpoint.interrupted]
    # Add unprocessed files
    queue += [(str(filepath), 0) for filepath in ALL_SENTENCE_FILES if str(filepath) not in exclude]
    return queue

# ========== CALCULATE ANTHROSCORES ==========

def get_prediction(sent):
    token_ids = tokenizer.encode(sent,return_tensors='pt').to(device)
    # Add a length check after getting a RunTimeError
    if token_ids.shape[1] > 512:
        masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
        try:
            mask_pos_in_full = masked_position[0].item()
            if (mask_pos_in_full + 256) > token_ids.shape[1]:
                token_ids = torch.reshape(token_ids[0][-512:], (1, 512)) # this needs to be modified as it might lead to Index errors
            else:
                token_ids = torch.reshape(token_ids[0][mask_pos_in_full - 256 : mask_pos_in_full + 256], (1, 512))
        except IndexError:
            return np.empty((len(TERMS),))
    masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    try:
        masked_pos = masked_position[0].item()
    except IndexError:
       return np.empty((len(TERMS),)) 
    
    with torch.no_grad():
        output = model(token_ids)

    last_hidden_state = output[0].squeeze()
    mask_hidden_state = last_hidden_state[masked_pos].cpu().numpy()

    probs = scipy.special.softmax(mask_hidden_state)
    scores = np.array([probs[i] for i in TARGET_INDS])
    return scores

def get_anthroscores_incremental(sentence_filename: str | Path, output_filename: str | Path, checkpoint: Checkpoint):
    column_names = ['sentence','masked_sentence','text_id','POS','verb','original_term','original_noun','year','domain']
    # Open scores file in append mode in case the progress is being returned to
    with open(sentence_filename, newline="") as infile, \
         open(output_filename, "a", newline="") as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = column_names + ["anthroscore"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(reader):
            # Skip the lines that have already been processed
            if checkpoint.current_row_index > i:
                continue

            checkpoint.current_row_index = i
            x = row["masked_sentence"]

            if i > 0 and i % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            if i > 0 and i % 1000 == 0:
                outfile.flush()
                logger.info(f"Calculating sentence {i}")
                checkpoint.save()

            try:
                scores = get_prediction(x)
            except (ValueError, RuntimeError, IndexError, OverflowError) as e:
                logger.error(f"Error in calculating scores for sentence {i} in file {checkpoint.current_file} from ID {row['text_id']}: {type(e).__name__}: {e} ")
                scores = None

            if scores is not None and not np.any(np.isnan(scores)):
                human_scores = np.sum(scores[:FIRST_NONHUMAN_INDEX])
                nonhuman_scores = np.sum(scores[FIRST_NONHUMAN_INDEX:])
                anthroscore = np.log(human_scores) - np.log(nonhuman_scores)
            else:
                anthroscore = float("nan")
            
            row["anthroscore"] = anthroscore
            writer.writerow(row)

def compute_average_scores_groupby(sentence_score_file: str | Path, average_score_file: str | Path):
    usecols = ['text_id','year','domain','anthroscore']
    sentence_df = pd.read_csv(sentence_score_file, usecols=usecols)
    doc_scores = sentence_df.groupby("text_id").agg(
        anthroscore=("anthroscore", "mean"),
        year=("year", "first"),
        domain=("domain", "first")
    ).reset_index()

    doc_scores.to_csv(average_score_file, index=False)

# ========== MAIN ==========

def main():

    logger.info("Loading checkpoint")
    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    logger.info(f"Checkpoint loaded: {checkpoint}")
    queue = build_queue(checkpoint)
    logger.info(f"Queue: {queue}")

    for i, (filepath, start_row) in enumerate(queue, start=1):
        checkpoint.sync_progress(filepath, start_row)

        sentence_file, sentence_score_file, doc_score_file = get_triple(filepath)        

        logger.info(f"Processing file {i}/{len(queue)}")

        try:
            logger.info("Starting to calculate scores for all sentences...")
            get_anthroscores_incremental(sentence_file, sentence_score_file, checkpoint)

            logger.info("Scores for all sentences calculated, writing the average scores per document to file...")
            compute_average_scores_groupby(sentence_score_file, doc_score_file)

            logger.info(f"Scores for {sentence_file.name} written!")
            checkpoint.mark_as_processed()

        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}")
            checkpoint.mark_as_interrupted()
    
    logger.info("All files iterated!")

if __name__ == "__main__":
    main()