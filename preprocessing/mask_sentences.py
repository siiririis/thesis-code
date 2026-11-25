from pathlib import Path
import spacy
from spacy.tokens import Doc, DocBin
import re
import csv
from typing import Iterator
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s (line %(lineno)d) | %(levelname)s: %(message)s"
    )

nlp = spacy.load("en_core_web_sm")
logger.info("spaCy model loaded")

MASK_TOKEN = "[MASK]"

RUN_ON_PUHTI = True

if not RUN_ON_PUHTI:
    # Local
    ENTITY_FILENAME = "test-data/all_entities.txt"
    SPACY_DOC_FILE = "inspection-data/spacy-data/fulltext-sentences_001.spacy"
    SENTENCE_FILENAME = "inspection-data/masked-sentences-xml.csv"
    TOKEN_FILENAME = "inspection-data/tokens-xml.csv"
    REPORT_INTERVAL = 100

else:
    # Puhti
    ENTITY_FILENAME = "/scratch/project_xxxxxxx/results-tests/target_entities/entities_all.txt"
    SPACY_DOC_DIR = "/scratch/project_xxxxxxx/data_collection/data/spacy-data/openalex-xml-extended-metadata/"
    CSV_DIR = "/scratch/project_xxxxxxx/results-tests/csv-data/masked-sentences/xml-fulltexts-extended-metadata/"
    TOKEN_DIR = "/scratch/project_xxxxxxx/results-tests/csv-data/token-data/xml-fulltexts-extended-metadata/"
    REPORT_INTERVAL = 1000

# ------------------------------

def get_filenames(dir: str, glob_pattern: str):
    return sorted(Path(dir).glob(glob_pattern))

def get_unprocessed(
          csv_dir:str,
          spacy_dir: str,
          token_dir: str,
          csv_glob_pattern: str = "masked-sentences-xml-extended-shard_*.csv",
          spacy_glob_pattern: str = "fulltext-sentences-extended_*.spacy"
    ) -> list[tuple[Path, Path, Path]]:
    
    csv_files = get_filenames(csv_dir, csv_glob_pattern)
    processed = set()
    for filepath in csv_files:
        match = re.search(r"shard_(\d{3})", filepath.name)
        if match:
            processed.add(match.group(1))
    
    spacy_docbin_files = get_filenames(spacy_dir, spacy_glob_pattern)
    unprocessed = []
    for filepath in spacy_docbin_files:
        match = re.search(r"fulltext-sentences-extended_(\d{3})", filepath.name)
        if match:
            shard_num = match.group(1)
            if shard_num not in processed:
                csv_filepath = Path(csv_dir, f"{csv_glob_pattern.replace('*', shard_num)}")
                token_filepath = Path(token_dir, f"tokens_xml_extended-shard_{shard_num}.csv")
                unprocessed.append((filepath, csv_filepath, token_filepath))
    return unprocessed

def get_entities(filename:str) -> list[str]:
    with open(filename) as file:
        entities = sorted(
            [line.strip() for line in file if line.strip()],
            key=len,
            reverse=True
        )
    return entities

def mask_sentences(doc: Doc, entity_pattern: re.Pattern):
    
    match_rows = []
    token_rows = []

    for sent_id, sent in enumerate(doc.sents):
        # Find all instances that match the regex
        matches = list(entity_pattern.finditer(sent.text))
        if not matches:
            continue

        # Map match indices to sentence indices with + sent.start_char
        m_starts = np.fromiter((sent.start_char + m.start() for m in matches), dtype=np.int64)
        m_ends   = np.fromiter((sent.start_char + m.end() for m in matches), dtype=np.int64)

        contained = np.zeros(len(matches), dtype=bool)
        idx = np.full(len(matches), -1, dtype=np.int64)

        # Find all noun chunks in the sentence
        chunk_spans = list(sent.noun_chunks)
        if chunk_spans:
        
            chunk_starts = np.fromiter((c.start_char for c in chunk_spans), dtype=np.int64)
            chunk_ends   = np.fromiter((c.end_char for c in chunk_spans), dtype=np.int64)

            # Check if a match is inside a noun chunk
            idx = np.searchsorted(chunk_starts, m_starts, side="right") - 1
            valid = idx >= 0
            contained[valid] = chunk_ends[idx[valid]] >= m_ends[valid]

        for match_id, m in enumerate(matches):
            match_span = sent.char_span(m.start(), m.end(), alignment_mode="expand")
            if match_span is None:
                logger.info(f"Match span was not found for {doc.user_data['paper_id']} sentence {sent_id}, match {match_id}")
                continue
            
            # No need to shift indices
            masked_sentence = sent.text[:m.start()] + MASK_TOKEN + sent.text[m.end():]

            in_chunk = bool(contained[match_id])
            chunk = chunk_spans[idx[match_id]] if in_chunk else None
            
            # Use doc indices which align to chunk indices
            masked_sentence_chunk = doc.text[sent.start_char:chunk.start_char] + MASK_TOKEN + doc.text[chunk.end_char:sent.end_char] if chunk else None

            match_rows.append(
                doc.user_data | {
                    "sent_id": sent_id,
                    "match_id": match_id,
                    "sentence": sent.text,
                    "masked_sentence_minimal": masked_sentence,
                    "masked_sentence_chunk": masked_sentence_chunk if chunk else masked_sentence,
                    "entity": m.group(),
                    "entity_lower": m.group().lower(),
                    "in_noun_chunk": in_chunk,
                    "chunk_text": chunk.text if chunk else None,
                    "chunk_root_text": chunk.root.text if chunk else None,
                    "chunk_root_dep": chunk.root.dep_ if chunk else None,
                    "head": chunk.root.head.lemma_.lower() if chunk else None
                })

            # token metadata for the matched span
            for tok in match_span:
                token_rows.append(
                    doc.user_data | {
                        "sent_id": sent_id,
                        "match_id": match_id,
                        "role": "match",       # distinguishes match tokens from chunk tokens
                        "token_i": tok.i,
                        "text": tok.text,
                        "pos": tok.pos_,
                        "tag": tok.tag_,
                        "dep": tok.dep_,
                        "ent_type": tok.ent_type_,
                        "ent_iob": tok.ent_iob_,
                    })

            # token metadata for the full noun chunk (superset, includes match tokens too)
            if in_chunk and chunk:
                for tok in chunk:
                    token_rows.append(
                        doc.user_data | {
                            "sent_id": sent_id,
                            "match_id": match_id,
                            "role": "chunk",
                            "token_i": tok.i,
                            "text": tok.text,
                            "pos": tok.pos_,
                            "tag": tok.tag_,
                            "dep": tok.dep_,
                            "ent_type": tok.ent_type_,
                            "ent_iob": tok.ent_iob_,
                        })

    yield match_rows, token_rows

# Iterate over all sentences in data
# and collect and mask any sentences that include target terms
# using both a minimal and a noun chunk masking strategy

def write_spacy_sentences_to_file(docs: Iterator[Doc] | list[Doc],
                                  entity_pattern: re.Pattern,
                                  output_filename: str | Path,
                                  token_filename: str | Path):

    fieldnames_sentence_file = [
        "paper_id", "year", "domain",
        "sentence_index", "section", "section_type", "paragraph_index",
        "sent_id", "match_id",
        "sentence", "masked_sentence_minimal", "masked_sentence_chunk",
        "entity", "entity_lower",
        "in_noun_chunk", "chunk_text",
        "chunk_root_text", "chunk_root_dep",
        "head"
    ]

    fieldnames_token_file = [
        "paper_id", "year", "domain",
        "sentence_index", "section", "section_type", "paragraph_index",
        "sent_id", "match_id", "role", "token_i",
        "text",
        "pos", "tag", "dep",
        "ent_type", "ent_iob"
    ]
    
    with open(output_filename, "w", newline="") as sentence_file, open(token_filename, "w", newline="") as token_file:
        sentence_writer = csv.DictWriter(sentence_file, fieldnames=fieldnames_sentence_file)
        sentence_writer.writeheader()

        token_writer = csv.DictWriter(token_file, fieldnames=fieldnames_token_file)
        token_writer.writeheader()

        num_sentences = 0
        for i, doc in enumerate(docs, start=1):
            if i % REPORT_INTERVAL == 0:
                logger.info(f"{i} documents iterated, {num_sentences} sentences found")
            for match_rows, token_rows in mask_sentences(doc, entity_pattern):
                sentence_writer.writerows(match_rows)
                num_sentences += len(match_rows)

                token_writer.writerows(token_rows)

    logger.info(f"{num_sentences} sentences written to the CSV file ({output_filename})")

def main():
    if RUN_ON_PUHTI:
        unprocessed = get_unprocessed(CSV_DIR, SPACY_DOC_DIR, TOKEN_DIR)
        logger.info(f"Found {len(unprocessed)} spaCy DocBin files that have not yet been converted to CSV files")

    else:
        unprocessed = [(SPACY_DOC_FILE, SENTENCE_FILENAME, TOKEN_FILENAME)]

    entities = get_entities(ENTITY_FILENAME)
    entity_pattern = re.compile(
        r"\b(" + "|".join(re.escape(entity) for entity in entities) + r")\b",
        re.IGNORECASE
    )
    logger.info(f"Looking for sentences with the following target entities:\n{entities}")

    for i, (docbin_filepath, csv_filepath, token_filepath) in enumerate(unprocessed, start=1):
        logger.info(f"Processing DocBin {i}/{len(unprocessed)}")

        doc_bin = DocBin().from_disk(docbin_filepath)
        logger.info(f"Loaded spaCy DocBin with {len(doc_bin)} Docs from disk ({docbin_filepath})")
        docs = doc_bin.get_docs(nlp.vocab)

        logger.info(f"Writing the sentences into {csv_filepath}")
        write_spacy_sentences_to_file(docs, entity_pattern, csv_filepath, token_filepath)
    
    logger.info("All done!")

if __name__ == "__main__":
    main()