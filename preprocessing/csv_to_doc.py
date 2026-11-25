import csv
import spacy
from spacy.tokens import DocBin
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s (line %(lineno)d) | %(levelname)s: %(message)s"
    )

logger.info("Loading the spaCy model...")
nlp = spacy.load("en_core_web_sm")

RUN_ON_PUHTI = True

if not RUN_ON_PUHTI:
    # Local
    DATA_FILE = "test-data/sample.csv"
    DOC_FILE = "fulltext-sentences"
    DOC_DIR = "inspection-data/spacy-data/"
    N_DOCS = 0
    LOGGING_INTERVAL = 5
    MIN_SHARDS = 1

else:
    # Puhti
    DATA_FILE = "/scratch/project_xxxxxxx/data_collection/data/parsed-fulltexts/fulltexts.csv"
    DOC_FILE = "fulltext-sentences-extended"
    DOC_DIR = "/scratch/project_xxxxxxx/data_collection/data/spacy-data/openalex-xml-extended-metadata/"
    N_DOCS = 0
    LOGGING_INTERVAL = 1000
    MIN_SHARDS = 10

def get_line_count(filename: str):
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        return sum(1 for row in reader)

def yield_tuples(
        filename: str,
        text_key: str = "text",
        metadata_keys: tuple = ("paper_id", "year", "domain"),
        n: int = 0
        ):
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        for i, line in enumerate(reader, start=1):
            text = line.get(text_key, "")
            metadata = {key: value for key, value in line.items() if key in metadata_keys}
            yield text, metadata
            if n > 0 and i == n:
                return

total_docs = get_line_count(DATA_FILE)
SHARD_SIZE = (total_docs // MIN_SHARDS)
logger.info(f"Total documents to process: {total_docs} -> Max shard size: {SHARD_SIZE}")

metadata_keys = (
    "paper_id",
    "year",
    "domain",
    "section_type",
    "section",
    "paragraph_index",
    "sentence_index"
)
logger.info(f"Storing the following fields as metadata: {metadata_keys}")

# Initialize a DocBin object
doc_bin = DocBin(store_user_data=True)

logger.info("Starting to process the texts...")

total = 0
latest_shard = 0
try:
    for i, (doc, metadata) in enumerate(
        nlp.pipe(yield_tuples(DATA_FILE, n=N_DOCS, metadata_keys=metadata_keys), as_tuples=True, batch_size=128), start=1):

        if i % LOGGING_INTERVAL == 0:
            logger.info(f"Processing abstract number {i}")

        if i % SHARD_SIZE == 0:
            shard = i // SHARD_SIZE
            save_to = f"{DOC_DIR}{DOC_FILE}_{shard:03d}.spacy"
            doc_bin.to_disk(save_to)
            logger.info(f"Shard {shard} saved to to disk at {save_to}")
            latest_shard = shard
            doc_bin = DocBin(store_user_data=True) # reset DocBin
        
        # Add to DocBin
        doc.user_data.update(metadata)
        doc_bin.add(doc)

        total = i
finally:
    # (Try to) commit the last batch, even in the event of an error
    shard = latest_shard + 1
    save_to = f"{DOC_DIR}{DOC_FILE}_{shard:03d}.spacy"
    doc_bin.to_disk(save_to)
    logger.info(f"Shard {shard} saved to to disk at {save_to}")
    logger.info(f"DocBins saved to {DOC_DIR}")

logger.info(f"{total} documents processed, run finished")