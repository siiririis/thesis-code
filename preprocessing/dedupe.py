"""
This program filters abstracts according to the following:

* Clean up texts with low ASCII/total alpha chars ratio
-> gets rid of other scripts, but misses certain languages

* Remove objects that don't include any target entity mentions

* If still applicable after above steps, remove objects where
abstract is too short

* Deduplicate
1) if abstract appears more than 5 times, delete all instances
2) if abstract appears 2-5 times and publication years differ,
   pick the one where pubyear is earliest, if pubyears are same,
   pick the one with latest modification date
3) if title appears more than 5 times, delete all instances
4) if title appears 2-5 times, pick the version that has been
   updated the latest; if equal timestamps, pick the longest version
   (based on manual inspection of the duplicated items)


"""

import logging
import re
import gzip
import json
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s (line %(lineno)d) | %(levelname)s: %(message)s"
    )

TEST_MODE = False
RUN_ON_PUHTI = True

if not RUN_ON_PUHTI:
    # === Local files ===
    DATA = "data/preprocessed/light-new-query.jsonl.gz"
    NEW_DATA = "data/preprocessed/light-new-query-deduplicated.jsonl.gz" if not TEST_MODE else "data/preprocessed/light-new-query-deduplicated-sample.jsonl.gz"
    LOG_FILE = "inspection-data/removed.txt" if not TEST_MODE else "inspection-data/removed-sample.txt"
    FINAL_IDS_FILE = "inspection-data/final-ids.txt" if not TEST_MODE else "inspection-data/final-ids-sample.txt"
    ENTITY_FILE = "test-data/entities.txt"

else:
    # === Puhti files ===
    DATA = "/scratch/project_xxxxxxx/data_collection/data/light-data/light-new-query.jsonl.gz"
    NEW_DATA = "/scratch/project_xxxxxxx/data_collection/data/light-data/light-new-query-dedupe.jsonl.gz" if not TEST_MODE else "/scratch/project_xxxxxxx/data_collection/data/inspection-data/light-new-query-dedupe-sample.jsonl.gz"
    LOG_FILE = "/scratch/project_xxxxxxx/data_collection/data/inspection-data/dedupe-log.txt" if not TEST_MODE else "/scratch/project_xxxxxxx/data_collection/data/inspection-data/dedupe-log-sample.txt"
    FINAL_IDS_FILE = "/scratch/project_xxxxxxx/data_collection/data/inspection-data/final-ids-dedupe.txt" if not TEST_MODE else "/scratch/project_xxxxxxx/data_collection/data/inspection-data/final-ids-dedupe-sample.txt"
    ENTITY_FILE = "/scratch/project_xxxxxxx/data_collection/data/entities.txt"

ASCII_THRESHOLD = 0.4
MIN_LEN = 250

def get_entities(filename:str) -> list[str]:
    with open(filename) as file:
        entities = sorted(
            [line.strip() for line in file if line.strip()],
            key=len,
            reverse=True
        )
    return entities

def get_entity_regex(filename: str) -> re.Pattern:
    entities = get_entities(filename)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(entity) for entity in entities) + r")\b",
        re.IGNORECASE
    )
    return pattern

def is_mostly_ascii(text: str, threshold: float = ASCII_THRESHOLD):
    total = ascii_count = 0
    for c in text:
        if c.isalpha():
            total += 1
            if c.isascii():
                ascii_count += 1
    if total == 0:
      return None
    return (ascii_count / total) > threshold

def yield_objects(filename: str):
    with gzip.open(filename) as file:
        for i, line in enumerate(file):
            if TEST_MODE:
                if i % 100 != 0:
                    continue
            yield json.loads(line)


ENTITY_REGEX = get_entity_regex(ENTITY_FILE)

total = 0

abstracts = defaultdict(set)
titles = defaultdict(set)

too_short = set()
no_entities = set()
maybe_not_english = set()

logger.info("Starting the first pass")
for obj in yield_objects(DATA):
    total += 1
    if total > 0 and total % 10000 == 0:
        logger.info(f"{total} objects iterated")
    id_ = obj.get("id")
    abst = obj.get("abstract", "").casefold()

    if len(abst) < MIN_LEN:
        too_short.add(id_)
        continue
    if not ENTITY_REGEX.search(abst):
        no_entities.add(id_)
        continue
    if not is_mostly_ascii(abst):
        maybe_not_english.add(id_)
        continue

    abstracts[abst].add(id_)

    title = obj.get("title", "").casefold()
    titles[title].add(id_)

skipped = len(no_entities) + len(maybe_not_english) + len(too_short)

logger.info(f"Initial dataset contained {total} works.")
logger.info(f"Skipped {skipped} works in total:")
logger.info(f"Skipped {len(too_short)} works due to abstract length being less than {MIN_LEN} characters.")
logger.info(f"Skipped {len(no_entities)} works as no target entities were found.")
logger.info(f"Skipped {len(maybe_not_english)} works due to low ASCII/alpha character ratio.")

# If the exact same abstract appears more than 5 times, remove all instances of it.
to_remove = set()
to_inspect = set()
to_keep = set()
for ids in abstracts.values():
    if len(ids) > 5:
        to_remove.update(ids)
    elif len(ids) > 1:
        to_inspect.update(ids)
    else:
        to_keep.update(ids)

logger.info("Status after first round (exact abstract duplicates):")
logger.info(f"To remove: {len(to_remove)} / To inspect: {len(to_inspect)} / To keep: {len(to_keep)}")

abstracts_to_deduplicate = defaultdict(list)
for obj in yield_objects(DATA):
    id_ = obj.get("id")
    if id_ in to_inspect:
        abst = obj.get("abstract", "").casefold()
        abstracts_to_deduplicate[abst].append(obj)

def parse_date(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str)

for versions in abstracts_to_deduplicate.values():
    # Pick the version that has been published the earliest; if publication years are equal, pick the one latest updated
    versions.sort(key=lambda x: (x["publication_year"], -parse_date(x["updated_date"]).timestamp()))
    to_keep.add(versions[0].get("id"))
    for version in versions[1:]:
        to_remove.add(version.get("id"))

logger.info(f"Marked {len(to_remove)} duplicate objects for removal")
logger.info("Starting deduplication by titles...")

# Empty the inspection set for title inspection
to_inspect.clear()

for ids in titles.values():
    if len(ids) > 5:
        to_remove.update(ids)
        to_keep -= ids
    elif len(ids) > 1:
        to_inspect.update(ids)
        to_keep -= ids
        to_remove -= ids

logger.info("Status after second round (exact title duplicates):")
logger.info(f"To remove: {len(to_remove)} / To inspect: {len(to_inspect)} / To keep: {len(to_keep)}")

# Empty the inspection dict
abstracts_to_deduplicate = defaultdict(list)
for obj in yield_objects(DATA):
    id_ = obj.get("id")
    if id_ in to_inspect:
        title = obj.get("title", "").casefold()
        abstracts_to_deduplicate[title].append(obj)

for versions in abstracts_to_deduplicate.values():
    # Pick the version that has been updated the latest; if equal timestamps, pick the longest version
    versions.sort(key=lambda x: (-parse_date(x["updated_date"]).timestamp(), -len(x["abstract"])))
    to_keep.add(versions[0].get("id"))
    for version in versions[1:]:
        to_remove.add(version.get("id"))

logger.info(f"Marked {len(to_remove)} duplicate objects for removal in total")
logger.info(f"to_keep {len(to_keep)} + to_remove {len(to_remove)} + skipped {skipped} = {len(to_keep)+len(to_remove)+skipped} (expected {total})")
assert (len(to_keep) + len(to_remove) + skipped) == total, f"Number of kept and removed documents doesn't match"
logger.info(f"Total dataset size after filtering/deduplication: {len(to_keep)} ({len(to_keep) / total:.2%} of the original kept)")

logger.info("Writing the IDs of removed objects into a file...")
with open(LOG_FILE, "w") as file:
    for id_set, title in zip(
        (no_entities, maybe_not_english, too_short, to_remove),
        ("NO ENTITIES", "NOT ASCII", f"UNDER {MIN_LEN} CHARACTERS", "DUPLICATES")
    ):
        print(f"# {title} ({len(id_set)} IDs)", file=file)
        for id_ in id_set:
            print(id_, file=file)
        print(file=file)

logger.info("Writing the IDs of the deduplicated objects into a file...")
with open(FINAL_IDS_FILE, "w") as file:
    for id_ in to_keep:
        print(id_, file=file)

logger.info("Writing the new dataset into a file...")
with gzip.open(NEW_DATA, "wt") as file:
    for obj in yield_objects(DATA):
        id_ = obj.get("id")
        if id_ in to_keep:
            print(json.dumps(obj, ensure_ascii=False), file=file)

logger.info("All done!")