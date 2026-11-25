import logging
from argparse import ArgumentParser
from pathlib import Path
import json
import gzip
import shutil
import traceback
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_next_file_index(destination_dir: Path, filename: str, suffix: str = ".jsonl.gz"):

    filename = f"{filename}_"
    pattern = re.compile(rf"^{re.escape(filename)}(\d+){re.escape(suffix)}$")

    max_index = -1
    for file in destination_dir.iterdir():
        if not file.is_file():
            continue
        match = pattern.match(file.name)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)
    return max_index + 1

def jsonl_writer(source_dir: Path, destination_dir: Path, filename: str, max_lines: int = 100_000):
    
    file_index = get_next_file_index(destination_dir, filename)
    line_count = 0
    file = None
    current_path = None

    try:
        while True:
            obj = yield

            if file is None or line_count >= max_lines:
                if file:
                    file.close()
                    logger.info(f"{current_path} reached max_lines limit ({max_lines}) and was closed.")
                    if current_path:
                        shutil.move(current_path, destination_dir)
                        logger.info(f"Moved {current_path} to {destination_dir}")
                
                current_path = (source_dir / f"{filename}_{file_index:03d}").with_suffix(".jsonl.gz")
                file = gzip.open(current_path, "wt", encoding="utf-8")
                logger.info(f"Writing the objects to {file.name}")

                file_index += 1
                line_count = 0
            
            file.write(json.dumps(obj))
            file.write("\n")
            line_count += 1
    
    finally:
        if file:
            file.close()
            shutil.move(file.name, destination_dir)
            logger.info(f"Moved {file.name} to {destination_dir}")

def report(total: int, errors: int) -> str:
    success = total-errors
    return f"""
Writing JSON files to JSONL completed
    Success: {success}/{total} ({success/total:.2%})
    Failed:  {errors}/{total} ({errors/total:.2%})
"""

def main(args):

    logger.info(args)
    source = Path(args.source_dir)
    destination = Path(args.destination_dir)

    total_files = 0
    errors = 0

    writer = jsonl_writer(
        source_dir=source,
        destination_dir=destination,
        filename=args.filename,
        max_lines=args.max_lines
        )
    next(writer)

    for json_path in sorted(source.rglob("W*.json")):

        try:
            with open(json_path) as metadata_file:
                total_files += 1
                obj = json.load(metadata_file)
                writer.send(obj)

        except json.JSONDecodeError:
            errors += 1
            logger.error(f"Invalid JSON in file {json_path.name}")
            
        except Exception:
            errors += 1
            logger.error(traceback.print_exc())
        
    writer.close()

    logger.info(report(total_files, errors))

if __name__ == "__main__":

    parser = ArgumentParser(
        description="Given a folder, locates all files that match the pattern 'W*.json' \
            and writes the objects into a single JSONL file, or multiple containing max 100,000 lines each."
    )
    
    parser.add_argument(
        "source_dir",
        help="Path to the directory where the JSON files are located"
        )
    parser.add_argument(
        "destination_dir",
        help="Path to file where JSONL file(s) should be stored permanently"
    )
    parser.add_argument(
        "--filename",
        default="openalex-articles",
        type=str,
        help="Filename for the JSONL file without the .jsonl extension"
        )
    parser.add_argument(
        "--max_lines",
        default=100_000,
        type=int,
        help="How many lines to write into an JSONL file before opening a new one."
    )
    
    args = parser.parse_args()

    main(args)