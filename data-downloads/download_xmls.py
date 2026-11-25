import logging
from argparse import ArgumentParser
import sys
from pathlib import Path
from contextlib import nullcontext
import requests
from collections import defaultdict
from time import sleep
from datetime import date

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DESTINATION_BASE = Path("/scratch/project_xxxxxxx/data_collection/data/xml-data/")
STATUS_FILE = "/scratch/project_xxxxxxx/data_collection/data/xml_download_status.txt"

def mark_downloaded_files(id_list: list[str], error_dict: dict[int, list[str]]) -> None:

    id2error = {id: error_code for error_code, id_list in error_dict.items() for id in id_list}
    failed = set(id2error.keys())
    downloaded = set(id_list) - failed

    updated_lines = []
    with open(STATUS_FILE) as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("#") or line.startswith("!"): # already marked as downloaded or failed, no need to change
                updated_lines.append(line)
            elif line in downloaded:
                updated_lines.append(f"# {line}")
            elif line in failed:
                updated_lines.append(f"! {line} {id2error[line]}")
            else:
                updated_lines.append(line)
    
    with open(STATUS_FILE, "w") as file:
        for line in updated_lines:
            file.write(line)
            file.write("\n")

def get_total(error_dict: dict[int, list[str]]) -> int:
    return sum(len(id_list) for id_list in error_dict.values())

def get_most_common(error_dict: dict[int, list[str]]):
    for error_code, id_list in sorted(error_dict.items(), key=lambda x: len(x[1]), reverse=True):
        yield error_code, len(id_list)

def main(api_key, id_list):

    TODAY = date.today().strftime("%Y-%m-%d")
    DEST_DIR = (DESTINATION_BASE / TODAY)
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    BASE_URL_TEMPLATE = "https://content.openalex.org/works/<ID>.grobid-xml"
    FILENAME_TEMPLATE = "<ID>.grobid.xml.gz"

    RATE_LIMIT_ENDPOINT = f"https://api.openalex.org/rate-limit?api_key={api_key}"
    sleep_time = 0.2 # should be allowed to do even 100 requests per second, so this is much slower

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/xml, application/gzip, binary/octet-stream"
    }

    errors = defaultdict(list)

    for id in id_list:
        url = BASE_URL_TEMPLATE.replace("<ID>", id)
        save_to = DEST_DIR / FILENAME_TEMPLATE.replace("<ID>", id)

        with requests.get(url, headers=headers, stream=True) as response:
            if response.status_code == 200:
                with open(save_to, "wb") as file:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            file.write(chunk)
            else:
                logger.error(f"Failed to fetch {id} (error {response.status_code})")
                errors[response.status_code].append(id)
                if response.status_code == 429:
                    with requests.get(RATE_LIMIT_ENDPOINT) as rate_limit_response:
                        logger.info("Response from Rate Limit Endpoint:")
                        logger.info(rate_limit_response.headers)
                        rl = rate_limit_response.headers.get("rate_limit", {})
                        if rl and isinstance(rl, dict):
                            remaining = rl.get("daily_remaining_usd", 0)
                            if remaining < 0.01:
                                logger.error("Balance exhausted. Quitting the program.")
                                break
                            else:
                                logger.warning(f"Sleeping for a bit longer from now on...")
                                sleep_time = 5
                                sleep(60)
        sleep(sleep_time)
    
    total_errors = get_total(errors)
    logger.info(f"{len(id_list)-total_errors} / {len(id_list)} downloads successful")
    if total_errors > 0:
        error_summary = f"Errors ({total_errors} in total):\n"
        for e, c in get_most_common(errors):
            error_summary += f"    {e} ({c})\n"
        logger.info(error_summary)
    else:
        logger.info("No errors detected.")

    mark_downloaded_files(id_list, errors)
            

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("api_key")
    parser.add_argument("infile", nargs="?", default=STATUS_FILE, help="Skips lines that do NOT start with 'W'. To read from stdin, pass '-'.")
    args = parser.parse_args()

    MAX_LINES = 100 # cannot download more per day

    if args.infile == "-":
        context = nullcontext(sys.stdin)
        logger.info("Reading stdin...")
    else:
        context = open(args.infile)
        logger.info(f"Reading from file {args.infile}")

    with context as file:
        ids = []
        for line in file:
            line = line.strip()
            if line.startswith("W"):
                ids.append(line)
            if len(ids) == MAX_LINES:
                break

    assert len(ids) > 0, "No ids provided"
    assert len(ids) <= MAX_LINES, "Too many ids received"

    main(args.api_key, ids)