import gzip
import json
import csv
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("jsonl_file", type=str, help="Path to the JSONL file you want to turn into a CSV file")
    parser.add_argument("--outfile", type=str, help="Where the CSV file should be saved to. If omitted, will be saved to same location as the JSONL file")
    parser.add_argument("--fields", nargs="+", help="Which fields to include from the JSON objects")
    parser.add_argument("-n", type=int, help="How many lines to use from the JSONL file")
    args = parser.parse_args()
    print(args)

    filename = args.jsonl_file
    assert filename.endswith(".jsonl") or filename.endswith(".jsonl.gz"), f"The provided file ({filename}) may not be a JSONL file"
    
    if not args.outfile:
        csv_filename = filename.replace(".gz", "").replace(".jsonl", ".csv")
        if args.n:
            csv_filename = csv_filename.replace(".csv", f"_{args.n}.csv")
    else:
        csv_filename = args.outfile

    opening_fn = gzip.open if filename.endswith(".gz") else open
    with opening_fn(filename) as infile:
        rows = [json.loads(line) for line in infile if line.strip()]

    if args.n:
        rows = rows[:args.n]

    if args.fields:
        modified_rows = []
        for row in rows:
            modified_row = {}
            for field in args.fields:
                modified_row[field] = row.get(field)
            modified_rows.append(modified_row)
        rows = modified_rows

    with open(csv_filename, "x", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)