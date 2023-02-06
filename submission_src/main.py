from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIRECTORY = Path("/code_execution")
DATA_DIRECTORY = Path("/data")
OUTPUT_FILE = ROOT_DIRECTORY / "submission" / "subset_matches.csv"


def generate_matches(query_video_ids) -> pd.DataFrame:
    raise NotImplementedError(
        "This script is just a template. You should adapt it with your own code."
    )
    matches = ...
    return matches


def main():
    # Loading subset of query images
    query_subset = pd.read_csv(DATA_DIRECTORY / "query_subset.csv")
    query_subset_video_ids = query_subset.video_id.values

    # Generation of query matches happens here #
    matches = generate_matches(query_subset_video_ids)

    matches.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
