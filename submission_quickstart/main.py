from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIRECTORY = Path("/code_execution")
DATA_DIRECTORY = Path("/data")
OUTPUT_FILE = ROOT_DIRECTORY / "subset_matches.csv"
QUERY_SUBSET = DATA_DIRECTORY / "test/query_subset.csv"
REFERENCE_METADATA = DATA_DIRECTORY / "test/reference_metadata.csv"


def predict_overlap(video_id, reference_video_ids) -> np.ndarray:
    overlap = dict()
    overlap["query_id"] = video_id
    overlap["reference_id"] = reference_video_ids[0]
    overlap["query_start"] = 1
    overlap["query_end"] = 2
    overlap["reference_start"] = 3
    overlap["reference_end"] = 4
    overlap["score"] = 0.5
    return overlap


def main():
    # Loading subset of query images
    query_subset_video_ids = pd.read_csv(QUERY_SUBSET).video_id.values
    reference_video_ids = pd.read_csv(REFERENCE_METADATA).video_id.values
    predictions = []
    for query_video_id in query_subset_video_ids:
        overlap = predict_overlap(query_video_id, reference_video_ids)
        predictions.append(overlap)
    pd.DataFrame(predictions).to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
