from pathlib import Path
import shlex
import shutil
import subprocess

import pandas as pd

ROOT_DIRECTORY = Path("/code_execution/")
DATA_DIRECTORY = Path("/data")
QUERY_SUBSET_VIDEOS_FOLDER = DATA_DIRECTORY / "queries"
OUTPUT_FILE = ROOT_DIRECTORY / "subset_matches.csv"
QUERY_SUBSET_FILE = DATA_DIRECTORY / "query_subset.csv"


def main():
    descriptor_command = shlex.split(
        f"""
    conda run --no-capture-output -n condaenv python -m vsc.baseline.inference
        --torchscript_path ./sscd_disc_mixup.no_l2_norm.torchscript.pt
        --accelerator=cuda --processes="1"
        --dataset_path "{QUERY_SUBSET_VIDEOS_FOLDER}"
        --output_file "./subset_query_descriptors.npz"
    """
    )
    subprocess.run(descriptor_command, cwd=(ROOT_DIRECTORY / "vsc2022"))

    matching_command = shlex.split(
        
        f"""
    conda run --no-capture-output -n condaenv python -m vsc.baseline.sscd_baseline \
        --query_features ./subset_query_descriptors.npz \
        --ref_features ./reference_descriptors.npz \
        --output_path ./matches
    """
    )
    subprocess.run(matching_command, cwd=(ROOT_DIRECTORY / "vsc2022"))

    df = pd.read_csv(ROOT_DIRECTORY / "vsc2022" / "matches" / "matches.csv")
    df = df[["query_id", "ref_id", "query_start", "query_end", "ref_start", "ref_end", "score"]]
    df.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
