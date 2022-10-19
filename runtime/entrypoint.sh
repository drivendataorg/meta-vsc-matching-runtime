#!/bin/bash
set -euxo pipefail
exit_code=0

{
    cd /code_execution

    echo "List installed packages"
    echo "######################################"
    conda list -n condaenv
    echo "######################################"

    echo "Unpacking submission..."
    unzip ./submission/submission.zip -d ./
    ls -alh

    # Use submitted code to generate matches on a subset of query videos
    if [ -f "main.py" ]
    then
        echo "Generating matches on a subset of query videos..."
        conda run --no-capture-output -n condaenv python main.py
	    echo "... finished"
        else
            echo "WARNING: Could not find main.py in submission.zip, generating empty file"
            touch subset_matches.csv
    fi

    # Tar the full matches csv and the subset matches csv together to form the submission file
    tar -czvf /code_execution/submission/submission.tar.gz \
        subset_matches.csv \
        full_matches.csv

    echo "================ END ================"
} |& tee "/code_execution/submission/log.txt"

cp /code_execution/submission/log.txt /tmp/log
exit $exit_code
