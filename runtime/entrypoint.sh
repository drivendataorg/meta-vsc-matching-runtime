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

    if [ -f "full_matches.csv" ]
    then
        echo "Validating submitted matches..."
        conda run --no-capture-output -n condaenv \
            python /opt/validation.py \
            --path full_matches.csv
    else
        echo "ERROR: Could not find full_matches.csv"
        exit 1
    fi

    # Use submitted code to generate matches on a subset of query videos
    if [ -f "main.py" ]
    then
        echo "Generating matches on a subset of query videos..."
        echo "Started at $(date -u +'%Y-%m-%dT%H:%M:%SZ') ($(date +%s))"
        conda run --no-capture-output -n condaenv python main.py
        echo "Finished at $(date -u +'%Y-%m-%dT%H:%M:%SZ') ($(date +%s))"

        echo "Validating matches subset..."
        conda run --no-capture-output -n condaenv \
            python /opt/validation.py \
            --path subset_matches.csv

	    echo "... finished"
    else
        echo $PHASE2
        if [ "$PHASE2" == "true" ]
        then
            echo "ERROR: Non-code execution submissions are not allowed in Phase 2"
            exit 1
        else
            echo "WARNING: Could not find main.py in submission.zip"
            touch subset_matches.csv
        fi
    fi

    # Tar the full matches csv and the subset matches csv together to form the submission file
    tar -czvf /code_execution/submission/submission.tar.gz \
        subset_matches.csv \
        full_matches.csv

    echo "================ END ================"
} |& tee "/code_execution/submission/log.txt"

cp /code_execution/submission/log.txt /tmp/log
exit $exit_code
