#!/bin/bash

# Checks there is a .wav file for each name listed in the transcript.alignment.txt indec file
verify_alignment() {
    # echo "start"
    DIRECTORY=$1
    all_matches_found=true

    while IFS= read -r line; do
        stem=$(echo "$line" | awk '{print $1}')

        if [[ ! -f "$DIRECTORY/$stem.wav" ]]; then
            all_matches_found=false
            break
        fi
    done < "$DIRECTORY/transcript.alignment.txt"

    if $all_matches_found && [[ -s "$DIRECTORY/transcript.alignment.txt" ]]; then
        echo "All files found in $DIRECTORY"
    fi
}

# Call the function with the provided argument
verify_alignment "$1"
