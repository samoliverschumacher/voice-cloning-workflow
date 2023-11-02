# 1. visit hf.co/pyannote/speaker-diarization and hf.co/pyannote/segmentation and accept user conditions (only if requested)
# 2. visit hf.co/settings/tokens to create an access token (only if you had to go through 1.)
# 3. instantiate pretrained speaker diarization pipeline
import csv
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import click
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization

from structures import Transcript

SpeakerActivity = List[ List[Tuple[float, float]] ]  # start and stop (seconds) each speaker is active, to override segmentation models outputs


def process_diarization(diarization: SpeakerDiarization, speaker_list) -> List[dict]:
    """
    Process the diarization of a speech recording.

    Args:
        diarization (Diarization): The diarization object containing the turns and labels of the speech recording.
        speaker_list (list): A list of speaker names.

    Returns:
        list: A list of dictionaries, each representing a turn in the speech recording with its start time, stop time, and speaker name.
    """
    print(f"{speaker_list=}")
    lines = []
    speaker_mapping = {}
    speech_counter = 0
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_mapping:
            print(f"{speaker} is not in the mapping: {speaker_mapping=}")
            if len(speaker_mapping) < len(speaker_list):
                print(f"mapping {speaker} to {speaker_list[len(speaker_mapping)]}")
                speaker_mapping[speaker] = speaker_list[len(speaker_mapping)]
            else:
                print(f"More speakers than the speaker_list. Mapping {speaker} to {speaker}")
                # Handle the case when there are more speakers than the length of speaker_list
                speaker_mapping[speaker] = speaker

        name = speaker_mapping[speaker]
        row = {"start": turn.start, "stop": turn.end, "speaker": name}
        lines.append(row)
        print(row, f"unmapped speaker = {speaker}")
        speech_counter += 1
    print(f"{speech_counter=}")
    return lines


def _diarize(file, n_speakers, semi_supervised: Optional[SpeakerActivity] = None) -> SpeakerDiarization:
    # Get auth token from enviornment variables
    HUGGING_FACE_API_AUTH_TOKEN = os.environ.get("HUGGING_FACE_API_AUTH_TOKEN")
    if HUGGING_FACE_API_AUTH_TOKEN is None:
        raise ValueError("HUGGING_FACE_API_AUTH_TOKEN is not set in environment variables...")
    
    # 4. apply pretrained pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                        use_auth_token=HUGGING_FACE_API_AUTH_TOKEN)
    pipeline=pipeline.to(torch.device('cuda:0'))

    # 4. Perform diarization
    diarization = pipeline(file, num_speakers=n_speakers, 
                           semi_supervised=semi_supervised
                           )
    return diarization


def _process_input_string(input_str) -> SpeakerActivity:
    # Split the input string into individual entries
    entries = input_str.split('|')
    
    # Convert the entries into a list of tuples
    data = []
    for entry in entries:
        values = entry.split(',')
        data.append((float(values[0]), float(values[1])))
    
    # Reshape the data
    semi_supervised = []
    for i in range(0, len(data), 2):
        semi_supervised.append([data[i], data[i+1]])
    
    return semi_supervised


def _process_csv(filename) -> SpeakerActivity:
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    
    # Reshape the data
    semi_supervised = {}
    for start, stop, speaker in data:
        if speaker not in semi_supervised:
            semi_supervised[speaker] = []
        semi_supervised[speaker].append((float(start), float(stop)))
    
    return list(semi_supervised.values())

@click.command()
@click.argument('speaker_list')
@click.argument('file', type=click.Path(exists=True))
@click.option('--output_file', type=click.Path(), help='Output file path')
@click.option('--consolidate_rows', '-cr', is_flag=True, help='Output file path')
@click.option('--override', '-or', type=click.Path(exists=True), help='Filepath or string with overrides. '
                                                               'If in a .csv format: "start,stop,speaker". '
                                                               'If string "start,stop,speaker|start,stop,speaker ..."')
@click.option('--check','-chk', type=click.Path(exists=True), help='Raw transcript filepath to cross check # of speaker turns against.')
def main(speaker_list, 
         file, 
         output_file, 
         consolidate_rows, 
         override, 
         check):  # used
    """
    Main function that performs speaker diarization on an audio file.
    
    Print the diarization to stdout;
    >>> python diarize.py "Joe Smith" "audio.wav"
    
    Specify the output file;
    >>> python diarize.py "Joe Smith" "audio.wav" --output_file "diary.csv"

    Example file contents;
    ```
    start: 0.1, stop: 2.1, speaker: Joe Smith
    start: 3.1, stop: 14.1, speaker: Jane Doe
    ```
    
    Parameters:
    - speaker_list (str): A comma-separated string of speaker IDs. This is a non-unqiue list
    - file (str): The path to the input audio file.
    - output_file (str, optional): The path to the output file. If not provided, the results will be printed to stdout.

    Returns:
    None
    """
    print("Diarizing with inputs;")
    print(f"\t- speaker_list: {speaker_list}")
    print(f"\t- file: {file}")
    print(f"\t- output_file: {output_file}")
    print(f"\t- consolidate_rows: {consolidate_rows}")
    print(f"\t- override: {override}")
    print(f"\t- check: {check}")
    
    # Start the timer
    start_time = time.time()
    # 2. Convert speaker_list from comma separated string to list
    speaker_list = speaker_list.split(',')
    speaker_list_unique = list(dict.fromkeys(speaker_list))

    # Load or reformat overrides if provided
    semi_supervised = None
    if override:
        if Path(override).exists():
            semi_supervised = _process_csv(override)
        else:
            semi_supervised = _process_input_string(override)
        
        print(f"Overrides to speaker segmentation in diarization:\n"+
              "\n".join([f"{str(i)}: {override}" for i, override in enumerate(semi_supervised)]))

    diarization = _diarize(file, n_speakers = len(speaker_list_unique), semi_supervised=semi_supervised)
    
    # 5. Process diarization
    if output_file:
        output_file_path = Path(output_file)
        with open(output_file_path, 'w') as file_handler:
            speaker_segments = process_diarization(diarization, speaker_list_unique)
            for line in speaker_segments:
                file_handler.write(f"start: {line['start']}, stop: {line['stop']}, speaker: {line['speaker']}\n")
        print(f"{output_file_path=}")
        if consolidate_rows:
            if output_file_path is not None:
                output_file_path: Path = Path(output_file_path)
                
            print(f"Diarisation complete. Consolidating speaker turn rows...")
            from split_wav import _consolidate_rows
            consolidated_rows = _consolidate_rows(diary_path=output_file_path, 
                                                  output_file_name=output_file_path)
            all_speakers = set([row.content['speaker'] for row in consolidated_rows])
            found = set()
            for i, row in enumerate(consolidated_rows):
                found.add(row.content['speaker'])
                if all_speakers == found:
                    break
            print(f"Speaker order: {', '.join([row.content['speaker'] for row in consolidated_rows[:i+1]])}")
            print(f"Consolidated speaker count = {len(all_speakers)}")

            # Check the number of rows in transcript is same as diary.
            # default name for transcript is "transcript.txt"
            if check:
                n_transcript_turns = len(Transcript(check))
                if len(consolidated_rows) != n_transcript_turns:
                    print(f"ERROR: {n_transcript_turns=}, diary turn counter={len(consolidated_rows)}")
    else:
        speaker_segments = process_diarization(diarization, speaker_list_unique)
        for line in speaker_segments:
            sys.stdout.write(f"start: {line['start']}, stop: {line['stop']}, speaker: {line['speaker']}\n")
        
    # Calculate the elapsed time
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")


if __name__ == '__main__':
    main()
