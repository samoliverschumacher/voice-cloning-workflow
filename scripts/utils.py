import json
import csv
from pathlib import Path
from typing import List, Union

import click
import pandas as pd

from process_transcript import _process_content, speaker_delimiter


def convert_rttm_to_csv(file: Union[Path, str]) -> pd.DataFrame:
    """
    Inputs:
    file: str
        file path of the rttm file to be converted

    Outputs:
    df: dataframe
        Dataframe containing the extracted information from the rttm file
        
    from: https://mohitmayank.com/a_lazy_data_science_guide/audio_intelligence/speaker_diarization/
    """
    # read the file
    df = pd.read_csv(file, delimiter=" ", header=None)
    df = df[[3, 4, 7]]
    df.columns = ['start_time', 'duration', 'speaker_name']
    # compute the end time
    df['end_time'] = df['start_time'] + df['duration']
    # convert time to miliseconds
    df['start_time'] *= 1000
    df['end_time'] *= 1000
    # sort the df based on the start_time
    df.sort_values(by=['start_time'], inplace=True)
    # return
    return df


@click.group()
def cli():
    pass


@cli.command()
@click.argument('transcript_file', type=click.Path(exists=True))
def extract_names(transcript_file):
    """
    Extracts unique names from a CSV file and prints them as a comma-separated string.
    
    python utils.py extract-names transcript.txt
    Jane Smith,David Adam
    """
    from structures import Transcript
    transcript = Transcript(transcript_file)
    
    click.echo(','.join(transcript.speaker_order))


@cli.command()
@click.argument('data_identifier', type=str)
@click.argument('audio_fpath', type=click.Path(exists=True))
def ini_file_dir(data_identifier, audio_fpath):  # used
    """
    Initializes a file directory for a given data identifier and audio file path.
    
    Args:
        data_identifier (str): The identifier for the data.
        audio_fpath (str): The file path of the audio file.
        
    Returns:
        None
    """
    
    root_dir = Path(__file__).parent.parent
    data_identifier_fpath: Path = root_dir / "data" / "01_interim" / data_identifier
    raw_audio_fpath = Path(audio_fpath)
    audio_fpath = data_identifier_fpath / "audio.wav"

    if data_identifier_fpath.exists():
        print(f"{data_identifier_fpath=} already exists.")
    else:
        data_identifier_fpath.mkdir(parents=True)
        
    # Create a symlink to the original file, or relink if already exists.
    if audio_fpath.exists():
        audio_fpath.unlink()
    audio_fpath.symlink_to(raw_audio_fpath)


@cli.command()
@click.argument('metadata_file_path', type=click.Path(dir_okay=False))
@click.option('--input-string', prompt='Enter the input string to append to metadata.json', help='Input string to append')
def append_metadata(metadata_file_path, input_string):  # used
    """
    Appends an input string to a JSON file.

    Args:
        metadata_file_path (str): The path to the metadata file.
        input_string (str): The string to append to the metadata file.

    Returns:
        None
    """
    print(f"Calling append_metadata with: {metadata_file_path=}, {input_string=}")
    
    file_path = Path(metadata_file_path)
    assert file_path.suffix == '.json', f"metadata_file_path must be a json file. {file_path.suffix=}"

    data = []
    if file_path.exists():
        with open(file_path, 'r') as file:
            data = json.load(file)

    data.append(input_string)

    with open(file_path, 'w') as file:
        json.dump(data, file)

    click.echo(f'Appended "{input_string}" to {file_path}')
    

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.argument('line_number', type=int)
@click.argument('substring')
@click.option('--save_as', '-s', type=str, help='Save as a new file with this name')
@click.option('--is_transcript', '--is-transcript', is_flag = True, help='treats the file as a "|"-delimited processed trnascript, and converts the input substring to match.')
def remove_text_after_substring(file_path, line_number, substring, save_as=None, is_transcript=True):
    """
    
    Usage example
    >>> input_file = 'path/to/speaker/transcript/Susan-Long-33.txt'
    >>> line_number = 1
    >>> substring = 'There is a guy called James Grotstein'
    >>> save_as = None # 'path/to/speaker/transcript/Susan-Long-33.txt'  # Optional
    >>> is_transcript = True
    >>> remove_text_after_substring(input_file, line_number, substring, save_as, is_transcript)
    
    """
    
    # Read the input file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if is_transcript:
        substring = _process_content(substring,False)
    # Remove text after the specified substring in the nominated line
    updated_lines = []
    for i, line in enumerate(lines):
        if i == line_number - 1:  # Line numbers start from 1
            
            index = line.find(substring)

            if index != -1:
                line = line[:index]
                
        # expect only word breaks between words.
        if is_transcript:
            line = line.strip(speaker_delimiter)
            
        updated_lines.append(line)

    # Save the modified content to a new file
    if save_as:
        with open(save_as, 'w') as file:
            file.writelines(updated_lines)
    else:
        for line in updated_lines:
            print(line, end='')

def concat_same_speakername(lines):
    """
    Process lines with the pattern <name> | <content> to concatenate content for the same name.

    Args:
        lines (list): List of lines in the format <name> | <content>.

    Returns:
        list: Updated list of lines.
    """
    updated_lines = []
    current_name = None
    current_content = None

    for line in lines:
        name, content = line.split('|')
        name = name.strip()
        content = content.strip()

        if name == current_name:
            current_content += ' ' + content
        else:
            if current_name is not None:
                updated_lines.append(f"{current_name}| {current_content}\n")
            current_name = name
            current_content = content

    if current_name is not None:
        updated_lines.append(f"{current_name}| {current_content}\n")

    return updated_lines


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('line_number', type=int)
@click.option('--save-as', type=str, help='Save as a new file with this name')
def remove_speaker_turn(input_file, line_number, save_as):
    """
    Remove a line from a speaker diarization file and optionally save it under a new filename.

    Args:
        input_file (str): Path to the input file.
        line_number (int): Line number to be removed.
        save_as (str): Optional. New filename to save the modified content.

    Returns:
        None
        
    Example:
        To remove the 2nd line from 'input.txt' and save it as 'output.txt', run:
        >>> remove_speaker_turn input.txt 2 --save-as output.txt
        Line 2 removed. File saved as output.txt.

        To remove the 2nd line from 'input.txt' where the speaker is the same as the preceding line, resulting in concatenated content, and save it as 'output.txt', run:
        >>> remove_speaker_turn input.txt 2 --save-as output.txt
        Line 2 removed. File saved as output.txt.

    Input Example:
        name1 | content1
        name2 | content2
        name1 | content3
        name3 | content4

    Output Example:
        name1 | content1 content3
        name3 | content4
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()

    if line_number < 1 or line_number > len(lines):
        click.echo(f"Error: Line number {line_number} is out of range.")
        return

    lines.pop(line_number - 1)

    updated_lines = concat_same_speakername(lines)

    if save_as:
        save_as_filepath = Path(input_file).parent / save_as
        with open(save_as_filepath, 'w') as file:
            file.writelines(updated_lines)
        click.echo(f"Line {line_number} removed. File saved to path {save_as_filepath}.")
    else:
        with open(input_file, 'w') as file:
            file.writelines(updated_lines)
        click.echo(f"Line {line_number} removed. File overwritten to path {input_file}")


def convert_to_rttm(data: List[str], file_id):
    """
    Rich Transcription Time Marked (RTTM) files are space-delimited text files containing one turn per line, each line containing ten fields:

    * Type -- segment type; should always by SPEAKER
    * File ID -- file name; basename of the recording minus extension (e.g., rec1_a)
    * Channel ID -- channel (1-indexed) that turn is on; should always be 1
    * Turn Onset -- onset of turn in seconds from beginning of recording
    * Turn Duration -- duration of turn in seconds
    * Orthography Field -- should always by <NA>
    * Speaker Type -- should always be <NA>
    * Speaker Name -- name of speaker of turn; should be unique within scope of each file
    * Confidence Score -- system confidence (probability) that information is correct; should always be <NA>
    * Signal Lookahead Time -- should always be <NA>
    
    For instance:

    SPEAKER CMU_20020319-1400_d01_NONE 1 130.430000 2.350 <NA> <NA> juliet <NA> <NA>
    SPEAKER CMU_20020319-1400_d01_NONE 1 157.610000 3.060 <NA> <NA> tbc <NA> <NA>
    SPEAKER CMU_20020319-1400_d01_NONE 1 130.490000 0.450 <NA> <NA> chek <NA> <NA>
    
    Example usage;
    >>> data = [ "start: 7.12, stop: 16.91, speaker: Jane Smith",
                 "start: 18.71, stop: 33.77, speaker: David Adam",
                 "start: 36.11, stop: 59.03, speaker: Jane Smith",
                 "start: 60.45, stop: 78.37, speaker: David Adam" ]
    >>> rttm_data = convert_to_rttm(data, file_id="AIM-1418")
    >>> print(rttm_data)
    """
    rttm_lines = []
    for line in data:
        start_time = line.split(",")[0].split(":")[1].strip()
        stop_time = line.split(",")[1].split(":")[1].strip()
        speaker_name = line.split(",")[2].split(":")[1].strip()
        duration = float(stop_time) - float(start_time)
        rttm_line = f"SPEAKER {file_id} 1 {start_time} {duration:.3f} <NA> <NA> {speaker_name} <NA> <NA>"
        rttm_lines.append(rttm_line)
    return "\n".join(rttm_lines)

if __name__ == '__main__':
    # TODO: centralise CLI
    cli()