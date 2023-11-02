import re
from collections import deque
from itertools import cycle, islice
from pathlib import Path
from typing import List, Optional, Tuple, Union

import click
import librosa
import numpy as np
import soundfile as sf

from move_files import gather_word_audio_file_pairs_all_subdirs
from process_transcript import _process_content, word_break_sym
from split_to_utterances import process
from structures import Annotation, Diarization

"""
    CLI for cropping, trimming & editing audio files (and some transcript editing tools).
"""

@click.group()
def cli():
    pass


@cli.command()
@click.argument('word_segments_filepath', type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=Path))
@click.argument('audio_filepath', type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=Path))
@click.option('--min_duration', '-mind', type=float, default=1.6, help='Minimum duration (seconds) of utterances')
@click.option('--max_duration', '-maxd', type=float, default=11, help='Maximum (seconds) of utterances')
@click.option('--score_threshold', type=float, default=0.97, help='Threshold score for word segments')
@click.option('--verbose', '-v', is_flag=True, help='print words and scores of each segments')
@click.option('--force_recreate', is_flag=True, help='Do not archive the speaker turns into subdirectoy "speaker_turns"')
def split_waveform_cli(word_segments_filepath, audio_filepath, min_duration, max_duration, score_threshold, verbose, force_recreate):   
    """
    A command-line interface function that splits waveform into subarrays based on
    word segments, and split the audio waveform array.
    
    Args:
        word_segments_filepath (str): The file path of the word segments file, or directory containing multiple 
                                      segments speaker-turns.
        audio_filepath (str): The file path of the audio file, or directory containing multiple audio speaker-turns.
        min_duration (float): The minimum duration (in seconds) of the segments. Default is 1.6.
        max_duration (float): The maximum duration (in seconds) of the segments. Default is 11.
        score_threshold (float): The threshold score for word segments. Default is 0.97.
        verbose (bool): Whether to print the words and scores of each segment. Default is False.
        force_recreate (bool): Whether to force the recreation of the speaker turns into subdirectories. 
                               Default is False.
    
    Returns:
        None
    """
        
    
    if word_segments_filepath.is_file() and audio_filepath.is_file():
        created_files = process(word_segments_filepath, audio_filepath, min_duration, max_duration, score_threshold, verbose)
        
        return

    # Inputs are directories.
    ignore_files = ['raw.wav', 'trimmed.wav']
    file_pairs = gather_word_audio_file_pairs_all_subdirs(audio_filepath, needs_words=True, ignore_files=ignore_files)

    for audiofile, wordfile, _ in file_pairs:
        if force_recreate:
            if ('utterance' in audiofile.stem) or ('utterance' in wordfile.stem):
                audiofile.unlink()
                wordfile.unlink()
                
        # Checks not already processed
        # TODO: this could be makefile pattern matching & file existence logic
        if (('utterance' in audiofile.stem) or ('utterance' in wordfile.stem)):
            # makes sure utterance transcript has been created for it
            print(f"Utterances already exist for {audiofile.name}, and {force_recreate=}. Skipping this utterance")
            # transcribe_utterance(wordfile)
            continue

        if verbose: print(f"Splitting speaker turns into shorter utterances. {wordfile, audiofile}")
        created_files = process(wordfile, audiofile, min_duration, max_duration, score_threshold, verbose)

@cli.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output_dir', type=click.Path(file_okay=False, path_type=Path))
@click.option('--from_diary', type=click.Path(exists=True, path_type=Path))
def split_audio(input_file: Path, 
                output_dir: Union[Path, None] = None, 
                from_diary: Union[Path, None] = None):
    """
    Split an audio file into multiple segments based on a given time & speaker name or a diary file.

    Args:
        input_file (str): The path to the input audio file.
        output_files (List[str]): A list of paths to the output audio files.
        from_diary (str): The path to the diary file containing the time ranges and speakers.

    Returns:
        None
    """
    
    diary_lines = Diarization(from_diary)
    time_ranges = []
    for line in diary_lines:
        time_ranges.append((line.start, line.stop, line['speaker']))
        
    out_dir = input_file.parent if output_dir is None else output_dir
    
    audio, sample_rate = librosa.load(input_file)
    for file_counter, (start, stop, speaker) in enumerate(time_ranges):
        start_sample = int(start * sample_rate)
        stop_sample = int(stop * sample_rate)
        speaker = speaker.replace(' ', '-')
        split_audio_wav = audio[start_sample:stop_sample]

        output_file = f"{speaker}-{file_counter}.wav"
    
        sf.write(out_dir / speaker / output_file, split_audio_wav, sample_rate)


def _consolidate_rows(diary_path: Union[str, Path], output_file_name: Union[str, Path, None] = None):
    """
    Consolidates rows from a diary file based on the speaker's name and time intervals.
    
    Args:
        diary_path (str): The path to the file containing the rows.
        output_file_name (Union[str, None], optional): The name of the output file. 
            If None, the consolidated rows will be printed to stdout. Default is None.
            
    Returns:
        List[Tuple[float, float, str]]: A list of tuples containing the start time, 
            end time, and speaker name for each consolidated row.
    """
    diary_lines = Diarization(diary_path)
    
    consolidated_rows = []

    current_speaker = diary_lines[0]['speaker']
    start_time = diary_lines[0].start
    end_time = diary_lines[0].stop

    for line in diary_lines:
        speaker = line['speaker']

        if current_speaker == speaker:
            end_time = line.stop
        else:
            consolidated_rows.append(Annotation(start=start_time,
                                                stop=end_time,
                                                content={'speaker': current_speaker}))
            current_speaker = speaker
            start_time = line.start
            end_time = line.stop

    consolidated_rows.append(Annotation(start=start_time, 
                                        stop=end_time, 
                                        content={'speaker': current_speaker}))

    if output_file_name is None:
        # Print to stdout
        for row in consolidated_rows:
            print(f"start: {row.start}, stop: {row.stop}, speaker: {row.speaker}")
    else:
        consolidated_rows = Diarization.from_annotations(consolidated_rows)
        consolidated_rows.save(to=output_file_name)

    return consolidated_rows

@cli.command()
@click.argument('diary_path', type=click.Path(exists=True))
@click.option('--output_file_name', type=str, default=None)
def consolidate_rows(diary_path: str, output_file_name: Union[str, None] = None):
    """
    Consolidates rows from a diary file based on the speaker's name and time intervals.
    
    Args:
        diary_path (str): The path to the file containing the rows.
        output_file_name (Union[str, None], optional): The name of the output file. 
            If None, the consolidated rows will be printed to stdout. Default is None.
            
    Returns:
        List[Tuple[float, float, str]]: A list of tuples containing the start time, 
            end time, and speaker name for each consolidated row.
    """
    
    _consolidate_rows(diary_path, output_file_name)


@cli.command()
@click.argument('index_file', type=click.Path(exists=True, path_type=Path))
@click.argument('transcript', type=click.Path(exists=True, path_type=Path))
def divide_long_text(index_file: Path, transcript: Path):
    """
    Saves the transcript as multiple parts in a directory of the original file name, 
    each ending in `segment_{i}.txt`.
    
    Example index file;
    0| So basically we started off in middle school.
    1| *toward post-traumatic growth.
    2| *both in males and females. 
    
    0: Transcript matching this text becomes the first transcript
    1: remaning undividide transcript, ending in this text becomes the next.
    2: same
    """
    
    # TODO: the wildcard is not necessary in the input index file.
    
    with open(index_file, 'r') as i_file:
        index_lines = i_file.readlines()
        
    with open(transcript, 'r') as t_file:
        long_text = t_file.read()


    long_text = long_text.replace(word_break_sym, ' ')

    segments = []

    for i, substring in enumerate(index_lines):
        
        substring = substring.split(word_break_sym)[1].strip()
        wildcard_match = substring.startswith('*')
        substring = _process_content(substring.strip())
        substring = ('*' if wildcard_match else '') + substring
        
        substring  = substring.replace(word_break_sym, ' ')
        pattern = re.escape(substring) if '*' not in substring else substring.replace('*', '.*')
        match = re.search(pattern, long_text)

        if match:
            segment = long_text[:match.end()]
            segments.append(segment)
            long_text = long_text[match.end():]
        else:
            click.echo("Break")
            break
    
        click.echo(index_file.parent / f"segment_{i}.txt")
        with open(index_file.parent / f"segment_{i}.txt", 'wt' ) as f:
            f.write(segment.strip(' ').replace(' ', word_break_sym))


def _rmv_array_section(waveform: np.ndarray, 
                       intervals: List[Tuple[int,int]]):
    """
    Removes a section of an array based on a list of tuples representing the start and stop indices of each section.

    Args:
        waveform (np.ndarray): The input array from which the intervals will be removed.
        intervals (List[Tuple[int,int]]): A list of tuples representing the start and stop indices of each section to be removed.

    Returns:
        np.ndarray: The resulting array after removing the specified intervals.
    """
    
    mask = np.ones_like(waveform, dtype=bool)
    indices_to_exclude = np.concatenate([np.arange(start, stop) for start, stop in intervals])
    mask[indices_to_exclude] = False
    return waveform[mask]


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--remove_front', '-f', type=float, default=0)
@click.option('--remove_end', '-e', type=float, default=0)
@click.option('--remove_sections', '-s', type=str, default=None)
@click.option('--output_file', type=click.Path(dir_okay=False, path_type=Path))
def shorten_audio(input_file, remove_front=0, remove_end=0, remove_sections = None, output_file: Optional[Path] = None):  # used
    """
    # Example usage
    >>> input_file = 'path/to/raw.wav'
    >>> remove_remove_front = 2  # Remove 2 seconds from the front
    >>> remove_end = 3  # Remove 3 seconds from the end
    >>> remove_sections = "4.3-5.3,6-10"  # Remove sections between 4.3-5.0 seconds, as well as 6-10 seconds
    >>> shorten_audio(input_file, output_file, remove_remove_front, remove_end)
    """
    
    # Load the input audio file
    audio, sample_rate = sf.read(input_file.resolve())

    if output_file is None:
        output_file = input_file.parent / "trimmed.wav"

    remove_indices = []
    # Calculate the number of samples to remove from the front
    if remove_front:
        samples_to_remove_front = int(remove_front * sample_rate)
        remove_indices.append((0, samples_to_remove_front))
        
    # Calculate the number of samples to remove from the end
    if remove_end:
        samples_to_remove_end = int(remove_end * sample_rate)
        remove_indices.append((audio.shape[0] - samples_to_remove_end, audio.shape[0]))
    
    if remove_sections:
        # Remove sections between 4.3 seconds and 5  seconds, as well as 6-10 seconds: "4.3-5.3,6-10"
        crops = []
        for section in remove_sections.split(','):
            cut_after_seconds, cut_before_seconds = section.split('-')
            print(f"{cut_after_seconds=} {cut_before_seconds=}")
            cut_after, cut_before = int(sample_rate * float(cut_after_seconds)), int(sample_rate * float(cut_before_seconds))
            crops.append((cut_after, cut_before))
        remove_indices = crops  # np.array(crops)
    
    if len(remove_indices) == 0:
        click.echo("Error - must specify one option: --remove_front, --remove_end or --remove_sections")
        return
    # Remove the specified number of samples from the front and/or end of the audio
    shortened_audio = _rmv_array_section(audio, remove_indices)

    # Save the shortened audio as a new file
    sf.write(output_file, shortened_audio, sample_rate)
    print(f"Shortened wav by {(audio.shape[0] - shortened_audio.shape[0]) / sample_rate} seconds: {output_file}")


def shuffle_speaker_names(speaker_names, input_file):
    """Rotate the speaker names in a file, until it matches the pattern of speaker name supplied.
    
    # Example usage
    speaker_names = ["David Patman", "Susan Long", "David Patman", "Franca Fubini"]
    input_file = 'diary3.txt'

    new_speaker_order = shuffle_speaker_names(speaker_names, input_file)
    """
    lines = Diarization(input_file)

    # Find the number of rows to cycle
    cycle_count = 0
    file_speaker_names = [row.content['speaker'] for row in lines]
    # First check the nominated order occurs in the file
    while file_speaker_names[cycle_count:cycle_count+len(speaker_names)] != speaker_names:
        cycle_count += 1
        if cycle_count >= len(lines):
            raise ValueError("The order of names in speaker_names does not occur in the file.")

    new_speaker_order = list(islice(cycle(file_speaker_names), 
                                    cycle_count, 
                                    cycle_count + len(file_speaker_names)))
    # Cycle the speaker names in the file
    for i in range(len(lines)):
        lines[i].content['speaker'] = new_speaker_order[i]

    return lines


@cli.command()
@click.argument('file_path', type=click.Path(exists=True, path_type=Path))
@click.option('--n', default=1, type=int)
@click.option('--output_file_name', '-out', type=str)
@click.option('--speaker_order', '-order', type=str)
def cycle_speakers(file_path: str, n: int = 1, output_file_name: Optional[str] = None, speaker_order: Optional[str] = None):
    """Rotate/Cycle the order of speakers in a diary file by N, or until it matches a nominated order.
    
    Optionally - specifiy the new speaker order as ","-separated string.
    Optionally - save the modified content to a new file.

    Args:
        file_path (str): The path to the input file.
        output_file_name (Union[str, None], optional): The name of the output file. If None, the modified content will be printed to stdout. Defaults to None.
        speaker_order (str, optional): The order of speakers to follow. If None, the modified content will be printed to stdout. Defaults to None.

    Returns:
        List[str]: The modified content as a list of strings.
    """
    
    output_file_path = None
    if output_file_name is not None:
        output_file_path = str(file_path).replace(file_path.stem, output_file_name)
        
    if speaker_order is not None:
        speaker_order = speaker_order.split(',')
        lines = shuffle_speaker_names(speaker_order, file_path)
        # Write the shuffled speaker names to the output file
        if output_file_path:
            lines.save(to=output_file_path)
        else:
            print(lines.to_string())
        return

    lines = Diarization(file_path)

    new_speakers_order = deque([line.content['speaker'] for line in lines])
    # Rotate the speaker order forward by 'n'
    new_speakers_order.rotate(n)
    
    for i in range(len(lines)):
        lines[i].content['speaker'] = new_speakers_order[i]

    assert len(speaker_order) == len(set(new_speakers_order)), \
            "a nominated speaker order to follow must have all speakers found"
    
    if output_file_path is None:
        # Print to stdout
        print(lines.to_string())
    else:
        print(f"Cycled speakers, Saving to {output_file_path=}")
        lines.save(to=output_file_path)
        
    return lines


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
def remove_first_line(file_path):
    """
    Removes the first line from the file specified by the given file path.

    Args:
        file_path (str): The path to the file.

    Returns:
        None
    """
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Remove the first line
    lines = lines[1:]

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)


if __name__ == '__main__':
    cli()
