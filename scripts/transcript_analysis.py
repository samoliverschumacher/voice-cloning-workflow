import io
import os
import re
import sys
from itertools import dropwhile, takewhile
from pathlib import Path
from typing import Optional, Union

import click
import pandas as pd
import syllables  # pip install syllables

from process_transcript import word_break_sym
from structures import Diarization, to_frame


def diary_to_frame(diary_filepath):
    
    data = to_frame(Diarization(diary_filepath))
    return data


def time_above_threshold(audio_filepath: Union[str,Path], 
                         probability_threshold: float, 
                         start: Optional[float] = None, 
                         stop: Optional[float] = None):
    """
    Calculates the total time that the audio file is above a given probability threshold.

    Parameters:
        audio_filepath (str): The filepath of the audio file.
        probability_threshold (float): The probability threshold above which the audio is considered speech.
        start (float, optional): The start time in seconds from which to calculate the total time. Defaults to None.
        stop (float, optional): The stop time in seconds until which to calculate the total time. Defaults to None.

    Returns:
        float: The total time in seconds that the audio file is above the given probability threshold.
    """
    
    audio_filepath = Path(audio_filepath)
    speech_activity_filepath = audio_filepath.parent / f"speech_probability.csv"
    # load the speech probability if it exists, or create a new
    if speech_activity_filepath.exists():
        with open(speech_activity_filepath, 'r') as f:
            speech_activity = f.read().splitlines()
    else:
        speech_probability, duration, _ = speech_proba(audio_filepath)
        s_p_step = duration / len(speech_probability)
        speech_activity = []
        for i, row in enumerate(speech_probability):
            speech_activity.append(f"{i*s_p_step:.4f}s, {row[0]}")

    sum_time = 0.0
    last_time = 0.0
    for row in speech_activity:
        time, probability = row.split(',')

        time = float(time[:-1])  # Remove 's' from the time value
        probability = float(probability)
        if (start is None or time >= start) and \
            (stop is None or time <= stop) and \
            (probability > probability_threshold):
            sum_time += (time - last_time)
        last_time = time

    return sum_time


def words_per_min(diary_filepath, ostream=sys.stdout):
    """Writes to ostream a csv: <words_per_min=N, word_count=N, length_seconds=N, filename=name>.
    
    Expects lines in diary_filepath like: "start: 7.124573378839591, stop: 16.919795221843003, speaker: John Smith"."""
    transcripts_dir = Path(diary_filepath).parent
    diary_rows = Diarization(diary_filepath)

    for row, line in enumerate(diary_rows):
        start = line.start
        stop = line.stop
        name = line.content['speaker'] 

        # Form filepaths from speaker rows
        speaker_directory = Path(transcripts_dir) / name.replace(' ', '-') 
        single_speaker_transcript_filepath = speaker_directory / f"{name.replace(' ', '-')}-{row+1}.processed.txt"
        if not single_speaker_transcript_filepath.exists():
            ostream.write(f"File not found: {str(single_speaker_transcript_filepath)}\n")
            continue
        
        with open(single_speaker_transcript_filepath) as f:
            word_count = len(f.read().split(word_break_sym))
        length_seconds = (stop - start)
        words_per_min = word_count / (length_seconds / 60)
        ostream.write(f"{words_per_min=:.0f}, {word_count=}, {length_seconds=:.2f}, {start=:.2f}, {stop=:.2f}, filename={single_speaker_transcript_filepath.stem}\n")


def speech_proba(audio_filepath):
    """Calculate and save the speech probability of a podcast audio file.
    
    Parameters:
        audio_filepath (str): Path to the podcast audio file.
    """
    import numpy as np
    import torch
    from pyannote.audio import Model
    from pyannote.audio.core.io import Audio
    HUGGING_FACE_API_AUTH_TOKEN = os.environ.get("HUGGING_FACE_API_AUTH_TOKEN")
    if HUGGING_FACE_API_AUTH_TOKEN is None:
        raise ValueError("HUGGING_FACE_API_AUTH_TOKEN is not set.")
    
    audio = Audio(sample_rate=16000)
    duration = audio.get_duration(audio_filepath)
    waveform, sample_rate = audio({"audio": audio_filepath})
    model = Model.from_pretrained("pyannote/segmentation", use_auth_token=HUGGING_FACE_API_AUTH_TOKEN)
    with torch.no_grad():
        segmentation = model(waveform[np.newaxis]).numpy()[0]
    speech_probability = np.max(segmentation, axis=-1, keepdims=True)
    return speech_probability, duration, sample_rate


def filter_wpm(file_buffer, word_per_min=None, word_count=None, length_seconds=None):
    """filter diary analysis
    
    Example;
    >>> buffer = io.StringIO()
    >>> words_per_min("diary.txt", ostream=buffer)
    >>> buffer.seek(0)
    >>> filtered_rows = filter_wpm(buffer, word_per_min=">100", word_count="<100", length_seconds=">0.5")
    >>> for row in filtered_rows:
    >>>     print(row)
    """
    
    filtered_rows = []
    for line in file_buffer:
        # Extract values from the line
        values = line.strip().split(', ')
        words_per_min = int(values[0].split('=')[1])
        count = int(values[1].split('=')[1])
        minutes = float(values[2].split('=')[1])
        
        # Check if the row satisfies the filter conditions
        if (word_per_min is None or eval(f"{words_per_min} {word_per_min}")) and \
           (word_count is None or eval(f"{count} {word_count}")) and \
           (length_seconds is None or eval(f"{minutes} {length_seconds}")):
            filtered_rows.append(line.strip('\n'))
    
    return filtered_rows


def _turn_file_number(filename):
    return ''.join(takewhile(str.isdigit, dropwhile(lambda c: not c.isdigit(), filename)))


@click.group()
def cli():
    pass


def count_syllables(words):
    # Isolate the long import time of BigPhoney to the function level
    from big_phoney import \
        BigPhoney  # pip install big-phoney. Then apply patch: https://github.com/repp/big-phoney/pull/8/commits/580d5a582e445510d28a6270aa16453ed868151e
    phoney = BigPhoney()
    ns=[]
    
    for word in words:
        if phoney.phonetic_dict.lookup(word) is not None:
            ns.append(phoney.count_syllables(word))
        else:
            ns.append(syllables.estimate(word))
    return ns


def summarise_syllables(word_alignment_file, verbose=True, return_data=False):
    """
    Summarizes the syllables in a given word alignment file.

    Parameters:
        word_alignment_file (str): The path to the word alignment file.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        return_data (bool, optional): Whether to return the data. Defaults to False.

    Returns:
        None: If return_data is False.
        pd.DataFrame: The summarized data if return_data is True.
    """
    data = pd.read_json(word_alignment_file)
    data['word_duration'] = data['end_ts'] - data['start_ts']
    
    ns = count_syllables(data['label'])
    data['num_syllables'] = ns
    
    data['duration_per_syllable'] = data['word_duration'] / data['num_syllables']

    quantile_90_percent = data.duration_per_syllable.quantile(q=0.9)
    q_id = data.duration_per_syllable >= quantile_90_percent
    if verbose:
        with pd.option_context('display.max_rows', None, 'display.precision', 2):
            print('')
            print("Words with syllables over the 90-percentile duration;")
            print(data.loc[q_id,['label', 'start_ts', 'word_duration', 'duration_per_syllable']])
        
            print('\nSummary;')
            print(data[['word_duration', 'duration_per_syllable']].describe())
    
    if return_data:
        return data
    return None


@cli.command
@click.argument('words_directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--turn_number', '-n', type=str, help="Comma-serpated Speaker turn numbers to select '*words.json' "
                                                     "file from words_directory")
@click.option('--quantile', '-q', default=">95", type=str, help="Quantile of seconds per syllable to filter-in above")
@click.option('--word_duration', '-wd', default=">1.5", type=str, help="Duration (seconds) of word to filter-in above")
@click.option('--summary_only', '-s',is_flag=True, help="Duration (seconds) of word to filter-in above")
def spm(words_directory, turn_number: Optional[str]=None, quantile=">98", word_duration=">1.5", summary_only=False):
    """Prints statistical summary of syllable counts."""
    
    quantile_direction = "above"
    if "<" in quantile:
        quantile_direction = "below"
        _, quantile = quantile.split('<')
    else:
        _, quantile = quantile.split('>')
    quantile = int(quantile)
    assert 1 <= quantile <= 100, "quantile has to be 1-100"
    
    word_direction = "above"
    if "<" in word_duration:
        word_direction = "below"
        _, word_duration = word_duration.split('<')
    else:
        _, word_duration = word_duration.split('>')
    word_duration = float(word_duration)
    assert word_duration > 0, "word duration has to be > 0"
        
    if turn_number is not None:
        turn_number = [int(n) for n in turn_number.split(',')]
    
    datas = []
    speaker_directories = [p for p in words_directory.glob('*') if p.is_dir()]
    for speaker_dir in speaker_directories:
        for words_file in speaker_dir.glob("*-alignments.json"):
            if not re.match(r".*\d+-alignments\.json$", str(words_file)):
                continue
            
            if turn_number:
                if not any([re.match(r".*-" + f"{n}" + r"*-alignments\.json$", str(words_file)) for n in turn_number]):
                    continue
            
            data = summarise_syllables(words_file, verbose=False, return_data=True)
            data['filename'] = words_file.stem
            data['turn_file_number'] = _turn_file_number(words_file.stem)
        
            datas.append(data)
            
    if not datas:
        raise FileNotFoundError(f"No data found in {words_directory=}")
    df = pd.concat(datas)
    df['syllables_per_second'] = 1 / df['duration_per_syllable']
    
    quantile_percent = df['duration_per_syllable'].quantile(q=quantile/100)
    if quantile_direction == 'above':
        q_id = df['duration_per_syllable'] >= quantile_percent
    else:
        q_id = df['duration_per_syllable'] <= quantile_percent
    
    if word_direction == 'above':
        w_id = df['word_duration'] >= word_duration
    else:
        w_id = df['word_duration'] <= word_duration
    filters = pd.concat([w_id, q_id],axis=1).any(axis=1)
    
    summary = df[['filename','word_duration', 'num_syllables']].groupby('filename').agg(['mean','sum','median','std'])
    
    with pd.option_context('display.max_rows', None, 'display.precision', 2):
        if not summary_only:
            print('')
            print(f"Words with syllables {quantile_direction} the {quantile}-percentile duration-per-syllable and per-word")
            print(df.loc[filters,['label', 'start_ts', 'num_syllables', 'syllables_per_second', 'word_duration', 'duration_per_syllable', 'turn_file_number', 'filename']])
    
        print('\nPer-word Summary;')
        print(df[['word_duration', 'duration_per_syllable']].describe())
        
        print('\nPer-file syllabels per second Summary;')
        print(pd.Series(summary['num_syllables']['sum'] / summary['word_duration']['sum']).describe())
        
        print(f"Syllables per second in normal speech should be: 3.3 - 5.9")
    

@cli.command(name='wpm')
@click.argument('diary_filepath')#, help='Path to the diary file')
@click.option('--word_per_min', '-wpm', help='Filter by words per minute')
@click.option('--word_count', '-wc', help='Filter by word count')
@click.option('--length_seconds', '-l', help='Filter by length in minutes')
def wpm(diary_filepath, word_per_min, word_count, length_seconds):
    """
    Calculate and filter rows from a diary file based on words per minute, word count, and length in minutes.

    Example;
    >>> python transcript_analysis.py wpm transcripts/AIM-1418/diary2.txt -wpm ">250" -wc ">20" -l ">0.1"
    
    Parameters:
        diary_filepath (str): Path to the diary file.
        word_per_min (int): Filter rows based on the minimum words per minute.
        word_count (int): Filter rows based on the minimum word count.
        length_seconds (int): Filter rows based on the minimum length in minutes.

    Returns:
        None
    """

    buffer = io.StringIO()
    words_per_min(diary_filepath, ostream=buffer)
    buffer.seek(0)
    if word_per_min or word_count or length_seconds:
        filtered_rows = filter_wpm(buffer, word_per_min=word_per_min, word_count=word_count, length_seconds=length_seconds)
    else:
        filtered_rows = buffer.readlines()

    if len(filtered_rows) == 0:
        click.echo("\t... No speaker turns made the words per minute filter...")
    for row in filtered_rows:
        click.echo(row.rstrip('\n'))


@cli.command(name='save-speech-probability')
@click.argument('podcast_audio_filepath')
@click.option('--save_dir', help='Directory to save the speech probability file')
def save_speech_probality(podcast_audio_filepath, save_dir=None):
    """
    Calculate and save the speech probability of a podcast audio file.

    Parameters:
        podcast_audio_filepath (str): Path to the podcast audio file.
        save_dir (str, optional): Directory to save the speech probability file. If not provided, the file will be saved in the same directory as the podcast audio file.

    Returns:
        None
    """

    if save_dir is not None:
        save_filepath = Path(save_dir) / f"speech_probability.csv"
    else:
        save_filepath = Path(podcast_audio_filepath).parent / f"speech_probability.csv"

    speech_probability, duration, _ = speech_proba(podcast_audio_filepath)
    s_p_step = duration / len(speech_probability)

    with open(save_filepath, 'w') as f:
        for i, row in enumerate(speech_probability):
            f.write(f"{i*s_p_step:.4f}s, {row[0]}\n")


@cli.command(name='check-speaker-order')
@click.argument('diary_filepath', type=click.Path(exists=True, path_type=Path, resolve_path=True))
@click.option('--transcript', type=click.Path(exists=True))
def check_speaker_order(diary_filepath, transcript):
    
    if not transcript:
        transcript = diary_filepath.parent / "transcript.txt"
    
    from structures import Transcript
    transcript_lines = Transcript(transcript)
        
    with open(diary_filepath, 'r') as f_b:
        diary_lines = f_b.readlines()
    diary_lines = Diarization(diary_filepath)

    for transcript_turn, diary_turn in zip(transcript_lines, diary_lines):
        if transcript_turn.speaker != diary_turn.content['speaker']:
            click.echo("Speaker name mismatch!!\n\t"
                       f"In the transcript, At {transcript_turn.number=}, "
                       f"{transcript_turn.speaker=}, but {diary_turn.content['speaker']=}", 
                       file=sys.stdout)
            sys.exit(1)

    click.echo("Speaker names match in both files.")


@cli.command(name='time-above-threshold')
@click.argument('audio_filepath', type=click.Path(exists=True))
@click.option('--probability_threshold', default=0.9, help='Probability threshold', type=float)
@click.option('--start', help='Start time', type=float)
@click.option('--stop', help='Stop time', type=float)
def time_above_threshold_cli(audio_filepath, probability_threshold, start, stop):
    """
    Calculate the total time above a probability threshold within a specified time range.

    Parameters:
        audio_filepath (str): Path to the audio file.
        probability_threshold (float, optional): Probability threshold. Defaults to 0.9.
        start (float, optional): Start time.
        stop (float, optional): Stop time.

    Returns:
        float: Total time above the probability threshold.
    """
    total_time = time_above_threshold(audio_filepath, probability_threshold, start, stop)
    click.echo(f"Total time above threshold: {total_time} seconds.")


@cli.command()
@click.argument('directory', type=click.Path(exists=True, path_type=Path))
def calc_wordscores(directory):
    """
    Calculate word scores for files in a given directory.

    Args:
        directory (str): The path to the directory containing the files.

    Returns:
        None
    """
    # Create the dataframe
    df = wordscore_dataframe(directory)
        
    # Calculate metrics
    result = df.groupby('filename', as_index=False).agg({'score': 'mean', 'duration': 'sum'})

    result['number'] = result['filename'].apply(lambda c: int(_turn_file_number(c)))
    # Sort the dataframe by the 'number' column in ascending order
    
    result = result.sort_values('number')
    
    # Calculate the thresholds for bottom 25% and top 25% scores
    score_threshold_25 = result['score'].quantile(0.25)
    score_threshold_75 = result['score'].quantile(0.75)
    ABOVE_75_SCORE = ' +'
    BELOW_25_SCORE = ' -'
    
    def _score_symbol(score: str) -> str:
        if float(score) >= score_threshold_75:
            return ABOVE_75_SCORE
        if float(score) <= score_threshold_25:
            return BELOW_25_SCORE
        return ' '
                                                   
    result['sl'] = result['score'].apply(_score_symbol)

    result['score'] = result['score'].apply(lambda x: f"{x:.2f}")
    result['duration'] = result['duration'].apply(lambda x: f"{x:.2f} sec")
    
    # Aggregate metrics for the entire data
    total_score = df['score'].mean()
    total_duration = df['duration'].sum()

    # Print the result
    result = result.rename(columns={'score': 'avg. score'})
    print(result[['filename', 'avg. score', 'sl', 'duration']].to_csv(index=False))
    print(f"Total, {total_score:.2f}, , {total_duration:.2f} sec")


def wordscore_dataframe(directory: Path):
    """
    Generate a pandas DataFrame by reading multiple JSON files of word scores, from a directory.

    Args:
        directory (Path): The directory containing the JSON files.

    Returns:
        DataFrame: A combined DataFrame containing data from all the JSON files.
    """
    filepaths = []
    dirs = [p for p in directory.glob("*") if p.is_dir()]
    for speaker_dir in dirs:
        for file in speaker_dir.glob("*-alignments.json"):
            filepaths.append(file)

    dfs = []
    for filepath in filepaths:
        data = pd.read_json(filepath)
        data['rolling_avg'] = data['score'].rolling(window=6).mean()
        data['filename'] = filepath.stem
        dfs.append(data)
        
    combined_df = pd.concat(dfs)
    combined_df['duration'] = (combined_df['end_ts'] - combined_df['start_ts'])
    return combined_df


if __name__ == '__main__':
    cli()
