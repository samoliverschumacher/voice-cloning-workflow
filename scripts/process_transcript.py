from itertools import dropwhile, takewhile
from pathlib import Path
from string import ascii_letters, printable, punctuation, whitespace
from typing import Tuple

import click

from structures import SpeakerTurn, Transcript

"""
    These scripts process text by converting words and letters into valid tokens according to the text-to-mel model
"""

punctuation += '–' + '’' + '…' + "—" # '\u2013' + '\u2019' + '\u2026'
word_break_sym = '|'  # Wav2Vec2 model uses as a word boundary. RTVC later uses ","
speaker_delimiter = '|'
pounds = '£'  # '\u00A3'
dollars = '$'  # '\u0024'

__all__ = ['speaker_delimiter', '_process_content', 'word_break_sym']

def _resolve_non_printable(transcript: str, verbose=False):
    cleaned_transcript = ''
    for i, char in enumerate(transcript):
        if char not in printable:
            if verbose:
                max_i = min(i+15, len(transcript))
                min_i = max(i-15, 0)
                print("found non-printable: ", transcript[min_i: max_i], repr(char), "removing..")
            prior_character = transcript[i-1] if i > 0 else ''
            after_character = transcript[i+1] if i < len(transcript)-1 else ''
            
            # Non-printable character is next to a space, so remove it.
            if prior_character == ' ' or after_character == ' ':
                cleaned_transcript += ''
            else:
                # Non-printable character is supposed to be a space
                cleaned_transcript += ' '
        else:
            cleaned_transcript += char
    return cleaned_transcript


def _numbers_to_words(word: str) -> str:
    from num2words import num2words
    
    word = word.lower()
    
    if word.endswith('th') | word.endswith('st') | word.endswith('nd') | word.endswith('rd'):
        num = ''.join(takewhile(lambda c: c.isdigit(), word))
        return num2words(num, to='ordinal').strip()
    
    if word.endswith('%'):
        num = ''.join(takewhile(lambda c: c.isdigit(), word))
        return f"{num2words(num)} percent".strip()
    
    if word.startswith('19') and len(word) == 4:  # i.e. 1998
        return f"{num2words(word[:2])} {num2words(word[2:])}"
        
    if word.startswith('20') and len(word) == 4:  # i.e. 2001
        return f"{num2words(word[:2])} {num2words(word[2:])}"
    
    # Word has all numbers, then all ascii letters
    number_part = ''.join(takewhile(lambda c: c not in ascii_letters, word))
    letter_part = ''.join(dropwhile(lambda c: c not in ascii_letters, word))
    # examples: "80s" -> eightys
    if len(number_part + letter_part) == len(word) and word.endswith('s'):
        
        lookup = {'10s': 'tens',
                  '20s': 'twenties',
                  '30s': 'thirties',
                  '40s': 'forties',
                  '50s': 'fifties',
                  '60s': 'sixties',
                  '70s': 'seventies',
                  '80s': 'eighties',
                  '90s': 'nineties',
                  '00s': 'aughts'}
        if word.startswith('19'): # i.e. 1960s
            return f'nineteen {lookup[word[2:]]}'
            
        return lookup[word]
        
    if len(number_part + letter_part) == len(word) and len(letter_part) > 0:
        return f"{num2words(number_part) + letter_part}".strip()
    
    if len(word) >= 4:  # i.e. 1871, or 18231 ~ 'eighteen thousand, two hundred and thirty-one'
        return num2words(word).strip().replace(',', ' ').replace('-', ' ')
    
    return num2words(word).strip()
     
@click.command()
@click.argument('transcript_file', type=click.Path(exists=True))
@click.option('--save_batch', '--sb', is_flag=True, help='Save the content as entire processed transcript')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose mode')
def partition_transcript(transcript_file, 
                         save_batch: bool = False, 
                         verbose=False):
    """
    Reads a transcript file and partitions specific rows of the transcript into separate text files.
    
    Example:
    >>> python partition-transcript.py data/01_interim/AIM-1418/transcript.txt --verbose
    
    Args:
        transcript_file (str): The path to the transcript file.
        save_batch (bool): If True, also save the processed transcript as a multiline file.
        verbose (bool, optional): Whether to print additional information during the partitioning process. 
            Defaults to False.
    
    Returns:
        None
    """

    transcript_lines = Transcript(transcript_file)
        
    for transcript_row in transcript_lines:
        transcript_row: SpeakerTurn
        
        save_filepath, processed_transcript = process_transcript_row(transcript_row, transcript_file, verbose)
        
        transcript_row = transcript_row._replace(transcript=processed_transcript)
        
        if verbose: 
            print(f"{save_filepath=}")

        save_filepath.parent.mkdir(exist_ok=True)
        with open(save_filepath, 'w') as f:
            f.write(processed_transcript)
        
    if save_batch:
        # Save the processed transcription
        processed_filepath = transcript_file.with_stem(transcript_file.stem + '-processed')
        transcript_lines.save(to=processed_filepath)


def process_transcript_row(transcript_row: SpeakerTurn, transcript_file, verbose=False) -> Tuple[Path, str]:
    """
    Process a transcript row and save the processed transcript to a file.

    Args:
        transcript_row (SpeakerTurn): The speaker turn object containing the transcript and speaker information.
        transcript_file (str): The file path of the transcript.
        verbose (bool, optional): Whether to print additional information during the partitioning process. 
            Defaults to False.
    
    Returns:
        tuple: A tuple containing the save file path and the processed transcript.
    """
    transcript = transcript_row.transcript
    speaker_name = transcript_row.speaker.replace(' ', '-')
    # TODO: centralise directory structure in separate python script for importing filepath consistency.
    save_filename = f"{speaker_name.strip().replace(speaker_delimiter, '').replace(' ','-')}-{transcript_row.number}.processed.txt"
    save_filepath = Path(transcript_file).parent / speaker_name / save_filename
        
    processed_transcript = _process_content(transcript)
        
    return save_filepath, processed_transcript

# TODO: could this be replaced with BigPhoney ?
def _process_content(transcript: str, verbose=False) -> str:
    # Non-printable characters could be either spaces or added irrelevant characters
    transcript = _resolve_non_printable(transcript, verbose=verbose)
            
    # One at a time, convert "$25, £25" to "25 dollars, 25 pounds", and " £ ", to " pounds "
    currency_replacement = ('£', 'pounds')
    while transcript.find(currency_replacement[0]) != -1:
        transcript = _resolve_currency(transcript, currency_replace=currency_replacement)
    currency_replacement = ('$', 'dollars')
    while transcript.find(currency_replacement[0]) != -1:
        transcript = _resolve_currency(transcript, currency_replace=currency_replacement)        

    word_breaks = ''.join(set(punctuation) - set("$'%"))
    if verbose:
        print(f"{word_breaks=}")
        print(f"{whitespace=}")
        # replace word-break punctuation (except apostrophes) with |, then remove all apostrophes
    for s in word_breaks:
        transcript = transcript.replace(s, word_break_sym)
    transcript = transcript.replace("'", '')

    for s in whitespace:
        transcript = transcript.replace(s, word_break_sym)
        
        # convert numbers to words (4 -> "four", 3rd - "third")
    result = []
    for word in transcript.split(word_break_sym):
        if any([c.isdigit() for c in word]):
            worded_number = _numbers_to_words(word)
            for s in whitespace:
                worded_number = worded_number.replace(s, word_break_sym)
            result.append(worded_number)
            continue
        
        result.append(word)
    transcript = word_break_sym.join(result)
        
    transcript = transcript.upper()
    # word breaks (|) occuring in sequence are merged.
    last_c = ''
    merged_transcript = ''
    for c in transcript:
        if c == word_break_sym and last_c == word_break_sym:
            continue
        else:
            merged_transcript += c
        last_c = str(c)
        # word breaks (|) at end and start are removed.
    if merged_transcript.startswith(word_break_sym):
        merged_transcript = merged_transcript[1:]
        
    if merged_transcript.endswith(word_break_sym):
        merged_transcript = merged_transcript[:-1]
    
    return merged_transcript


def _resolve_currency(text, currency_replace: Tuple[str, str] = ("£", "pounds")):
    """Replace currency symbols with words
    
    "whereas Mensa, all £ you have to do is turn up and pay £25, and then you"
    Converts to;
    >>> "whereas Mensa, all  pounds  you have to do is turn up and pay 25 pounds , and then you"
    """
    # Find the index of the curreency symbol (i.e. '£')
    symbol, word = currency_replace
    index = text.find(symbol)

    # Check if '£' is found and if the characters after '£' are integers
    if index != -1:
        index_2 = index + 1
        for char in text[index+1:]:
            if char.isdigit():
                index_2 += 1
                continue
            else:
                processed_text = text[:index] + text[index+1:]
                index_2 -= 1
                # Replace '£' with the word 'pounds', after the integers
                processed_text = processed_text[:index_2] + f' {word} ' + processed_text[index_2:]
                break
        return processed_text
    else:
        return text


if __name__ == '__main__':
    partition_transcript()
