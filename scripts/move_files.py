import json
import re
import shutil
from itertools import filterfalse, takewhile
from pathlib import Path
from typing import List, Tuple

import click
import torchaudio

INTERIM_DATA_DIR = Path("../data/01_interim")


def _trim_alignments_file(afile: Path):
    """Alignments file must contain only the files that exist in the directory 
    next to it. Removes ones lines dont"""
    
    speaker = afile.parent.parent.name
    program = afile.parent.name
    
    with open(afile, 'r') as f:
        data = f.readlines()
    
    with open(afile, 'w') as f:
        for line in data:
            wavfname, content, times = line.split(' ')
            wavfname = wavfname.replace('"', '')
            
            if not wavfname.startswith(speaker):
                continue
            newline = ' '.join([f"{program}_{wavfname}", content, times])
            f.write(newline)


def reorganize_files_for_encoder(old_directory, new_directory, force_recreate=False):
    """Moves wav files to speaker-program directory from their original program-speaker directory.
    
    Also edits then copies alignment files to new directory.
    
    Renames audiofiles: {program_name}_{identifier}.wav
    """

    for program_dir in Path(old_directory).glob('*'):
        
        if not valid_dataset(Path(str(program_dir))):
            print(f"{program_dir.relative_to(old_directory)} is not a clean dataset. Skipping.")
            continue
            
        # Read the metadata from the JSON file
        with open(program_dir / "metadata.json") as f:
            metadata = json.load(f)
        
        group_name = metadata["group"]    
        program_name = program_dir.name
        alignments_file_is_moved = False
        moved_alignment_files = set()

        audio_files = gather_word_audio_file_pairs_all_subdirs(program_dir,
                                                               needs_words=True)

        
        for speaker_audio_file, words_file, identifier in audio_files:
            speaker_name = str( ''.join(takewhile(lambda c: not c.isdigit(), 
                                                  identifier ))).strip('-')  # Vince-Polito-31-utterance6.wav -> Vince-Polito-
            
            # Make the directory structure if not exist
            new_prog_dir = Path(new_directory) / group_name / speaker_name / program_name
            new_prog_dir.mkdir(parents=True, exist_ok=True)
            
            # Move the speaker-turn audio file
            new_audio_file = new_prog_dir / f"{program_name}_{identifier}.wav"
            if not new_audio_file.exists() or force_recreate:
                shutil.copy2(speaker_audio_file, new_audio_file)

            # Move the alignments file
            new_alignmemnts_file = new_prog_dir / "transcript.alignment.txt"
            
            # alignment file in new directory is one per speaker per program, not one per program.
            if new_alignmemnts_file not in moved_alignment_files:
                shutil.copy2(program_dir / "transcript.alignment.txt", new_alignmemnts_file)
                
                # edit alignments file so it only contains that speaker name
                _trim_alignments_file(new_alignmemnts_file)
                alignments_file_is_moved = True
                
                print(f"{alignments_file_is_moved=}, new_alignmemnts_file={new_alignmemnts_file.relative_to(Path(new_directory))}")
                moved_alignment_files.add(new_alignmemnts_file)
                    
    print("Files moved to new directory structure.")
        
# TODO: SINCE ITS RTVC specific, should be a method on a class, or in a file to show its for a specific purpose
def create_text_string(json_file: Path, duration: float):
    with open(json_file) as file:
        data = json.load(file)
    
    wav_filename = json_file.stem
    labels = [obj["label"] for obj in data]
    end_times = [str(obj["end_ts"]) for obj in data]
    
    # prepend and append blanks
    labels = [""] + labels + [""]
    end_times = [str(data[0]['start_ts'])] + end_times + [str(duration)]
    
    text_string = f'{wav_filename} "{",".join(labels)}" "{",".join(end_times)}"'
    return text_string


def get_duration(file):
    
    info = info = torchaudio.info(file)

    frames = info.num_frames
    sample_rate = info.sample_rate

    return frames / sample_rate

            
def _gather_audio_word_file_pairs(directory: Path, needs_words: bool=True, ignore: List[str]=['raw.wav', 'trimmed.wav']) -> List[Tuple[Path,Path]]:
    """list of `audio_filepath`, `words_filepath` pairs from a program directory.
    
    If needs_words = True, only adds audio and word filepaths if word file exists.
    """
    pairs = []
    for audio_file in filterfalse(lambda f: f.name in ignore,
                                  directory.glob('*.wav')):
        words_file = audio_file.with_name(audio_file.stem + '-alignments.json')
        if needs_words and not words_file.exists():
            print(f"{words_file.relative_to(directory).name} file does not exist.")
            continue
        
        pairs.append((audio_file, words_file))
    return pairs


def gather_word_audio_file_pairs_all_subdirs(program_directory: Path, 
                                             needs_words: bool=True, 
                                             ignore_files: List[str]=['raw.wav', 'trimmed.wav']) -> List[Tuple[Path,Path,str]]:
    """Gathers triplets of audio_filepath, words_filepath, file_identifier from a directory and its speaker segment subdirectories.
    
    If needs_words is True, requires word files to exist."""
    pairs = _gather_audio_word_file_pairs(program_directory, needs_words=needs_words, ignore=ignore_files)
    pairs = [(af, wf, af.stem) for af,wf in pairs]  # Add filename

    for subdir in filter(Path.is_dir, program_directory.glob('*')):
        equivalent_wav = subdir.with_name(subdir.name + '.wav')
        if equivalent_wav.is_file():
            speakerturn_pairs = _gather_audio_word_file_pairs(subdir, needs_words=needs_words, ignore=ignore_files)
            speakerturn_pairs = [(af, wf, f"{subdir.name}-{af.stem}") # directory is speaker's name
                                 for af,wf in speakerturn_pairs] # Add filename
            pairs.extend(speakerturn_pairs)
        else:
            pass  # Its not a subdirectory of speaker turn segments

    assert len(set(pairs)) == len(pairs), f"All audiofile names in {program_directory} must be unique"
    return pairs


def save_alignments_file(program_directory: Path, ignore_files = ['audio.wav', 'trimmed.wav']):
    """Saves alignment file from wav-word file pairs found in a directory."""

    pairs = gather_word_audio_file_pairs_all_subdirs(program_directory, ignore_files=ignore_files)

    # Create Alignment file
    with open(program_directory / "transcript.alignment.txt", "w") as f:
        for wav_path, words_path, identifier in pairs:
            # identifier = wav_path.stem
            duration = get_duration(wav_path)
            alignment = create_text_string(words_path, duration)
            alignment = alignment.replace("-alignments", '')
            oldid, _, content = alignment.partition(' ')
            alignment = ' '.join([identifier, content])
            f.write(alignment+'\n')


def valid_dataset(audio_directory):
    """Checks metadata for `clean=true/false`."""
    if not (audio_directory / "metadata.json").is_file():
        return True
    
    if Path(audio_directory / "metadata.json").stat().st_size == 0:
        return True
    
    with open(audio_directory / "metadata.json") as f:
        meta = ', '.join(json.load(f))
    pattern = r'(\w+)=(\w+)'
    res = dict(re.findall(pattern, meta))
    return res.get('clean', 'true').lower() == 'true'


@click.group()
def cli():
    pass


@cli.command()
def generate_alignments():
    """Creates alignments file mapping audio files to timestamped transcripts."""
    # Create Alignment file
    for program_dir in INTERIM_DATA_DIR.glob('*'):
        if not valid_dataset(program_dir):
            print(f"{program_dir} is not a valid dataset. skipping")
            continue
        print(f"saving alignments file for: {program_dir.name}...")
        save_alignments_file(program_dir)


@cli.command()
def move(new_dir = None, old_dir = INTERIM_DATA_DIR):
    """Moves wav files to speaker-program directory from their original program-speaker directory, for all programs.
    
    Also edits then copies alignment files to new directory
    """
    
    print("Moving encoder files for preprocessing.")
    if new_dir is None:
        training_dataname = "AllInTheMind"
        new_directory = old_dir.parent / "Real-Time-Voice-Cloning" / "datasets" / training_dataname
    else:
        new_directory = Path(new_dir)

    reorganize_files_for_encoder(old_dir, new_directory)


if __name__ == '__main__':
    cli()
