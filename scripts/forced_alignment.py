import json
import re
from dataclasses import dataclass
from itertools import filterfalse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.random.manual_seed(0)


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

@dataclass
class WordSegment(Segment):
    
    start_score: float
    end_score: float
    
    def __repr__(self):
        return f"{self.label}\t({self.start_score:4.2f}-{self.end_score:4.2f}): [{self.start:5d}, {self.end:5d})"
    

def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

# Merge words
def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                start_score = segs[0].score
                end_score = segs[-1].score
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(WordSegment(word, segments[i1].start, segments[i2 - 1].end, score, start_score, end_score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

def plot_waveform(waveform, sample_rate, ax, plt_params: dict = {}):
    from matplotlib import pyplot as plt
    ax.plot(waveform, **plt_params)

    xticks = ax.get_xticks()
    plt.xticks(xticks, xticks / sample_rate)
    ax.set_xlabel("time [second]")
    ax.set_yticks([])
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(0, waveform.size(-1))
    
    
def plot_alignments(trellis, segments, word_segments, waveform, sample_rate, save_path='alignments.png'):
    from matplotlib import pyplot as plt
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))

    ax1.imshow(trellis_with_path[1:, 1:].T, origin="lower")
    ax1.set_xticks([])
    ax1.set_yticks([])

    for word in word_segments:
        ax1.axvline(word.start - 0.5)
        ax1.axvline(word.end - 0.5)

    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start, i + 0.3))
            ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 4), fontsize=8)

    # The original waveform
    plot_waveform(waveform, sample_rate, ax2)
    ratio = waveform.size(0) / (trellis.size(0) - 1)
    for word in word_segments:
        x0 = ratio * word.start
        x1 = ratio * word.end
        ax2.axvspan(x0, x1, alpha=0.1, color="red")
        ax2.annotate(f"{word.score:.2f}", (x0, 0.8))

    for seg in segments:
        if seg.label != "|":
            ax2.annotate(seg.label, (seg.start * ratio, 0.9))
    
    plt.savefig(save_path)
    plt.show()
    
    
def valid_alignment_labels():
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    _ = bundle.get_model().to(device)
    labels = bundle.get_labels()

    dictionary = {c: i for i, c in enumerate(labels)}
    return dictionary


# TODO: test putting the alignment in a loop over files, so overhead for 
# loading the pytorch model for each file isnt huge
def force_align(SPEECH_FILE, TRANSCRIPT_FILE, verbose, save_audiofiles, plot_result):
    
    if verbose:
        print(device)
        print(torch.__version__)
        print(torchaudio.__version__)
    
    with open(TRANSCRIPT_FILE) as f:
        transcript = f.read()
    
    if len(transcript) == 0:
        raise ValueError(f"{TRANSCRIPT_FILE} has no contents")

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    sample_rate = bundle.sample_rate
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()

    dictionary = {c: i for i, c in enumerate(labels)}
    tokens = [dictionary[c] for c in transcript]

    with torch.inference_mode():
        waveform, org_sr = torchaudio.load(SPEECH_FILE)
        waveform = torchaudio.functional.resample(waveform, orig_freq = org_sr, new_freq = sample_rate)
        emissions, _ = model(waveform.to(device))
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu().detach()

    trellis = get_trellis(emission, tokens)

    path = backtrack(trellis, emission, tokens)

    segments = merge_repeats(path, transcript)

    word_segments = merge_words(segments)
    if verbose:
        for word in word_segments:
            print(word)

    if save_audiofiles:
        audio_snippets_save_dir = SPEECH_FILE.parent / "examples"
        audio_snippets_save_dir.mkdir(parents=True, exist_ok=True)
    else:
        audio_snippets_save_dir = None
        
    segment_data = prepare_output_segments(waveform, word_segments, trellis, sample_rate, audio_snippets_save_dir)
    if plot_result:
        if audio_snippets_save_dir:
            plot_save_path = audio_snippets_save_dir.parent / f"{SPEECH_FILE.stem}-alignments.png"
        else:
            plot_save_path = f"{SPEECH_FILE.stem}-alignments.png"

        plot_alignments(
        trellis,
        segments,
        word_segments,
        waveform[0],
        sample_rate,
        save_path=plot_save_path
    )
    return segment_data


def prepare_output_segments(waveform, word_segments, trellis, sample_rate, audio_snippets_save_dir):
    ratio = waveform.size(1) / (trellis.size(0) - 1)
    segment_data = []
    for i in range(len(word_segments)):
        word = word_segments[i]
        x0 = int(ratio * word.start)
        x1 = int(ratio * word.end)
        print(f"{word.label} ({word.score:.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
        segment = waveform[:, x0:x1]
        data = dict(**word.__dict__, **{'start_ts': x0 / sample_rate, 'end_ts': x1 / sample_rate, "sample_rate": sample_rate})
        segment_data.append(data)
        
        if audio_snippets_save_dir:
            audio_snippets_save_filepath = audio_snippets_save_dir / f"{i}-{word.score:.2f}-{word.label.strip()}.wav"
            output_format = "wav"
            torchaudio.save(audio_snippets_save_filepath, segment, sample_rate=sample_rate, format=output_format)
    
    return segment_data


import click


@click.group()
def cli():
    pass

@cli.command('align-single')
@click.argument('speech_file', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument('transcript_file', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--verbose', is_flag=True, help='Print verbose information.')
@click.option('--plot-filepath', '-plotfp', type=click.Path(), help='Plot the result and save to file.')
def align_single(speech_file: Path, transcript_file: Path, verbose: bool = False, plot_filepath: Optional[Path] = None):
    """Runs forced alignment and saves the word-alignments to a <*>.words.json file.
    
    transcript text must be processed to contain only tokes expected by model: `torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H`
    
    Parameters:
        speech_file (Path): The path to the speech file.
        transcript_file (Path): The path to the processed transcript file.
        verbose (bool): Whether to print verbose output
        plot_filepath: (Path): If true, plot the word alignments, and save to this path
    
    Example;
    >>> python forced-alignment.py align-single path/to/audio.wav path/to/transcript.txt -plotfp path/to/plot.png
    """
    word_alignments = force_align(speech_file, 
                                  transcript_file, 
                                  verbose, 
                                  save_audiofiles=False,
                                  plot_result=plot_filepath is not None)
    
    save_path = speech_file.with_name((str(speech_file.stem) + "-alignments.json"))
    print(f"Processed alignment. {save_path=}")
    with open(save_path, "w") as f:
        json.dump(word_alignments, f)
    

@cli.command()
@click.argument('speech_directory', type=click.Path(file_okay=False, path_type=Path))
@click.option('--verbose', is_flag=True, help='Print verbose information.')
@click.option('--save-audiofiles', is_flag=True, help='Save snippets of words as audio files.')
@click.option('--plot-result', is_flag=True, help='Plot the result.')
def align(speech_directory: Path, verbose: bool, save_audiofiles: bool, plot_result: bool):
    """
    Aligns the speech file with the corresponding transcript file, for all pairs under `speech_directory` directories.

    Example;
    ```
    python forced-alignment.py align AIM-1418
    ```
    
    Parameters:
        speech_directory (str): The directory of the speech file.
        verbose (bool): If True, verbose information will be printed.
        save_audiofiles (bool): If True, audio files will be saved.
        plot_result (bool): If True, the result will be plotted.

    Returns:
        None
    """
    
    def _get_wav_files(speech_directory: Path) -> List[Tuple]:
        
        files = []
        for speech_file in speech_directory.glob("*.wav"):
            if not re.match(r".*-\d+\.wav$", str(speech_file)):
                continue
            
            transcript_file = Path(str(speech_file.parent).replace("audio", "transcripts")) / (str(speech_file.stem) + '.txt')
            if not transcript_file.exists() or not speech_file.exists():
                print("The SPEECH or TRANSCRIPT file does not exist.")
                continue
            files.append((speech_file, transcript_file))
        
        return files
    
    def _get_segmented_wav_files(speech_directory: Path) -> List[Tuple]:
        
        files = []
        for speaker_turn_dir in filter(Path.is_dir, speech_directory.glob("*")):
            for segment_speech_file in speaker_turn_dir.glob('*.wav'):
                
                transcript_file = Path(str(segment_speech_file.parent).replace("audio", "transcripts")) / f'{segment_speech_file.stem}.txt'
                if not transcript_file.exists():
                    print(f"{transcript_file=} file does not exist.")
                if not speaker_turn_dir.exists():
                    print(f"{speaker_turn_dir=} file does not exist.")
                    
                files.append((segment_speech_file, transcript_file))
        
        return files
    
    alignments = {}
    speech_directory = Path(speech_directory)
    
    def _is_segmented(wav_file: Path):
        for file in wav_file.parent.glob('*'):
            if file.is_dir() and file.stem == wav_file.stem:
                return True
        return False
    
    segmented_files = _get_segmented_wav_files(speech_directory)
    files = list(filterfalse(lambda ff: _is_segmented(ff[0]), 
                             _get_wav_files(speech_directory)))
    click.echo(f"Aligning {len(files)} files, and {len(segmented_files)} segments of audio files")
        
    for SPEECH_FILE, TRANSCRIPT_FILE in segmented_files:
        
        print(f"\t{SPEECH_FILE=},\t{TRANSCRIPT_FILE=}")
        save_path = SPEECH_FILE.parent / (str(SPEECH_FILE.stem) + "-alignments.json")
        out = force_align(SPEECH_FILE, TRANSCRIPT_FILE, verbose=verbose, save_audiofiles=save_audiofiles, plot_result=plot_result)
        alignments[save_path] = out
        
    for SPEECH_FILE, TRANSCRIPT_FILE in filterfalse(lambda ff: _is_segmented(ff[0]), files):
        
        print(f"\t{SPEECH_FILE=},\t{TRANSCRIPT_FILE=}")
        save_path = SPEECH_FILE.parent / (str(SPEECH_FILE.stem) + "-alignments.json")
        out = force_align(SPEECH_FILE, TRANSCRIPT_FILE, verbose=verbose, save_audiofiles=save_audiofiles, plot_result=plot_result)
        alignments[save_path] = out
        

    print(f"Processed {len(alignments)} alignments")
    for save_path, out in alignments.items():
        print(f"{save_path=}")
        with open(save_path, "w") as f:
            json.dump(out, f)


if __name__ == '__main__':
    cli()
