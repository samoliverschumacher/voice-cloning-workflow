from pathlib import Path

import click
from pydub import AudioSegment
from pydub.silence import split_on_silence


@click.command()
@click.argument('input_audio_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output_directory', type=click.Path(file_okay=False))
@click.option('--silence_threshold', type=int, default=-30,
              help='Threshold (in dB) for considering a segment as silence.')
@click.option('--max_segments', type=int, default=None,
              help='Maximum number of segments to create.')
def split_audio_on_silences(input_audio_file, output_directory, silence_threshold, max_segments):
    """
    Split an audio file into segments based on silences.

    Args:
        input_audio_file (str): Path to the input audio file.
        output_directory (str, optional): Path to the directory where the output segments will be saved. If not provided, segments will be saved in a directory with the same name as the input audio file.
        silence_threshold (int, optional): Threshold (in dB) for considering a segment as silence. Defaults to -30.
        max_segments (int, optional): Maximum number of segments to create. If provided, the function will downsample the number of segments to this value. Defaults to None.

    Returns:
        None
    """
    
    audio = AudioSegment.from_file(input_audio_file)
    click.echo(f"{audio.dBFS=} (avg intensity of file)")
    click.echo(f"{silence_threshold=}")

    # Split audio on silences
    segments = split_on_silence(audio, min_silence_len=300, silence_thresh=silence_threshold)
    
    if max_segments and max_segments < len(segments):
        step = len(segments) // max_segments
        segments = [sum(segments[i:i+step]) for i in range(0, len(segments), step)]

    if not output_directory:
        file = Path(input_audio_file).stem
        output_directory = Path(input_audio_file).parent / file
    output_directory.mkdir(exist_ok=True)

    for i, segment in enumerate(segments):
        output_file = Path(output_directory) / f'segment_{i}.wav'
        segment.export(output_file, format="wav")
        length = len(segment) / 1000  # Length in seconds
        click.echo(f'Saved segment {i} to {output_file.relative_to(input_audio_file.parent.parent)}, Length: {length:.2f} seconds')

if __name__ == "__main__":
    split_audio_on_silences()
