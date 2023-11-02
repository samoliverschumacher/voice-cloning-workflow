
import json
from pathlib import Path

import click
import requests
from bs4 import BeautifulSoup

from process_transcript import speaker_delimiter


@click.command()
@click.argument('out_dir', type=click.STRING, metavar='FILEs_NAME')#, help='Name of the output file')
@click.argument('program_extension', type=click.STRING, metavar='PROGRAM_EXTENSION')#, help='Program extension for the URL')
@click.argument('group_name', type=click.STRING, metavar='GROUP_NAME')
@click.option('--verbose', ' /-v', is_flag=True, show_default=True, default=True, help='verbose output')
def main(out_dir, program_extension, group_name, verbose):  # used
    """
    Parses program data from the website and saves it to a CSV file and a JSON file.
    
    Example;
    python main.py "AIM-1418" "season-1/episode-12" "2018"
    """
    URL = f"https://www.media.com/listen/programs/allinthemind/{program_extension}"
    print(f"{URL=}")
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="transcript")
    if results is None:
        raise ValueError("Couldnt find {URL = }")

    save_folder = Path(out_dir)
    save_filepath = save_folder / "transcript.txt"
    metadata_filepath = save_folder / "metadata.json"
    
    print(f"Saving to {save_filepath=}")

    with open(save_filepath, 'wt') as f:
        name = None
        paragraph_counter = 0
        speaker_turn_counter = 0
        for par in results.find_all('p'):
            paragraph_counter += 1
            speaker_turn_counter += 1
            
            words = par.text.split(' ')
            
            # Check if this paragraph is the start of a new speaker (this paragraph is structured <speaker name> : < content>)
            if len(words) > 2:
                fname, sname, thirdname = par.text.split(' ')[:3]
                has_speaker_delim = any([':' in n for n in [fname, sname, thirdname]])
            else:
                has_speaker_delim = False

            assert name or has_speaker_delim, ("Last paragraph didn't have a name, neither does this one. "
                                               "Expected the paragraph to start with a speaker name "
                                               "containing <= 3 words, and ending with colon (:)\n"
                                               f"First 6 words: {par.text.split(' ')[:6]}")

            # This paragraph doesnt start with <speaker>:
            if not has_speaker_delim:
                if par.text:
                    f.write(par.text)
                continue

            # add newline between paragraphs
            if name is not None:
                f.write('\n')
                
            name, _, content = par.text.partition(':')
            if name.startswith('Published'):
                break
            
            # Checking for unprintable characters
            from string import printable
            weird_text = ''.join(char for char in content if char not in printable)
            if weird_text:
                if verbose: print(f" Found weird text \t\t\t {repr(weird_text)}")
                content = ''.join(char for char in content if char in printable)
                
            weird_text = ''.join(char for char in name if char not in printable)
            if weird_text:
                name = ''.join(char for char in name if char in printable)
                if verbose: print(f" Found weird text \t\t\t {repr(weird_text)}")
                
            row = speaker_delimiter.join( [name, content] )

            f.write(row)
            if verbose: print(row)
        if verbose: print(f"Finished crawl with: {paragraph_counter=}, {speaker_turn_counter=}")
        
    # Add details of file provenance
    metadata = []
    if metadata_filepath.exists():
        with open(metadata_filepath, 'rt') as f:
            metadata = json.load(f)

    with open(metadata_filepath, 'wt') as f:
        metadata.append({'url': URL, "filename": out_dir, "group": group_name})
        json.dump(metadata, f)


if __name__ == "__main__":
    main()
