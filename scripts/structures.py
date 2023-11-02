import json
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

"""
    Convenience data models for manipulating transcripts and diarizations.
"""

SpeakerTurn = namedtuple('SpeakerTurn', 'number speaker transcript')
        
@dataclass
class Transcript:
    """Pairs of speaker name, and the words spoken.
    
    >>> list(Transcript)
    [SpeakerTurn(number=0, speaker="name1", transcript="content1",
     SpeakerTurn(number=1, speaker="name2", transcript="content2")])]
    """
    fpath: str
    _sep: Optional[str] = '|'  # If None, line(s) are words only, no speaker name
    
    def save(self, to: Union[str, Path, None]):
        if to is None:
            if self.fpath is None:
                raise ValueError("self.fpath is None. Set key-word arg `to`")
            outpath = self.fpath
        else:
            outpath = to

        with open(outpath, 'w') as f:
            f.write(self.to_string() + '\n' )
        
    def to_string(self):
        return '\n'.join([f"{row.speaker}{self._sep} {row.transcript}" 
                          for row in self.speaker_turns])

    def _load(self):
        with open(self.fpath) as f:
            self._content_rows = f.readlines()
        
        self.speaker_turns: List[SpeakerTurn] = []
        for row_index, row in enumerate(self._content_rows):
            turn_number = row_index + 1
            
            if self._sep:
                speaker, _, transcript = row.partition(self._sep)
            else:  # The line only contains transcript words.
                transcript = row
                speaker = None

            self.speaker_turns.append(SpeakerTurn(turn_number, speaker, transcript.strip('\n')))
            
    def __post_init__(self):
        self._load()
        
    def __len__(self):
        return len(self.speaker_turns)
    
    def __getitem__(self, index):
        return self.speaker_turns[index]
    
    def __iter__(self):
        return iter(self.speaker_turns)
    
    @property
    def speaker_order(self):
        return list(OrderedDict.fromkeys([row.speaker for row in self]))


@dataclass
class Annotation:
    """
    Speaker diary:
    ```csv
    start: 1.0, stop: 1.4, speaker: name
    ```
    >>> Annotation(start=1.0, stop=1.4, content={"speaker": "name"})
    
    Words:
    ```json
    {"start": 0.0, "stop": 0.5, "label": "Hello.", "score": 0.98}
    ```
    >>> Annotation(start=0.0, stop=0.5, content={"label": "Hello", "score": 0.98})
    """
    start: float
    stop: float
    content: dict
    
    def __getitem__(self, index):
        return self.content[index]
    

@dataclass
class Diarization:
    """A list of timestamps for the start & stop of something (in an audio file).
    
    File type 1: Speaker diarization .csv
    ```csv
    start: 6.20, stop: 22.29, speaker: Speaker A
    start: 23.55, stop: 36.45, speaker: Speaker B
    ```
    >>> [Annotation(start=1.0, stop=1.4, content={"speaker":"name"}), ...]
    
    File type 2: Word alignments .json
    ```json
    [
        {"start": 0.0, "stop": 0.5, "label": "Hello.", "score": 0.98},
        {"start": 0.9, "stop": 1.4, "label": "World.", "score": 0.87}
    ]
    ```
    >>> [Annotation(start=1.0, stop=1.4, content={"label":"Hello", "score": 0.98}), ]
    """
    
    fpath: Union[str, Path, None]
    
    def __post_init__(self):
        self.annotations: List[Annotation] = []
        if self.fpath is not None:
            self.fpath = Path(self.fpath)
            self._load()
    
    @staticmethod
    def from_annotations(annotations: List[Annotation]) -> "Diarization":
        diarization = Diarization(None)
        for a in annotations:
            diarization.append(a)
        return diarization
        
    def append(self, annotation: Annotation):
        self.annotations.append(annotation)
        
    def save(self, to: Union[str, Path, None]):
        if to is None:
            if self.fpath is None:
                raise ValueError("self.fpath is None. Set key-word arg `to`")
            outpath = self.fpath
        else:
            outpath = to

        self._type = Path(outpath).suffix
        with open(outpath, 'w') as f:
            f.write(self.to_string() + '\n' )
        
    def to_string(self):
        if self._type == '.json':
            return json.dumps([{'start_ts': item.start, 'end_ts': item.stop, **item.content} 
                                for item in self.annotations])
        
        # Is a diarization .csv file of 3 columns.
        return '\n'.join([f"start: {e.start}, stop: {e.stop}, speaker: {e.content['speaker']}" 
                            for e in self.annotations])
        
    def _load(self):
        
        if self.fpath.suffix == '.json':
            self._type = 'json'
            with open(self.fpath) as f:
                self._content_rows = json.load(f)
                
            for item in self._content_rows:
                try:
                    a = Annotation(start=item['start_ts'], 
                                stop=item['end_ts'], 
                                content={k:v for k,v in item.items() if k not in ['start_ts','end_ts']})                
                except KeyError as e:
                    print("'start_ts' and 'end_ts' must be in each item of the json array")
                    raise e
                self.annotations.append(a)
        else:
            self._type = 'csv'
            with open(self.fpath) as f:
                self._content_rows = [l.strip('\n') for l in f.readlines()]
            for row in self._content_rows:
                start, stop, speaker = row.split(',')
                start = float(start.split('start: ')[-1])
                stop = float(stop.split('stop: ')[-1])
                speaker = speaker.split('speaker: ')[-1]
                a = Annotation(start=start, 
                               stop=stop, 
                               content={"speaker": speaker})
                self.annotations.append(a)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        return self.annotations[index]
    
    def __iter__(self):
        return iter(self.annotations)
    

def to_frame(diarization: Diarization):
    return pd.DataFrame([{'start': e.start, 'stop': e.stop, **e.content} for e in diarization])

    
@dataclass
class FinalisedItem:
    """The final data item ready for training.
    
     - Single speaker.
     - path to audio file: length between 1.6 and 11 seconds.
     - Transcript.
     - Words aligned to the audio (timestamped).
    """
    speaker: str
    audio: str
    transcript: Transcript
    annotations: List[Annotation]
