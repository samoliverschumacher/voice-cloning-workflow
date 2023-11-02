import inspect
from pathlib import Path
from typing import List, Tuple

import soundfile as sf
from process_transcript import word_break_sym
from structures import Annotation, Diarization


def split_waveform(word_segments: Diarization, min_duration, max_duration, score_threshold=0.94) -> List[Annotation]:
    """
    Splits a waveform into utterances based on word segments.
    
    Keeps subarrays that are not within the min & max duration thresholds, that are at the end of the audio file.
    
    Args:
        word_segments (Diarization): A list of word segments with their start and stop times.
        min_duration (float): The minimum duration of an utterance in seconds.
        max_duration (float): The maximum duration of an utterance in seconds.
        score_threshold (float, optional): The score threshold for a word segment to be considered valid. Defaults to 0.94.
    
    Returns:
        List[Annotation]: A list of utterances, each represented as an Annotation object with start, stop, and content attributes.
    """
    
    gap_ts = 0.005  # time increment added between end of one segment, and start of another
    subarrays = []
    start_ts = 0
    end_ts = 0
    i_s = 0
    # if first word is well identified, assume the time prior to start of word can be cut
    time_zero = word_segments[0].start if word_segments[0]['score'] > score_threshold else 0
    
    for i, segment in enumerate(word_segments):
        if segment.stop > end_ts and segment['score'] > score_threshold:
            end_ts = segment.stop
        
        if max_duration >= (end_ts - start_ts - time_zero) >= min_duration:

            a = Annotation(start = start_ts, 
                           stop = end_ts, 
                           content = {'indices': (i_s, i), 
                                      'score': segment['score'], 
                                      'labels': [e['label'] for e in word_segments[i_s: i+1]]})
            subarrays.append(a)
            
            start_ts = end_ts + gap_ts
            i_s = i+1
        elif (end_ts - start_ts - time_zero) > max_duration:  # duration is longer than allowed.
            word_scores = ' '.join([f"{e['label']}({e['score']:.2f})" for e in word_segments[i_s: i+1]])
            duration = end_ts - start_ts
            print(f"Utterance too long ({duration:.2f}s > {max_duration}s). Splitting anyway... {segment['score']=}\n{word_scores}")

            a = Annotation(start = start_ts, 
                           stop = end_ts, 
                           content = {'indices': (i_s, i), 
                                      'score': segment['score'], 
                                      'labels': [e['label'] for e in word_segments[i_s: i+1]]})
            subarrays.append(a)
            
            start_ts = end_ts + gap_ts
            i_s = i+1
        else:  # duration is shorter than allowed.
            pass
        
        if (end_ts - start_ts - time_zero) > max_duration:
            start_ts = end_ts + gap_ts
    
    # If score threshold was not met, but entire file is under the max duration threshold, add utterance anyway
    if len(subarrays) == 0:
        first_segment = word_segments[0]
        last_segment = word_segments[-1]
        if min_duration < last_segment.stop < max_duration:
            print(f"Utterance ends on low score ({last_segment['score']:.2f}s < {score_threshold}). Adding as utterance anyway... ")
        elif last_segment.stop < min_duration:  # last and only segment is too short
            word_scores = ' '.join([f"{e['label']}({e['score']:.2f})" for e in word_segments])
            duration = last_segment.stop - first_segment.start
            print(f"Utterance too short ({duration:.2f}s > {max_duration}s). Splitting anyway... {last_segment['score']=}\n{word_scores}")            
        else:  # last and only segment is too long
            word_scores = ' '.join([f"{e['label']}({e['score']:.2f})" for e in word_segments])
            duration = last_segment.stop - first_segment.start
            print(f"Utterance too long ({duration:.2f}s > {max_duration}s). Splitting anyway... {last_segment['score']=}\n{word_scores}")

        a = Annotation(start = 0, 
                       stop = last_segment.stop, 
                       content = {'indices': (0, len(word_segments)-1), 
                                  'score': last_segment['score'], 
                                  'labels': [e['label'] for e in word_segments]})
        subarrays.append(a)
        return subarrays
        
    # Check if the last segment doesn't have a score high enough
    print(subarrays[-1])
    if subarrays[-1].content['indices'][1] < (len(word_segments)-1):
        # Use 'None' as the end value if segment isnt long enough / didnt score highly enough
        subarrays.append(None)
        
    return subarrays


def process(word_alignments_path: Path, 
            audio_path: Path, 
            min_duration: float, 
            max_duration: float, 
            score_threshold: float, 
            verbose: bool) -> List[Tuple[Path,Path,Path]]:
    """Split audiofile into shorter utterances using timestamps in transcription word alignment file.
    
    Splits adhere to min and max duration limits, and diarization score threshold.
    """
    word_segments = Diarization(word_alignments_path)
    # Create a directory for the utterances
    utterance_dir = audio_path.parent / (audio_path.stem +"-utterance")
    utterance_dir.mkdir(exist_ok=True)

    # Load the audio waveform array from the audio filepath
    waveform, sample_rate = sf.read(audio_path)
    audio_duration = waveform.shape[0] / sample_rate

    if verbose:
        print(f"Splitting {word_alignments_path}")
        
    subarrays = split_waveform(word_segments, min_duration, max_duration, score_threshold=score_threshold)
    if len(subarrays) == 0 or subarrays[0] == None:
        raise ValueError(f"No word segments found for {audio_path}, with {min_duration=} & {max_duration=}")

    last_duration = audio_duration - subarrays[-2].stop
    last_words = word_segments[subarrays[-2]['indices'][1]+1 : ]
    if subarrays[-1] is None and (last_duration > min_duration):
        a = Annotation(start = subarrays[-2].stop,
                        stop = audio_duration,
                        content = {'indices': (subarrays[-2]['indices'][1]+1, len(word_segments)), 
                                   'score': last_words[0]['score'], 
                                   'labels': [e['label'] for e in last_words]})
        subarrays[-1] = a
    elif subarrays[-1] is None:  # last few words are removed, and utterance is too short
        subarrays.pop(-1)
        print(f"Last few words are removed from {word_alignments_path.name}. Last words: {[e['label'] for e in last_words]}. Duration: {last_duration}")
    
    # Split the waveform array based on the subarrays
    split_waveforms = [waveform[int(sample_rate * e.start):int(sample_rate * e.stop)] for e in subarrays]

    # Print and save the split waveforms
    created_files = []
    for i, waveform in enumerate(split_waveforms):
        # Filenames designated with utterance number
        wav_filepath = utterance_dir / (audio_path.name.replace('.wav', f'-utterance{i}.wav'))
        words_filepath = utterance_dir / (word_alignments_path.name.replace('-alignments.json', f'-utterance{i}-alignments.json'))
        # Create new single-speaker-turn-utterance transcript file
        transcript_filepath = utterance_dir / (audio_path.stem + f'-utterance{i}.processed.txt')
        
        utterance_word_segments: List[Annotation] = word_segments[subarrays[i]['indices'][0]: 1+subarrays[i]['indices'][1]]
        # timestamps are reset to be reletaive to the shortened wav file
        time_zero = utterance_word_segments[0].start
        for word in utterance_word_segments:
            word.start -= time_zero
            word.stop -= time_zero
        
        if verbose:
            print(f"Split Waveform {i+1}: Last score: {subarrays[i]['score']}" 
                  + ', '.join([f"{e['label']}({e['score']:.3f})" for e in utterance_word_segments]))

        utterance_segments = Diarization.from_annotations(utterance_word_segments)
        
        if verbose:
            print(f"saved: {words_filepath=}")
        utterance_segments.save(to=words_filepath)
            
        if verbose:
            print(f"saved: {wav_filepath=}")
        sf.write(wav_filepath, waveform, sample_rate)
            
        
        sorted_words = sorted(list(utterance_segments), key=lambda e: e.start)
        transcript = word_break_sym.join([e.content['label'] for e in sorted_words])
        with open(transcript_filepath, 'w') as file:
            file.write(transcript)
        if verbose:
            print(f"saved: {transcript_filepath=}")
            
        created_files.append((wav_filepath, words_filepath, transcript_filepath))
        
    return created_files


def transcribe_utterance(words_filepath: Path) -> None:
    """Transcribe a single utterance. Saves  "|"-delimited `<audiofile-name>.txt` into  equivalent directory in `transcripts` folder"""
    directory = Path(str(words_filepath).replace('audio', 'transcripts')).parent
    filename = words_filepath.name.replace('-alignments.json', '.txt')
    utterance_transcript_filepath = directory / filename
    # Check if the transcripts already there
    if utterance_transcript_filepath.exists():
        with open(utterance_transcript_filepath, 'rt') as f:
            transcript = f.read()
        if len(transcript) > 0:
            return  # File already exists, dont overwrite
    
    word_segments = Diarization(words_filepath)
    transcript = word_break_sym.join([e['label'] for e in word_segments])
    
    with open(utterance_transcript_filepath, 'wt') as file:
        file.write(transcript)
