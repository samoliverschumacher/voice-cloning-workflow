
from math import ceil, floor
from typing import List, Tuple

import numpy as np

SpeakerActivity = List[ List[Tuple[float, float]] ]


"""
    This script is to be placed in the same folder as `pyannote-audio/pyannote/audio/pipelines`.
    Its used by `pyannote.audio.pipelines.speaker_diarization` to overwrite the segmentation scores in `SpeakerDiarization.apply`
"""

def overwrite_segmentation_scores(segmentation_scores: np.ndarray, 
                                  true_speaker_activity: SpeakerActivity, 
                                  window_duration: float, 
                                  frames_per_window: int, 
                                  step_duration: float):
    """
    Overwrites the segmentation scores with the true speaker activity.
    
    This function is 

    Args:
        segmentation_scores (np.ndarray): The segmentation scores to be overwritten.
        true_speaker_activity (SpeakerActivity): The true speaker activity.
        window_duration (float): The duration of the window.
        frames_per_window (int): The number of frames per window.
        step_duration (float): The duration of the step.

    Returns:
        np.ndarray: The overwritten segmentation scores.
    """
    
    
    frames_per_second: int = floor(frames_per_window / window_duration)
    frames_per_step: int = floor(frames_per_second * step_duration)
    
    n_chunks, n_frames, n_speakers = segmentation_scores.shape
    n_frames_total = n_chunks * frames_per_step + (frames_per_window - frames_per_step)

    ss = np.zeros((n_frames_total, n_speakers))
    for speaker_index, speaker_turn in enumerate(true_speaker_activity):
        for start_seconds, stop_seconds in speaker_turn:
            global_start_frame = floor(start_seconds * frames_per_second)
            global_stop_frame = ceil(stop_seconds * frames_per_second)
            
            ss[global_start_frame: global_stop_frame, speaker_index] = 1

    # Create a 1D array of indices for active frames
    id_c = np.arange(n_chunks)
    active_frames_indices = id_c[:, None] * frames_per_step + np.arange(frames_per_window)
    
    # Filter active frames and apply broadcasting to get active_frame array
    active_frame = ss[active_frames_indices]
    
    # Create a boolean mask for frames with any speaker active
    speaker_frame_filter = np.any(active_frame == 1, axis=2)
    
    segmentation_scores[speaker_frame_filter] = active_frame[speaker_frame_filter]
    return segmentation_scores

if __name__=='__main__':
    exit()
