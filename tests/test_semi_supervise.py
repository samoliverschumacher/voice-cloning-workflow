import numpy as np

import sys
import os
from pathlib import Path

scripts_dir = Path(__file__).parent.parent / "scripts"
print(scripts_dir)
sys.path.append(str(scripts_dir))

from teacher_forced_diarize import overwrite_segmentation_scores

# TEST 1 - non-overlapping window
def test_one():
    seconds_per_window = 5.0
    frames_per_second = 293 / 5.0
    seconds_per_step = 0.5
    frames_per_step = frames_per_second * seconds_per_step
    frames_per_window = int(seconds_per_window * frames_per_second)

    segs = np.random.random((51, 293, 3))
    input_shape = segs.copy().shape

    semi_sup = [[(1.78, 2.00), (3.01, 4.05)],  # speaker 1
                [(2.1, 2.9), (3.89, 4.00)]]  # speaker 2

    segs = overwrite_segmentation_scores(segs, semi_sup, seconds_per_window, frames_per_window, seconds_per_step)
    
    np.testing.assert_equal(segs.shape, input_shape)


# TEST 2 - overlapping window
def test_two():
    seconds_per_window = 2
    seconds_per_step = 0.5
    frames_per_second = 2
    # frames_per_step = frames_per_second * seconds_per_step
    frames_per_window = seconds_per_window * frames_per_second

    segs = np.array([[ [0.99, 0, 0],
                       [0.87, 0, 0],
                       [0, 0.99, 0],
                       [0.74, 0.21, 0] ]])  # type: ignore

    semi_sup = [[(0.1, 0.24), (1.76, 2.05)], 
                [(0.75,1)]]


    segs = overwrite_segmentation_scores(segs, 
                                        semi_sup, 
                                        seconds_per_window, 
                                        frames_per_window, 
                                        seconds_per_step)

    expected_segs = np.array([[ [1.00, 0, 0],
                                [0, 1.00, 0],
                                [0, 0.99, 0],
                                [1.00, 0, 0] ]])  # type: ignore

    # frame_mask = np.isnan(segs)
    np.testing.assert_array_equal(expected_segs, segs)

# TEST 3 - chunks only contain window they pertain to
def test_three():
    seconds_per_window = 1
    seconds_per_step = 0.5
    frames_per_second = 2
    frames_per_step = frames_per_second * seconds_per_step
    frames_per_window = seconds_per_window * frames_per_second

    segs = np.array([[ [0.99, 0, 0],
                       [0.87, 0, 0] ],
                    
                    [ [0.87, 0, 0],
                      [0, 0.99, 0] ],
                    
                    [ [0, 0.99, 0],
                      [0.74, 0.21, 0] ]])  # type: ignore

    semi_sup = [[(1.78, 2.00), (3.01, 4.05)], [(2.1, 2.9), (3.89, 4.00)]]
    semi_sup = [[(1.78/2, 2.00/2), (3.01/2, 4.05/2)], [(2.1/2, 2.9/2), (3.89/2, 4.00/2)]]


    segs = overwrite_segmentation_scores(segs, 
                                        semi_sup, 
                                        seconds_per_window, 
                                        frames_per_window, 
                                        seconds_per_step)

    expected_segs = np.array([[ [0.99, 0, 0],
                                [1., 0, 0] ],
                            
                            [ [1., 0, 0],
                                [0, 1., 0] ],
                            
                            [ [0, 1., 0],
                                [1., 1., 0] ]])  # type: ignore

    frame_mask = np.isnan(segs)
    np.testing.assert_array_equal(segs[~frame_mask], expected_segs[~frame_mask])

test_one()
test_two()
test_three()
