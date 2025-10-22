import numpy as np, matplotlib.pyplot as plt, os, pandas as pd
from scipy.signal import butter, filtfilt
import intan.importrhdutilities as rhd_utils
from utils.filtering import *

### SOURCE: EHSAN + GABRIEL
def detectTransitions(analogSignal, earliestSample=0, histBinNumber=4, upTransition=False,\
                        latestSample=None, outputPlot=None, lowpassFilter=True,\
                             fs=20e3, lowPassfilterBand=500):
    """
    Input an analog signal and find the largest transitions 
    in the signal using histogram digitization

    Args:
        analogSignal (array-like): Signal to detect transitions

        earliestSample (int): First sample (in 20e3 hz rate) of signal to detect

        histBinNumber (int): How many histogram bins used to digitize signal

        upTransition (bool): whether signal transitions are up or down
        
        latestSample (int or None; default: None (entire signal duration)): last sample to detect transition (in 20e3 hz) 

        outputPlot (bool): whether to show binned signal histogram plot

        lowpassFilter (bool): whether to filter signal before transition detection

        latestSample (int or None; default: None (entire signal duration)): last sample to detect transition (in 20e3 hz) 

        fs (int; default is intan rate = 20000 hz): signal sampling rate 

        lowPassfilterBand (int): lowpass filter cutoff for signal when lowpassFilter == True

    Returns:
        triggerStart (np.ndarray)
            Indices of transition starts

        triggerStop (np.ndarray):
            Indices of end of transitions   

    """
    if not(latestSample):
        latestSample = len(analogSignal)

    if lowpassFilter:
        analogSignal = butter_lowpass_filter(analogSignal, lowPassfilterBand, fs, order=5)
    # plotting the histogram of values of analog signal
    histOutput = np.histogram(analogSignal,bins=histBinNumber)

    # identifing the two peaks on the distribution of analog values
    # we use these two values to set a decision boundry to detect the transition between two levels
    firstHistPeakEdge = np.argsort(histOutput[0])[-1]  # the position of the first peak on the histogram
    secondHistPeakEdge = np.argsort(histOutput[0])[-2] # the position of the second peak on the histogram

    # difining the cut level as the distance between the edges of the two peaks on the histogram 
    cutLevel = (histOutput[1][firstHistPeakEdge] + histOutput[1][firstHistPeakEdge + 1] \
                + histOutput[1][secondHistPeakEdge] + histOutput[1][secondHistPeakEdge + 1]) / 4 

    # defining a degitized version for the analog signal
    digitizedSignal = np.zeros(analogSignal.shape)
    # set the digitized strobe to 1 wherever the analog signal is more than the threshold value
    digitizedSignal[analogSignal>cutLevel] = 1

    # detecting the up and down transitions in the digitized signal
    upTransitionSignal = np.where(np.diff(digitizedSignal)==1)[0]
    downTransitionSignal = np.where((np.diff(digitizedSignal)==-1))[0]

    if upTransition:
        triggerStart = upTransitionSignal
        triggerStop = downTransitionSignal
    else:
        triggerStart = downTransitionSignal
        triggerStop = upTransitionSignal
        
    # just keeping those are that happen later than the earliest valid moment for the signal
    triggerStart = triggerStart[triggerStart>earliestSample]
    triggerStop = triggerStop[triggerStop>earliestSample]

    # and those that are happening before the latest desired time
    triggerStart = triggerStart[triggerStart<latestSample]
    triggerStop = triggerStop[triggerStop<latestSample]
    triggeredTransitionPlot = None

    if len(triggerStart) != len(triggerStop) or len(triggerStart) == 0:
        print(f"Warning: transition counts mismatched or empty. \
              Found {len(triggerStart)} starts but {len(triggerStop)} stops.")
        meanDur = np.nan
        triggeredTransitionPlot = 0
    else:
        meanDur = np.mean(triggerStop - triggerStart) / fs
        triggeredTransitionPlot = 1

    if outputPlot:
        plt.figure()
        plt.hist(analogSignal,bins=histBinNumber)
        plt.title('histogram of the analog values')
        # 
        if triggeredTransitionPlot:
            plt.figure()
            plt.title('all transitions triggered by detected transition time')
            transWindowToLook = int(1.25*meanDur*1e3)
            plt.xlabel('ms')
            for transitionTime in triggerStart[:]: 
                plt.plot(np.arange(-transWindowToLook,transWindowToLook,1e3/fs),\
                        analogSignal[int(transitionTime-transWindowToLook*fs/1e3):\
                                int(transitionTime+transWindowToLook*fs/1e3)],'gray')       
            plt.figure()
            plt.title('5 sample transitions zoomed-in')
            transWindowToLook = 25
            plt.xlabel('ms')
            for transitionTime in triggerStart[:5]:
                plt.plot(np.arange(-transWindowToLook,transWindowToLook,1e3/fs),\
                        analogSignal[int(transitionTime-transWindowToLook*fs/1e3):\
                                int(transitionTime+transWindowToLook*fs/1e3)],'gray')
            plt.axvline(0)
            plt.show()
    return triggerStart, triggerStop

def get_analog_times(analog_signal, fs=20000, lowpassFilter=True, lowPassfilterBand=500.0, histBins=4,
                 start_sample=0, last_sample=None, upTransition=False, outputPlot=False):
    """
    Wrapper for detectTransitions function to return start and end transition TIMES instead of signal indices

    Args:
        analogSignal (array-like): Signal to detect transitions

        fs (int; default is intan rate = 20000 hz): signal sampling rate 

        lowpassFilter (bool): whether to filter signal before transition detection

        lowPassfilterBand (int): lowpass filter cutoff for signal when lowpassFilter == True
        
        start_sample (int): First sample index (in 20e3 hz rate) of signal to detect

        last_sample (int or None; default: None (entire signal duration)): last sample to detect transition (in 20e3 hz) 

        histBins (int): How many histogram bins used to digitize signa
        
        upTransition (bool): whether signal transitions are up or down

        outputPlot (bool): whether to show binned signal histogram plot

    Returns:
        start_times (np.ndarray): signal transition start times
        end_times (np.ndarray): signal transition end times
    """
    # We want the rising edges of the scope TTL
    analog_signal = np.asarray(analog_signal)
    if last_sample is None:
        last_sample = len(analog_signal) - 1
    print(f'Getting transitions from {0}:{last_sample / fs} s')
    starts, ends = detectTransitions(analog_signal, earliestSample=start_sample, latestSample=last_sample, histBinNumber=histBins,
        upTransition=upTransition, lowpassFilter=lowpassFilter, fs=fs, lowPassfilterBand=lowPassfilterBand, outputPlot=outputPlot)
    start_times = starts / fs
    end_times = ends / fs
    print(f'Found {len(start_times)} raw triggers')   
    return start_times, end_times


def align_scope_triggers_to_frames(s2p_output, scope_times):
    """
    ERROR CHECKING FUNCTION: ENSURE NUMBER 2P FRAMES AND NUM TRIGGERS ARE SAME LENGTH AND CORRECT

    Args:
        s2p_output (Suite2POutput object from getSuite2POutput.py): s2p object output 

        scope_times (array-like): current detected times of scope pulses 

    Returns:
        scope_times, scope_end_times (array-like, array-like): adjusted start and end times of 2P scope

    """
    if len(scope_times) != s2p_output.nframes:
        ### FIX ERROR WHERE MORE FRAMES THAN TRIGGERS (INTAN TURNED OFF EARLY): 
        # add -1 to end of scope times where no frame found
        if len(scope_times) < s2p_output.nframes:
            frame_num_diff = s2p_output.nframes - len(scope_times)
            print(f'WARNING!!! Found {len(scope_times)} triggers but {s2p_output.nframes} frames. \
                    Adding {frame_num_diff} from experiment to match trigger counts...')
                        # add fake triggers to end when scope still on and experiment end 
            extra_frame_times = [-1] *  frame_num_diff
            print(f'Adding {frame_num_diff} extra scope trigger times for frames')
            scope_times = np.append(scope_times, extra_frame_times)
            scope_times_end = np.append(scope_times_end, extra_frame_times)

        ## FIX ERROR WHERE MORE TRIGGERS THAN FRAMES (SCOPE TURNED OFF EARLY)
        # truncate trigger times to num frames 
        elif len(scope_times) > s2p_output.nframes:
            scope_times = scope_times[:s2p_output.nframes]
            scope_times_end = scope_times[:s2p_output.nframes]

    # CHECK TRIGGER START AND STOP LENS
    # make sure scope start and end triggers are same length, if not add fake transitions to 
    # to shorter transition times list 
    scope_triggers_diff = len(scope_times_end) - len(scope_times)
    if scope_triggers_diff != 0:
        print(f'WARNING!! Found a difference of {scope_triggers_diff} for stop - start scope triggers.')
        transitions_to_append = [-1] * abs(scope_triggers_diff)
        if scope_triggers_diff > 0:
            print(f'Appending {len(transitions_to_append)} empty transitions to start scope times...')
            scope_times = np.append(scope_times, transitions_to_append)
        elif scope_triggers_diff < 0:
            print(f'Appending {len(transitions_to_append)} empty transitions to end scope times...')
            scope_times_end = np.append(scope_times_end, transitions_to_append)

    return scope_times, scope_times_end
    
def resample_traces(traces, frame_times, len_cam_times):
    """
    Resample 2P traces to match sampling rate of other data/signals 

    Args:
        traces (2D np.ndarray of shape (num_cells, num_frames)): Calcium traces to resample
        frame_times (array-like): frame times of scope
        len_cam_times (int): number of samples to stretch traces to

    Returns:
        traces_resampled (np.ndarray): resampled calcium traces matching fps of time samples 
    
    """
    from scipy.signal import resample
    # if 1d array reshape to create 2d for resampling
    if traces.ndim == 1:
        traces = traces.reshape(1, -1)
    resampled_traces = resample(traces, len_cam_times, t=frame_times, axis=1)
    traces_resampled, resampled_timestamps = resampled_traces
    return traces_resampled

def debug_triggers(signal, earliestSample=0, signal_fs=20e3):
  """
  Debugging function to check trigger durations and determine if they align with expected timings of signal
  Output digitized histogram plot to see transition amplitude bins

  Args:
    signal (array-like): signal to detect transitions bug 

    earliestSample (int): first index of sample to detect transitions

    signal_fs (int; default intan fs = 20e3): sampling rate of signal

  Returns:
    triggers_df (pd.DataFrame): dataframe containing transition starts, ends, durations, and times between triggers
    use to examine output of transitions function
  
  """
  starts, stops = detectTransitions(signal, earliestSample=earliestSample, outputPlot=True)

  start_times, stop_times = starts/signal_fs,stops/signal_fs

  triggers_df = pd.DataFrame({
  'start':start_times, 'end':stop_times, 'trigger dur': stop_times - start_times,
  'time_since_last_trigger': np.diff(start_times, append=True)})
  
  return triggers_df


# def match_column_by_nearest_time(
#     camera_times,
#     frame_df,
#     time_col: str,
#     value_col: str,
#     method: str = "nearest",      # "nearest" | "backward" | "forward"
#     tolerance=None,               # None, float seconds, or pd.Timedelta
#     dedup: str = "last",          # how to resolve duplicate times in frame_df: "last"|"first"|None
#     return_indices: bool = False  # also return the chosen row indices into frame_df (after sort/dedup)
# ):
#     """
#     Map each camera time to a value from frame_df[value_col] chosen by temporal proximity.

#     Parameters
#     ----------
#     camera_times : array-like of timestamps (float seconds or datetime-like)
#     frame_df     : DataFrame with at least [time_col, value_col]
#     time_col     : name of the time column in frame_df
#     value_col    : name of the value column to sample from frame_df
#     method       : "nearest" (default), "backward" (last <= t), or "forward" (first >= t)
#     tolerance    : optional max time gap allowed (float seconds or pd.Timedelta).
#                    If provided and the nearest/selected sample is farther than tolerance,
#                    the output will be set to None for that camera time.
#     dedup        : drop duplicate frame times keeping the "last" or "first" before matching.
#     return_indices : if True, also return the integer indices (into the sorted/deduped view).

#     Returns
#     -------
#     values : np.ndarray of length len(camera_times)
#     (indices) : optional np.ndarray of chosen indices into the sorted/deduped frame_df view
#     """
#     if time_col not in frame_df or value_col not in frame_df:
#         raise KeyError("time_col or value_col not found in frame_df")

#     # Prepare source times/values (sorted, optional de-dup)
#     src = frame_df[[time_col, value_col]].copy()
#     src = src.sort_values(time_col, kind="mergesort")  # stable sort
#     if dedup in ("first", "last"):
#         src = src.drop_duplicates(subset=time_col, keep=dedup)

#     # Convert times to a common numeric axis (seconds) while supporting datetimes
#     def _to_seconds(a):
#         s = pd.Series(a)
#         if pd.api.types.is_datetime64_any_dtype(s):
#             return pd.to_datetime(s).view("int64") / 1e9  # ns -> s
#         return s.astype(float).to_numpy()

#     t_src = _to_seconds(src[time_col].to_numpy())
#     v_src = src[value_col].to_numpy()
#     t_cam = _to_seconds(camera_times)

#     if len(t_src) == 0:
#         raise ValueError("frame_df has no rows after optional dedup.")
#     n = len(t_src)

#     # Optional tolerance (seconds)
#     if tolerance is None:
#         tol_sec = None
#     elif isinstance(tolerance, pd.Timedelta):
#         tol_sec = tolerance.total_seconds()
#     else:
#         tol_sec = float(tolerance)

#     # Vectorized index selection
#     idx = np.searchsorted(t_src, t_cam, side="left")

#     if method == "backward":
#         pick = np.clip(idx - 1, 0, n - 1)
#     elif method == "forward":
#         pick = np.clip(idx, 0, n - 1)
#     elif method == "nearest":
#         left = np.clip(idx - 1, 0, n - 1)
#         right = np.clip(idx, 0, n - 1)
#         # choose nearer; on ties, prefer left (earlier)
#         choose_right = np.abs(t_src[right] - t_cam) < np.abs(t_src[left] - t_cam)
#         pick = left.copy()
#         pick[choose_right] = right[choose_right]
#     else:
#         raise ValueError("method must be 'nearest', 'backward', or 'forward'")

#     # Apply tolerance if requested
#     if tol_sec is not None:
#         dt = np.abs(t_src[pick] - t_cam)
#         invalid = dt > tol_sec
#     else:
#         invalid = np.zeros_like(pick, dtype=bool)

#     # Build output (preserve dtype if possible; fall back to object when mixing None)
#     out = v_src[pick].copy()
#     if invalid.any():
#         # ensure we can place None without error
#         if out.dtype.kind in "fiu":  # numeric -> promote to object to hold None
#             out = out.astype(object)
#         out[invalid] = None

#     if return_indices:
#         return out, pick
#     return out