from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import traceback

def get_state_df(
    facemap_data, 
    camera_times, 
    treadmill_signal=None, 
    treadmill_data=True,
    cam_fps=30, 
    smoothing_kernel=5, 
    movement_percentile=70, 
    min_dur_s=3,
    to_parquet=False,
    parquet_output_name = None,
    to_csv = False,
    csv_output_name = None
):
    """
    Function to get an easy to use dataframe that aligns timing information with state:
    Saves raw face motion, pupil area, and locomotion (treadmill) information
    Boolean columns (containing 0 or 1; 0 == False, 1 == True) indicate whether animal was moving at time

    Args:
        facemap_data (dict): Loaded facemap data, can be extracted from get_facemap_data function
        camera_times (numpy arr): Camera times from triggers 
        treadmill_signal (numpy arr; default:None): Raw treadmill analog signal from intan
        treadmill_data (bool; default:True): whether you have treadmill data/have it stored
        cam_fps (float; default: 30): sampling rate of cam, usually 30 hz
        treadmill_fps (float; default: 20e3): sampling rate of intan (20000 hz)
        smoothing_kernel (int; default: 5): sigma factor to smooth motion signal
        movement_percentile (int; default: 70): percentile threshold for movement detection

    Returns:
        state_dataframe (pandas DataFrame): Loaded facemap data 
    """

    # ---------- Helper to avoid redefining logic for motion vs treadmill ----------
    def get_motion_signal(
        raw_signal: np.ndarray,
        camera_times: np.ndarray,
        smoothing_kernel: int,
        movement_percentile: float,
        cam_fps: float,
        min_dur_s: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Shared pipeline:
        - Smooth with gaussian
        - Percentile threshold → boolean
        - Rising/Falling edge detection & start/end handling
        - Enforce minimum duration (in frames at cam_fps)
        - Compute per-signal sampling rate via total_time and index by nearest time
        - Stretch raw / smooth / boolean to camera_times

        Returns:
            smoothed, bool_chunk (post-min-dur), rescaled_indices, stretched_bool, stretched_smooth, stretched_raw, signal_fs
        """
        # Smooth signals with gaussian filter
        smoothed = gaussian_filter1d(raw_signal, sigma=smoothing_kernel)

        # Set threshold for movement. Default: 70
        thresh = np.percentile(smoothed, movement_percentile)

        # Create boolean array of whether value is in top percentile
        bool0 = smoothed > thresh

        # Get differences between boolean values
        # Rising difference (Not moving -> moving) = 1 
        # Falling difference (Moving -> not moving) = -1 
        diffs = np.diff(bool0.astype(np.int8), prepend=bool0[0])
        rising = np.where(diffs == 1)[0]
        falling = np.where(diffs == -1)[0]

        # Handle start/end inside an ON segment
        if bool0[0] and (len(rising) == 0 or (len(rising) and rising[0] > falling[0])):
            rising = np.r_[0, rising]
        if bool0[-1] and (len(falling) == 0 or (len(rising) and falling[-1] < rising[-1])):
            falling = np.r_[falling, len(bool0)]

        # set minimum duration of movement -> default min_dur_s seconds
        min_duration_frames = int(cam_fps * min_dur_s)
        bool_chunk = np.zeros_like(bool0, dtype=bool)

        # Fill segments with min-duration enforcement (pair rises with falls)
        for start, end in zip(rising, falling):
            if (end - start) >= min_duration_frames:
                bool_chunk[start:end] = True

        # ---- Time-based alignment to camera triggers ----
        # Use trigger times and signal_fs to pick nearest sample index
        t0 = camera_times[0]
        total_time = camera_times[-1] - camera_times[0]
        signal_fs = (len(smoothed) - 1) / total_time if total_time > 0 else 0.0

        rescaled_indices = np.round((camera_times - t0) * signal_fs).astype(int)
        rescaled_indices = np.clip(rescaled_indices, 0, len(bool_chunk) - 1)

        # Stretch/interpolate (nearest by time)
        stretched_bool = bool_chunk[rescaled_indices].astype(int)
        stretched_smooth = smoothed[rescaled_indices]
        stretched_raw = raw_signal[rescaled_indices]

        return smoothed, bool_chunk, rescaled_indices, stretched_bool, stretched_smooth, stretched_raw, signal_fs

    # -------------------- Motion branch (kept names/comments) --------------------
    motion_1 = facemap_data['motion'][1]  # (1D np array)

    # Pupil data: optional, if key in facemap data then use it
    pupil = None
    if 'pupil' in facemap_data:
        pupil = facemap_data['pupil']
        pupil_area = pupil[0]['area_smooth']

    # Use shared helper for motion
    motion_smoothed, facial_motion_chunk, rescaled_indices_motion, facial_motion_stretched, fMot_smooth_stretched, fMot_raw_stretched, motion_fs = \
        get_motion_signal(motion_1, camera_times, smoothing_kernel, movement_percentile, cam_fps, min_dur_s)

    print(f'Facemap motion signal contains {len(motion_smoothed)} frames, camera captured {len(camera_times)} frames.')

    # #Percent above threshold on the SAME array used to threshold
    p70 = (motion_smoothed > np.percentile(motion_smoothed, movement_percentile)).mean()
    print("Motion threshold value is:", p70)

    # pupil aligned to motion timebase (same indexing choice as before)
    if pupil is not None:
        pupil_stretched = pupil_area[rescaled_indices_motion]

    # ------------------ Treadmill branch (no duplicate logic) --------------------
    treadmill_smoothed = treadmill_raw_stretched = treadmill_smooth_stretched = None
    treadmill_indices_rescaled = None
    treadmill_fs = 0.0
    treadmill_stretched = None
    treadmill_chunk = None

    if treadmill_data and treadmill_signal is not None:
        (treadmill_smoothed,
         treadmill_chunk,
         treadmill_indices_rescaled,
         treadmill_stretched,
         treadmill_smooth_stretched,
         treadmill_raw_stretched,
         treadmill_fs) = get_motion_signal(
            treadmill_signal, camera_times, smoothing_kernel, movement_percentile, cam_fps, min_dur_s)


    # -------------------- Build DataFrame (kept names/structure) -----------------
    facial_motion_df = pd.DataFrame({
        'time': camera_times,
        #'motion': facial_motion_stretched,
        "motion_bool": facial_motion_stretched,
        'motion_raw': fMot_raw_stretched,
        'motion': fMot_smooth_stretched,
    })

    if pupil is not None:
        facial_motion_df['pupil_area'] = pupil_stretched

    if treadmill_data and treadmill_signal is not None:
        treadmill_df = pd.DataFrame({
            #'locomotion': treadmill_stretched,
            "locomotion_bool": treadmill_stretched,
            'treadmill_raw': treadmill_raw_stretched,          # uses treadmill indices (fixed)
            'treadmill': treadmill_smooth_stretched,
        }) 
        facial_motion_df = pd.concat([facial_motion_df, treadmill_df], axis=1)

    # Keep annotate_state usage and column name
    facial_motion_df['state'] = [None] * len(facial_motion_df)
    facial_motion_df = facial_motion_df.apply(lambda x: annotate_state(x), axis=1)

    if to_parquet:
        if parquet_output_name is not None:
            if not parquet_output_name.endswith('.parquet'):
                parquet_output_name += '.parquet'
            facial_motion_df.to_parquet(parquet_output_name)
        else:
            print(f'Did not save dataframe to parquet file. No output file name was provided.')

    if to_csv:
        if csv_output_name is not None:
            if not csv_output_name.endswith('.csv'):
                csv_output_name += '.csv'
            facial_motion_df.to_csv(csv_output_name)
        else:
            print(f'Did not save dataframe to CSV file. No output file name was provided.')

    return facial_motion_df

'''GET STATE FROM MOTION AND LOCOMOTION BOOLEAN VALUES'''
def annotate_state(state_df_row):
    if state_df_row['motion_bool'] and state_df_row['locomotion_bool']:
        state_df_row['state'] = 'aroused'
    elif state_df_row['motion_bool'] and not state_df_row['locomotion_bool']:
        state_df_row['state'] = 'quiet awake'    
    elif not state_df_row['motion_bool'] and not state_df_row['locomotion_bool']:
        state_df_row['state'] = 'unaroused' 
    return state_df_row

### USE FOR PLOTTING COLORED STATE TIMELINE
### USE AS PROVIDED ARGUMENT FOR X VALS -> 
# in the format [(x_start_1, segment_len_1), (x_start_2, segment_len_2)...]
def state_timeline_ranges(state_indices, restrict_range =False,
                           start_s=None, end_s=None, fps=30):
    idx_ranges = []
    last_start_idx = state_indices[0]
    last_idx_seen = state_indices[0]
    curr_segment_len = 0

    for state_idx in state_indices[1:]:
        curr_segment_len += 1
        if state_idx > last_idx_seen + 2:
            idx_ranges.append((last_start_idx, curr_segment_len))
            curr_segment_len = 0
            last_start_idx = state_idx
        last_idx_seen = state_idx
    idx_ranges = np.array(idx_ranges, dtype=tuple)
    try:
        if restrict_range and start_s is not None and end_s is not None:
            # get samples in range
            idx_ranges = np.array(idx_ranges, dtype=tuple)
            indices_in_range = np.where((idx_ranges[:,0] < end_s*fps) & (idx_ranges[:,0] > start_s*fps))[0]
            idx_ranges = idx_ranges[indices_in_range]
    except:
        traceback.print_exc()
    return [tuple(range) for range in idx_ranges / 30]

def state_samples_map(state_df):
    aroused_samples = state_df[state_df['state'] == 'aroused']
    unaroused_samples = state_df[state_df['state'] == 'unaroused']
    quiet_awake_samples = state_df[state_df['state'] == 'quiet awake']
    return {'aroused': aroused_samples, 'unaroused': unaroused_samples,
             'quiet awake': quiet_awake_samples}

def plot_state_timeline(state_df, dff_trace, start=None, end=None):
    state_map = state_samples_map(state_df)
    if start is None or end is None:
        start, end = 0, len(state_df)
    aroused_timeranges = state_timeline_ranges(state_map['aroused'].index,
                                                            restrict_range=True, start_s=state_df['time'].iloc[0], end_s=state_df['time'].iloc[-1]) 
    unaroused_timeranges = state_timeline_ranges(state_map['unaroused'].index,
                                                            restrict_range=True, start_s=state_df['time'].iloc[0], end_s=state_df['time'].iloc[-1])
    quiet_awake_timeranges = state_timeline_ranges(state_map['quiet awake'].index, 
                                                            restrict_range=True, start_s=state_df['time'].iloc[0], end_s=state_df['time'].iloc[-1])
    fig, axs = plt.subplots(5, 1, figsize=(50, 30))
    start, end = 0, len(state_df)
    axs[0].plot(np.arange(start, end), state_df['motion'][start:end], color='black')
    axs[1].plot(np.arange(start, end), state_df['treadmill'][start:end], color='black')
    axs[2].plot(np.arange(start, end), state_df['pupil_area'][start:end], color='black')
    #axs[0].axhline(y=18379.056640625, color='r', linestyle='--', label='Reference Line')
    #axs[2].axhline(y=1.2562316279032602, color='r', linestyle='--', label='Reference Line')
    axs[3].broken_barh(aroused_timeranges, (0.2,0.2), facecolors=("black"), label='aroused')
    axs[3].broken_barh(quiet_awake_timeranges, (0.2, 0.2), facecolors=("#4436426E"), label='quiet awake')
    axs[3].broken_barh(unaroused_timeranges, (0.2,  0.2), facecolors=("#E9E9E96D"), label='unaroused')
    axs[3].legend(loc='upper right')
    axs[0].set_title('Facial motion')
    axs[1].set_title('Locomotion')
    axs[2].set_title('Pupil area')
    axs[3].set_title('State timeline')
    axs[4].plot(np.arange(len(dff_trace)), dff_trace)
