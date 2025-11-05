import numpy as np, matplotlib.pyplot as plt
import traceback

'''GET STATE FROM MOTION AND LOCOMOTION BOOLEAN VALUES'''
def annotate_state(state_df_row):
    if state_df_row['motion_bool']:
        state_df_row['state'] = 'aroused'
    else:
        state_df_row['state'] = 'unaroused'
    return state_df_row

def annotate_state_pupil(state_df_row):
    if state_df_row['motion_bool'] and state_df_row['pupil_bool']:
        state_df_row['state'] = 'aroused'
    elif state_df_row['motion_bool'] and not state_df_row['pupil_bool']:
        state_df_row['state'] = 'quiet awake'
    else:
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
    axs[4].broken_barh(aroused_timeranges, (0.2,0.2), facecolors=("black"), label='aroused')
    axs[4].broken_barh(quiet_awake_timeranges, (0.2, 0.2), facecolors=("#4436426E"), label='quiet awake')
    axs[4].broken_barh(unaroused_timeranges, (0.2,  0.2), facecolors=("#E9E9E96D"), label='unaroused')
    axs[4].legend(loc='upper right')
    axs[0].set_title('Facial motion')
    axs[1].set_title('Locomotion')
    axs[2].set_title('Pupil area')
    axs[2].set_title('Spiking')
    axs[4].set_title('State timeline')
    axs[3].plot(np.arange(len(dff_trace)), dff_trace, color='black')


def spike_triggered_average_motion(
    spikes, frame_times, motion, window_s=(-6.0, 6.0),
    spike_thresh=None, n_shuffles=500, rng=None
):
    """
    STA of facial motion around spikes, with null (chance) distribution via circular shuffles.
    Returns:
      tau: time lags (s)
      sta_mean: mean STA across spikes
      sta_sem:  SEM across spikes
      z_sta: z-score of STA vs shuffle null (per lag)
    """
    rng = np.random.default_rng(rng)
    ft = np.asarray(frame_times, float)
    m  = np.asarray(motion, float)
    sp = np.asarray(spikes, float)

    # bin width (assume nearly constant)
    dt = np.median(np.diff(ft))
    pre, post = window_s
    k_pre  = int(np.floor(abs(pre) / dt))
    k_post = int(np.floor(abs(post) / dt))
    tau = np.arange(-k_pre, k_post + 1) * dt

    # pool spikes across cells or pick a subset
    if spike_thresh is None:
        spike_thresh = np.nanpercentile(sp[sp>0], 50) if np.any(sp>0) else 0.0
    spike_mask = sp > spike_thresh
    spike_idx = np.where(spike_mask.sum(axis=0) > 0)[0]  # any cell spiked in that frame
    # (If you want per-cell STAs, loop cells instead of pooling.)

    # collect windows
    segs = []
    for t0 in spike_idx:
        a = t0 - k_pre
        b = t0 + k_post + 1
        if a < 0 or b > m.size:
            continue
        segs.append(m[a:b])
    if len(segs) == 0:
        raise ValueError("No spike-centered windows within bounds.")
    segs = np.vstack(segs)  # (n_spikes, win_len)

    sta_mean = np.nanmean(segs, axis=0)
    sta_sem  = np.nanstd(segs, axis=0, ddof=1) / np.sqrt(segs.shape[0])

    # --- build shuffle null via circularly shifting motion ---
    win_len = segs.shape[1]
    null_means = np.empty((n_shuffles, win_len), float)
    for s in range(n_shuffles):
        shift = rng.integers(0, m.size)
        m_shuf = np.roll(m, shift)
        segs_shuf = []
        for t0 in spike_idx:
            a = t0 - k_pre
            b = t0 + k_post + 1
            if a < 0 or b > m.size:
                continue
            segs_shuf.append(m_shuf[a:b])
        segs_shuf = np.vstack(segs_shuf)
        null_means[s] = np.nanmean(segs_shuf, axis=0)

    null_mu = null_means.mean(axis=0)
    null_sd = null_means.std(axis=0, ddof=1) + 1e-12
    z_sta = (sta_mean - null_mu) / null_sd

    return tau, sta_mean, sta_sem, z_sta

def plot_sta(tau, sta_mean, sta_sem, z_sta=None, title=None):
    # Left panel look: mean ± SEM and dashed line at 0 s
    plt.figure(figsize=(4,3))
    plt.fill_between(tau, sta_mean - sta_sem, sta_mean + sta_sem, alpha=0.3)
    plt.plot(tau, sta_mean)
    plt.axvline(0, linestyle='--')
    plt.xlabel("Delay (s)")
    plt.ylabel("STA (motion)")
    plt.tight_layout()
    plt.show()
    plt.title('Population: spike triggered average by facial motion')

    if z_sta is not None:
        plt.figure(figsize=(4,3))
        plt.plot(tau, z_sta)
        plt.axvline(0, linestyle='--')
        plt.xlabel("Delay (s)")
        plt.ylabel("STA (z-score vs chance)")
        plt.tight_layout()
        plt.show()
    if title is not None:
        plt.title(title)