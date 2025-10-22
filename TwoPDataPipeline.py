import traceback
from utils.getDataFiles import *
from utils.alignmentFunctions import get_analog_times, align_scope_triggers_to_frames
from intan.readIntan import *
from twop.getSuite2POutput import *
from state.getFacemapData import *

class TwoPData:
    # PROVIDE ANALOG CHANNELS AS NUM ADC CHANNEL (0-8), NOT ANALOG/AUX TOTAL
    '''
    Class to encapsulate ALL recording data 

    Parameters:
        suite2p_basepath (str or Path-like): filepath where suite2p data is located (*MUST end in 'suite2p')
        
        intan_basepath (str or Path-like): filepath where intan data is located (*required: info.rhd and amp_analog_aux_int.dat.. files)

        scope_fps (float; default=1.366): sampling rate of 2P scope

        twop_channel (int; default=2): recording channel of scope

        pd_channel (int; default=5): channel of photodiode

        camera_channel (int; default=3): channel of camera

        treadmill_channel (int; default=6): channel of treadmill
    '''
    def __init__(self, suite2p_basepath, intan_basepath, facemap_path=None, scope_fps= 1.366,
                  twop_channel=2, pd_channel=5, camera_channel=3, treadmill_channel=6):
        # datapaths
        self.suite2p_basepath = suite2p_basepath
        self.intan_basepath = intan_basepath
        self.facemap_path = facemap_path

        # recording channel info
        self.scope_fps = scope_fps
        self.twop_chan = twop_channel
        self.pd_chan = pd_channel
        self.camera_chan = camera_channel
        self.treadmill_chan = treadmill_channel

        if not os.path.exists(self.suite2p_basepath):
            raise ValueError(f'Suite2p path {self.suite2p_basepath} does not exist.')
        
        self.s2p_out = Suite2POutput(self.suite2p_basepath, scope_fs=self.scope_fps)

        (self.fs_intan, self.photodiode_raw, self.twop_raw, self.camera_raw, self.treadmill_raw) = get_all_intan_data(self.intan_basepath,
                                                                                        twop_chan=twop_channel, pd_chan=pd_channel,
                                                                                        camera_chan=camera_channel, treadmill_chan=treadmill_channel)  
        ## get times in seconds of TTL triggers for scope, photodiode, camera, and treadmill
        self.scope_times, self.scope_times_end = get_analog_times(self.twop_raw, upTransition=True)
        try:
            self.scope_times, self.scope_times_end = align_scope_triggers_to_frames(self.s2p_out, self.scope_times)
        except:
            print(f'Could not confirm alignment of scope and triggers.')

        if self.photodiode_raw is not None:
            self.pd_times, self.pd_times_end = get_analog_times(self.photodiode_raw)
        if self.camera_chan is not None:
            self.camera_times, self.camera_times_end = get_analog_times(self.camera_raw)
        if treadmill_channel is not None:
            self.treadmill_times, self.treadmill_times_end = get_analog_times(self.treadmill_raw) 
        # read in facemap data
        if self.facemap_path is not None:
            self.facemap_data = get_facemap_data(self.facemap_path)

    def make_frame_df(self, output_csv=False, output_filepath=None):
        """
        Create a dataframe of relative time estimates from trigger and raw frame times in seconds since recording start

        Args:
            output_csv (bool): whether to output a csv  
            output_filepath (str, Path-like):
        Returns:
            scope_df (pd.DataFrame): dataframe of raw scope times 
        """
        timeEst = np.arange(self.s2p_out.nframes) / self.s2p_out.scope_fs
        scope_df = pd.DataFrame({'timeEst':timeEst, 'frame_time':self.scope_times})
        if output_csv:
            self.df_to_csv(scope_df, output_filepath= output_filepath)
        return scope_df
    
    def make_state_df(self, cam_fps=30, treadmill_data=True, smoothing_kernel=5, movement_percentile=70,
                       output_csv=False, output_filepath=None):
        """
        Create state dataframe (from getFacemapData.py) using data from base folder

        Args:
            cam_fps (int; default is 30): camera ttl rate

            treadmill_data (bool; default is True): whether treadmill data was recorded

            smoothing_kernel (int): smoothing factor for treadmill signal

            movement_percentile (int; default=70): percentile for movement detection

            output_csv (bool): whether to output a csv file, if so provide a path to the output_filepath argument 

            output_filepath (str, Path-like): path to save CSV file to if output_csv == True

        Returns:
            state_dataframe (pd.DataFrame): dataframe containing camera aligned timestamps, treadmill, motion, and pupil signals
        """
        state_dataframe = get_state_df(facemap_data= self.facemap_data,
                                camera_times= self.camera_times,
                                treadmill_data=treadmill_data, treadmill_signal= self.treadmill_raw,
                                    cam_fps=cam_fps, smoothing_kernel=smoothing_kernel, movement_percentile= movement_percentile)
        if output_csv:
            self.df_to_csv(state_dataframe, output_filepath)
        return state_dataframe
    
    def frame_state_df(self, state_df, output_csv=False, output_filepath= None, tolerance=None):
        """
        Create dataframe of camera times, scaled state data, and nearest frames

        Args:  
            state_df (pd.DataFrame): 

            tolerance (float ; default is None): 

        Returns:
            frame_state_df (pd.DataFrame)
        
        """
        # create a copy of state dataframe for modification
        statedf_copy = state_df.copy()

        # min max normalize pupil area and motion
        motion_scaled = MinMaxScaler().fit_transform(statedf_copy['motion'].to_numpy().reshape(-1, 1)).flatten()
        pupil_scaled = MinMaxScaler().fit_transform(statedf_copy['pupil_area'].to_numpy().reshape(-1, 1)).flatten()
        treadmill_scaled = MinMaxScaler().fit_transform(statedf_copy['treadmill'].to_numpy().reshape(-1, 1)).flatten()
        motion_bool, locomotion_bool = statedf_copy['motion_bool'], statedf_copy['locomotion_bool']

        # drop curr cols and replace with minmax normalized 
        #statedf_copy.drop(['motion_raw', 'motion_smooth', 'pupil_area', 'treadmill_raw'], axis=1, inplace=True)

        # create dataframe for used merging to state vals/cam times
        frame_df = pd.DataFrame({'nearest_frame_idx': np.arange(self.s2p_out.nframes),
                                  'frame_start_time': self.scope_times}).reset_index()
        # merge on nearest 
        nearest_frames = pd.merge_asof(statedf_copy[['time']],
                                        frame_df[['nearest_frame_idx', 'frame_start_time']],
                                          left_on='time', right_on='frame_start_time',
                                            direction='forward', tolerance=tolerance)
        frame_state_df = nearest_frames
        data_to_add = {'motion_bool': motion_bool,
                       'locomotion_bool': locomotion_bool,
                       'motion': motion_scaled,
                        'pupil':pupil_scaled,
                          'treadmill':treadmill_scaled}

        frame_state_df = pd.concat([frame_state_df, pd.DataFrame(data_to_add)], axis=1)

        frame_state_df['nearest_frame_idx'] = frame_state_df['nearest_frame_idx'].fillna(-1).astype(int)

        if output_csv:
            self.df_to_csv(frame_state_df, output_filepath=output_filepath)
        return frame_state_df

def df_to_csv(self, dataframe, output_filepath):
    """
    Helper function to save dataframes to a csv file if output_csv = True

    Args:
        dataframe (pd.DataFrame): pandas dataframe to export

        output_filepath (str, Path-like): output path to save dataframe csv. if None, put into data basepath folder

    Returns:
        None, saves dataframe to output_filepath
    """
    if output_filepath is None:
        output_filepath = os.path.join(self.intan_basepath)
    if not output_filepath.endswith('.csv'):
        output_filepath += '.csv'
    try:
        dataframe.to_csv(output_filepath)
    except:
        traceback.print_exc()