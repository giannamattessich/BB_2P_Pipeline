import os
import numpy as np
import pickle, joblib
from intan.readIntan import get_intan_data


def get_all_intan_data(intan_basepath, twop_chan=2, pd_chan=5, camera_chan=3, treadmill_chan=6):
## PROVIDE ANALOG CHANNELS AS NUM ADC CHANNEL (0-8), NOT ANALOG/AUX TOTAL

    """
    Extract and load data from whole basepath: including 2P and intan data
    
    Args:

        intan_basepath (str, Path-like): path of directory where intan data is located 

        twop_chan (int; default 2): recording channel for scope

        pd_chan (int; default 5): photodiode channel number

        camera_chan (int; default 3): camera channel number

        treadmill_chan (int; default 6): treadmill channel number 

    Returns:

        fs_intan (default is 20e3): intan sampling rate

        phodiode_raw (np.array): raw photodiode signal, None if no channel provided

        twop_raw (np.array): raw 2P scope signal, None if no channel provided

        camera_raw (np.array): raw camera TTL signal, None is no channel provided

        treadmill_raw (np.array): raw treadmill signal, None if no channel

    """
    data, fs_intan, convertUnitsVolt, header = get_intan_data(intan_basepath)
    # if amp channels were recorded, add number amp channels to analog channels, as the 
    # amplifier_analogin_auxiliary_int16.dat file is concatenated with ALL channels
    # if not keep the same analog channel numbers
    num_amp_channels = header['num_amplifier_channels']
    if num_amp_channels > 0:
        twop_chan += num_amp_channels
        pd_chan += num_amp_channels
        camera_chan += num_amp_channels
        treadmill_chan += num_amp_channels
    photodiode_raw, twop_raw, camera_raw, treadmill_raw = None, None, None, None
    try:
        photodiode_raw = data[pd_chan] * convertUnitsVolt
    except:
        print(f'Could not find photodiode data, or channel was not provided.')
    try:
        twop_raw = data[twop_chan] * convertUnitsVolt
    except:
        print(f'Could not find 2P trigger data, or channel was not provided.')
    try:
        camera_raw = data[camera_chan] * convertUnitsVolt
    except:
        print(f'Could not get camera data, or channel was not provided.')
    try:
        treadmill_raw = data[treadmill_chan] * convertUnitsVolt
    except:
        print(f'Could not find treadmill data, or channel was not provided.')
    return fs_intan, photodiode_raw, twop_raw, camera_raw, treadmill_raw

def get_facemap_data(data_basepath):
    """
    HELPER FUNCTION TO INPUT DATA BASEPATH OF WHERE FACEMAP OUTPUT IS STORED,
    OUTPUT IS DICT WITH FACEMAP DATA-> if numpy file is saved it will load data 
    from numpy file. if not found, try to load .mat 

    Args:
        data_basepath (str): Path to directory where facemap data is stored

    Returns:
        facemap_data (dict): Loaded facemap data 
    """
    file_type = None
    for file in os.listdir(data_basepath):
        # if numpy file found, break loop
        if file.endswith('_proc.npy'):
            facemap_file_path = os.path.join(data_basepath, file)
            file_type = 'npy'
            break
        # only use mat file if numpy file doesnt exist
        elif file.endswith('_proc.mat'):
            facemap_file_path = os.path.join(data_basepath, file)
            file_type = 'mat'
    if file_type is None:
        print(f'WARNING!! The facemap file was not found in the path {data_basepath}')
    if file_type == 'npy':
        facemap_data = np.load(facemap_file_path, allow_pickle=True).item()
    elif file_type == 'mat':
        import scipy.io as sio
        facemap_data = sio.loadmat(facemap_file_path)
    print(f'Found facemap file path {facemap_file_path}')
    return facemap_data

'''
Generic function to load .pkl file into memory

Args:
    file_path: .pkl file to read

Returns:
    data(np.ndarray): data from file
'''
def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


# create an array of len (ncells_TOTAL, nframes_TOTAL) by stacking values of all recording and filling with nan for recordings that 
# have less frames than the max num of frames in recordings
def pad_and_stack_vals(values_dict, s2p_outs_dict):
    max_frames_group = max({recording:s2p_out.nframes for recording,s2p_out in s2p_outs_dict.items()}.values())
    padded_arrs = {recording:np.pad(value_arr, pad_width=((0, 0), (0, max_frames_group - value_arr.shape[1])), constant_values=np.nan) 
                            for recording, value_arr in values_dict.items()}
    flat_valuelist = list(padded_arrs.values())
    stacked_arr = np.vstack(flat_valuelist)
    return stacked_arr

def get2p_foldername_field(data_basepath):
    """ 
    Get 2P folder name, and append suite2p path to get the datapath of suite2p files.
    ***NOTE: this only works if you have data saved under folder starting with the word 'field'

    Args:
        data_basepath (str): parent base folder of tif files
    Returns:
        suite2p folder path (str): path to s2p folder
    """
    files = os.listdir(data_basepath)
    foldername = ''
    for folder in files:
        if folder.lower().startswith('field'):
            foldername = os.path.join(data_basepath, folder)
    if foldername == '':
        print(f'Warning: no tif folder found for {os.path.basename(data_basepath)}')
        return None
    return os.path.join(data_basepath, foldername, 'suite2p')