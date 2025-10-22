import os,traceback, numpy as np
from intan.importrhdutilities import read_header
from utils.filtering import *
import pandas as pd
from intan.readIntan import get_intan_files 

#import data, fs_intan, convertUnitsVolt, header = get_intan_data(intan_basepath)
'''Python translated matlab code inspired by buz_code function LFPfromDat: https://github.com/buzsakilab/buzcode'''

'''Check if cuda available and cupy installed. '''
try:
    import cupy as cp
    from cupyx.scipy.signal import firwin as firwin_gpu, filtfilt as filtfilt_gpu
    os.environ["CUPY_NVRTC_EXTRA_FLAGS"] = "--std=c++17"
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    print("CuPy not available — using CPU fallback.")

def extract_lfp(basepath, out_fs=1250, lopass=450, output_lfppath=None, csv_savepath=None, overwrite=True):
    '''Extract LFP and store as .lfp binary and an lfp csv for easy use

    Args: 
        basepath (str): input folder that contains amplifier_analogin_auxiliary_int16.dat file OR same dat file renamed as {basepath_name}.dat, 
                        that folder should also contain an info.rhd file

        out_fs (int): sampling rate to downsample to

        lopass (int): lowpass filter cutoff frequency

        output_lfppath (str, Path-like; default:None): name of directory to output LFP file. If None, save to base

        csv_savepath (str, Path-like; default:None): if not None, save a CSV of lfp data to 

        overwrite (bool): if True and path already exists, skip write. Else write .lfp file.

    Returns:
        LFP (numpy mmap): 

        None (save as .lfp or .csv)

    '''
    try:
        amp_analog_aux_in, intan_header, time_file = get_intan_files(basepath)
        fsize = os.path.getsize(amp_analog_aux_in)
        basename = os.path.basename(basepath)

        if output_lfppath is None:
            lfp_out = os.path.join(basepath, f"{basename}.lfp")
        else:
            if not output_lfppath.endswith('.lfp'):
                output_lfppath += '.lfp'
            lfp_out = output_lfppath

            # if not overwrite and os.path.exists(output_lfppath):
            #     print(f'.lfp file already found in data. skipping overwrite. \
            #         change overwrite=True to overwrite .lfp file.')
            #     return
            
        if csv_savepath is not None:
            if not csv_savepath.endswith('.csv'):
                csv_savepath += '.csv'
        #get sampling rate and num amp channels from header
        in_fs = intan_header['sample_rate']
        num_channels = intan_header['num_amplifier_channels'] + intan_header['num_board_adc_channels'] + intan_header['num_aux_input_channels']
        print(f'Found {num_channels} channels recorded at {in_fs}')
        # each sample is int16 = 2 bytes
        bytes_per_sample = 2  
        # calculate downsampling factor 
        sample_ratio = int(in_fs / out_fs)
        chunk_size = int(1e5)
        #check if chunk size is divisible by downsampling factor 
        if chunk_size % sample_ratio != 0:
            # if not divisible, add difference of ratio by the remainder of chunk size and sample ratio
            chunk_size += sample_ratio - (chunk_size % sample_ratio)
        # get process batches -> divide file size by num_chunk_bytes * num channels * int16 per sample = bytes per sample * bytes per chunk
        n_batches = fsize // (chunk_size * num_channels * bytes_per_sample)
        print(f'Processing in {n_batches} batches.')
        # open input signal file and output file to write
        with open(amp_analog_aux_in, "rb") as fid_in, open(lfp_out, "wb") as fid_out:
            downsampled = []
            batch_num = 0
            while True:
                raw = np.fromfile(fid_in, dtype=np.int16, count=chunk_size * num_channels)
                if raw.size == 0:
                    break  # End of file
                if raw.size % num_channels != 0:
                    print(f"Warning: Incomplete sample at end of file. Dropping {raw.size % num_channels} values.")
                    raw = raw[:raw.size - (raw.size % num_channels)]
                data = raw.reshape(-1, num_channels).T
                filtered = sinc_lowpass_filter(data, cutoff=lopass, fs=in_fs)
                downsampled_batch = filtered[:, ::sample_ratio]
                downsampled_float = downsampled_batch.astype(np.float32)
                if GPU_AVAILABLE:
                    downsampled_int = cp.asnumpy(cp.around(downsampled_batch).astype(np.int16))
                    downsampled_float = cp.asnumpy(downsampled_float)
                else:
                    downsampled_int = np.around(downsampled_batch).astype(np.int16)
                downsampled_int.tofile(fid_out)
                downsampled.append(downsampled_float)
                batch_num += 1
        final_data = np.hstack(downsampled)
        f"LFP extraction complete. Saved to: {lfp_out}"
        if csv_savepath is not None:
            np.savetxt(csv_savepath, final_data.T, delimiter=",", fmt="%.4f")
        return final_data
    except Exception as e:
        print(e)
        traceback.print_exc()

def extract_lfp_preallocated(basepath, out_fs=1250, lopass=450, output_lfppath=None, csv_savepath=None, return_data=True):
        """
        FASTER, Same as function above, but speed up by preallocating numpy memory map before write
        """
        amp_analog_aux_in, intan_header, time_file = get_intan_files(basepath)
        basename = os.path.basename(basepath)

        if output_lfppath is None:
            lfp_out = os.path.join(basepath, f"{basename}.lfp")
        else:
            if not output_lfppath.endswith('.lfp'):
                output_lfppath += '.lfp'
            lfp_out = output_lfppath
        if csv_savepath is not None:
            if not csv_savepath.endswith('.csv'):
                csv_savepath += '.csv'
        #get sampling rate and num amp channels from header
        in_fs = intan_header['sample_rate']
        num_channels = intan_header['num_amplifier_channels'] + intan_header['num_board_adc_channels'] + intan_header['num_aux_input_channels']
    
        sample_ratio = int(in_fs / out_fs)
        # samples are int16 
        bytes_per_sample = np.dtype(np.int16).itemsize
        fsize = os.path.getsize(amp_analog_aux_in)
        total_frames = fsize // (bytes_per_sample * num_channels)  
        # Output length (time dimension after decimation)
        out_frames = (total_frames + sample_ratio - 1) // sample_ratio
        #check if chunk size is divisible by downsampling factor 
        if chunk_size % sample_ratio != 0:
            # if not divisible, add difference of ratio by the remainder of chunk size and sample ratio
            chunk_size += sample_ratio - (chunk_size % sample_ratio)
        # get process batches -> divide file size by num_chunk_bytes * num channels * int16 per sample = bytes per sample * bytes per chunk
        n_batches = fsize // (chunk_size * num_channels * bytes_per_sample)
        print(f'Processing in {n_batches} batches.')

        lfp_out_map = np.memmap(lfp_out, dtype=np.int16, mode="w+", shape=(out_frames, num_channels))

        batch_num = 0
        out_pointer = 0

        with open(amp_analog_aux_in, "rb") as fid_in:
            downsampled_float_list = []

            while True:
                raw = np.fromfile(fid_in, dtype=np.int16, count=chunk_size * num_channels)
                if raw.size == 0:
                    break  # End of file

                if raw.size % num_channels != 0:
                    print(f"Warning: Incomplete sample at end of file. Dropping {raw.size % num_channels} values.")
                    raw = raw[:raw.size - (raw.size % num_channels)]

                # reshape into (channels, samples)
                data = raw.reshape(-1, num_channels).T

                # filter and downsample
                filtered = sinc_lowpass_filter(data, cutoff=lopass, fs=in_fs)
                downsampled_batch = filtered[:, ::sample_ratio]

                # float copy if needed later
                downsampled_float = downsampled_batch.astype(np.float32)

                if GPU_AVAILABLE:
                    downsampled_int = cp.asnumpy(cp.around(downsampled_batch).astype(np.int16))
                    downsampled_float = cp.asnumpy(downsampled_float)
                else:
                    downsampled_int = np.around(downsampled_batch).astype(np.int16)

                # write into memmap
                n_out = downsampled_int.shape[1]
                lfp_out_map[out_pointer:out_pointer + n_out, :] = downsampled_int.T
                out_pos += n_out

                downsampled_float_list.append(downsampled_float)
                batch_num += 1

        # flush memmap to disk
        lfp_out_map.flush()

        # also return float data in RAM
        final_data = np.hstack(downsampled_float_list)

        print(f"LFP extraction complete. Saved to: {lfp_out}")

        if csv_savepath is not None:
            np.savetxt(csv_savepath, final_data.T, delimiter=",", fmt="%.4f")

        return final_data


