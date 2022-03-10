import numpy as np
import opensmile
from glob import glob
from os.path import basename, splitext
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.io.wavfile import read

def extract_opensmile_dir(in_dir, out_dir, config_file, microphones, cultures, feature_type="egemaps"):
    #smile = opensmile.Smile(feature_set=config_file, feature_level="lld")
    #smile = opensmile.Smile(feature_set=opensmile.FeatureSet., feature_level="lld")
    files = glob(in_dir + "*.wav")
    files.sort()
    if feature_type == "egemaps":
        opensmile_options = '-configfile ' + config_file + ' -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1'
        outputoption = '-lldcsvoutput'
    else:
        opensmile_options = '-configfile ' + config_file + ' -appendcsv 0 -timestampcsv 1 -headercsv 1'
        outputoption = '-csvoutput'
    exe_opensmile = '/home/manu/opensmile/opensmile3_src/opensmile-3.0.0/build/progsrc/smilextract/SMILExtract'


    for file in files:
        instname = splitext(basename(file))[0]
        microphone_id = instname[9]
        culture = instname[0]
        if not microphone_id in microphones or not culture in cultures:
            continue
        print(file)
        outfilename = out_dir + instname + '.csv'
        opensmile_call = exe_opensmile + ' ' + opensmile_options + ' -inputfile ' + file + ' ' + outputoption + ' ' + outfilename + ' -instname ' + instname + ' -output ?'  # (disabling htk output)
        os.system(opensmile_call)

def extract_opensmile_dir_python(in_dir, out_dir, microphones, cultures, feature_type="egemaps", window_size=0.025, hop_size=0.01):
    #smile = opensmile.Smile(feature_set=config_file, feature_level="lld")
    #smile = opensmile.Smile(feature_set=opensmile.FeatureSet., feature_level="lld")
    files = glob(in_dir + "*.wav")
    files.sort()



    for file in files:
        instname = splitext(basename(file))[0]
        microphone_id = instname[9]
        culture = instname[0]
        if (not microphone_id in microphones) or (not culture in cultures):
            continue
        print(file)
        feature_df = extract_opensmile_windowed_python(file, feature_type=feature_type, window_size=window_size, hop_size=hop_size)
        outfilename = out_dir + instname + '.csv'
        feature_df.to_csv(outfilename, index=False)

def extract_opensmile_windowed_python(file, feature_type="egemaps", window_size=0.025, hop_size=0.01):
    if feature_type == "egemaps":
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    else:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    sr, signal = read(file)
    hop_size_idx = int(hop_size * sr)
    signal = normalise_signal(signal)
    window_size_idx = int(window_size * sr)
    file_length_idx = signal.shape[0]
    for idx in range(0, file_length_idx - window_size_idx, hop_size_idx):
        signal_chunk = signal[idx:idx + window_size_idx]
        if len(signal_chunk) < window_size_idx:
            signal_chunk = np.concatenate((signal_chunk, np.zeros(window_size_idx - len(signal_chunk))))


        if idx == 0:
            df_full = smile.process_signal(signal_chunk, sr)
            df_full["timestamp"] = idx / sr
        else:
            df = smile.process_signal(signal_chunk, sr)
            df["timestamp"] = idx / sr
            df_full = df_full.append(df)
    return df_full

def normalise_signal(signal):
    scaler = MinMaxScaler()
    signal = signal[..., np.newaxis]
    signal = scaler.fit_transform(signal)
    signal = np.squeeze(signal)
    return signal