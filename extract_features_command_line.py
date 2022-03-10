#!/usr/bin/python
import os
import pandas as pd
import glob 
import librosa
import numpy as np
import scipy
import scipy.io.wavfile
# Modify openSMILE paths HERE:
SMILEpath = 'tools/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract'
SMILEconf = 'tools/opensmile-2.3.0/config/ComParE_2016.conf'

# Task name
task_name = 'CVAD-'

# Paths
audio_folder = 'chunked_1/'
features_folder = 'features/'
normalised = 'normalised/'
def normalise_data():
	if not os.path.isdir(normalised):
		os.mkdir(normalised)
	
	for audio_file in glob.glob(audio_folder +'*.wav'):
		print(audio_file)
		fname = audio_file.split('/')[1]
		if not os.path.exists(normalised + fname):
			audio, sr = librosa.core.load(audio_file, sr=16000)
			amp =np.max(np.abs(audio))
			audio = audio * (0.7079 / np.max(np.abs(audio)))
			maxv = np.iinfo(np.int16).max
			audio = (audio * maxv).astype(np.int16)
			scipy.io.wavfile.write(normalised + fname, sr, audio)
		else:
			print('skip')

def extract_os():
	audio_folder = normalised
	if not os.path.isdir(features_folder):
		os.mkdir(features_folder)

	# Extract openSMILE features standard ComParE and LLD-only
	for inst in glob.glob(audio_folder + '*.wav'):
		filename = inst.split('/')[1]
		output_file      = features_folder + filename[:-4] + '.ComParE.csv'
		output_file_lld  = features_folder + filename[:-4] + '.ComParE-LLD.csv'
		if not os.path.exists(output_file):
			if os.path.exists(output_file):
				os.remove(output_file)
			if os.path.exists(output_file_lld):
				os.remove(output_file_lld)
			print(inst)
			os.system(SMILEpath + ' -C ' + SMILEconf + ' -I ' + inst + ' -instname ' + inst + ' -csvoutput '+ output_file + ' -timestampcsv 0 -lldcsvoutput ' + output_file_lld + ' -appendcsvlld 0')
		else:
			print('skip')

if __name__ == "__main__":
	normalise_data()
	extract_os()
