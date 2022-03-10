import webrtcvad
from src.utils.wavfile_utils import read, frame_generator, vad_collector
from tqdm import tqdm
from src.utils.path_utils import get_system_dependendent_paths, make_directory
from glob import glob
from src.utils.constants import *
import pandas as pd
from os.path import basename, splitext


executing_PC = LOCAL

nas_dir, code_dir = get_system_dependendent_paths(LOCAL)

audio_dir = nas_dir + "data_work/manuel/data/EMBOA/Valence_Arousal/audio_normalised/"
vad_output_dir = nas_dir + "data_work/manuel/data/EMBOA/Valence_Arousal/webrtcvad_alternatv/"
make_directory(vad_output_dir)
audios = glob(audio_dir + "*_4*")
audios.sort()

for audio_file in tqdm(audios):
    sample_rate, audio = read(audio_file)
    vad = webrtcvad.Vad(1)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    vad_df = pd.DataFrame(columns=["start", "end", "label"])
    for i, (start, end) in enumerate(segments):
        vad_df = vad_df.append(pd.DataFrame([[start, end, "speaker"]], columns=["start", "end", "label"]))
    out_file_name = vad_output_dir + splitext(basename(audio_file))[0] + ".csv"
    vad_df.to_csv(out_file_name, index=False)
