from src.utils.plot_utils import plot_audio_file, plot_audio_file_chunks

wav_path = "/home/manu/eihw/data_work/manuel/data/EMBOA/Valence_Arousal/audio_normalised/B001_R01_4.wav"
image_path = "audio.png"
image_path_chunked = "audio_chunked.png"
fig_size1 = (8,2)
fig_size2 = (4,2)
ylimits = [-0.5, 0.5]
plot_audio_file(wav_path, image_path, start=10, end=20.1, xticks=1, ylimits=ylimits, axes=False, fig_size=fig_size1)
plot_audio_file_chunks(wav_path, image_path_chunked, chunks=[[13,14], [15,16], [16,17], [19,20.1]], xticks=1, ylimits=ylimits, axes=False, fig_size=fig_size2)