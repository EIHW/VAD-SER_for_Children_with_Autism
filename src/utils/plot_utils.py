import matplotlib.pyplot as plt
import numpy as np

from src.utils.wavfile_utils import read_wav_file_audiofile


def save_image(image, path):
    """
    Plots and saves image under given path
    :param image: image as np-array (or similar)
    :param path: Path to store image
    :return:
    """
    plt.imshow(image)
    plt.savefig(path)

def plot_line_chart_textual_x(names, values, path):
    plt.plot(names, values)
    plt.savefig(path)
    plt.clf()

def simple_line_plot(x,y, path=None):
    plt.plot(x,y)
    if path==None:
        plt.show()
    else:
        plt.savefig(path)
        print("saved: \t" + path)
    plt.clf()

def plot_training(train_results, devel_results, path):
    plt.plot(train_results, label="train")
    plt.plot(devel_results, label="devel")
    plt.legend()
    plt.savefig(path)
    plt.clf()

def plot_value_arrays(path, value_arrays, labels=None):
    for i in range(len(value_arrays)):
        if labels != None:
            label = labels[i]
        plt.plot(value_arrays[i], label=label)
    plt.legend()
    plt.savefig(path)
    plt.clf()

def scatter_plot(data_list, filename, labels=[], label_prefix=""):
    if len(labels) == 0:
        for i, data in enumerate(data_list):
            plt.scatter(data[:, 0], data[:, 1], label="{} - {}h".format(i*6, (i+1)*6))
        plt.legend()
        plt.title("Passive Audio Representations (Single Subject per Quarterday)")
    else:
        for data, label in zip(data_list, labels):
            plt.scatter(data[:, 0], data[:, 1], label=label)
        plt.legend()
        plt.title("Passive Audio Representations (Single Subject per Quarterday)")

    plt.savefig(filename)
    plt.clf()


def plot_audio_file(wav_file, image_file, start=0, end=-1, xticks=0, ylimits=[], size=[], axes=True, fig_size= []):
    signal, sr = read_wav_file_audiofile(wav_file)
    if end != -1:
        signal_chunk = signal[int(start * sr) : int(end * sr)]
    else:
        signal_chunk = signal[int(start * sr) :]
    if len(fig_size) != 0:
        plt.figure(figsize=fig_size, dpi=80)
    x = np.arange(0, len(signal_chunk) / sr, 1 / sr)
    plt.figure(figsize=(8, 2), dpi=80)
    plt.plot(x, signal_chunk)
    if xticks != 0:
        plt.xticks(np.arange(0, len(signal_chunk) / sr, xticks))
    ax = plt.gca()
    if len(ylimits) != 0:
        ax.set_ylim(ylimits)
    if not axes:
        plt.axis("off")
    plt.savefig(image_file, bbox_inches='tight')
    plt.clf()

def plot_audio_file_chunks(wav_file, image_file, chunks, xticks=0, ylimits=[], axes=True, fig_size=[]):
    signal, sr = read_wav_file_audiofile(wav_file)
    signal_chunks = []
    total_length = 0
    for start, end in chunks:
        if end != -1:
            signal_chunks.append(signal[int(start * sr) : int(end * sr)])
        else:
            signal_chunks.append(signal[int(start * sr) :])
        total_length += end - start
    signal_chunks = np.hstack(signal_chunks)
    x = np.arange(0, len(signal_chunks) / sr, 1 / sr)
    if len(fig_size) != 0:
        plt.figure(figsize=fig_size, dpi=80)
    plt.plot(x, signal_chunks)
    if xticks != 0:
        plt.xticks(np.arange(0, total_length, xticks))
    ax = plt.gca()
    if len(ylimits) != 0:
        ax.set_ylim(ylimits)
    if not axes:
        plt.axis("off")
    plt.savefig(image_file)
    plt.clf()

def make_nice_line_plot(path ,Xs, Ys, labels, title="", font_size=12, x_axis="", y_axis="", fig_size=[], colors=[], line_styles=[]):
    plt.figure(figsize=fig_size)
    plt.rcParams.update({'font.size': font_size})
    plt.title(title)
    plt.xlabel(x_axis, fontsize=font_size)
    plt.ylabel(y_axis, fontsize=font_size)
    for i in range(len(Xs)):
        X = Xs[i]
        Y = Ys[i]
        label = labels[i]
        if len(colors) == 0:
            plt.plot(X, Y, label=label)
        else:
            plt.plot(X, Y, color=colors[i], linestyle=line_styles[i], label=label)


    plt.legend()
    plt.savefig(path)
    plt.clf()