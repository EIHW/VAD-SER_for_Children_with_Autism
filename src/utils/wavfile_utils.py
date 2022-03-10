import scipy
from scipy.io.wavfile import write, read
import librosa
import numpy as np
from glob import glob
from os.path import basename, splitext
import collections
import audiofile
import librosa

def read_and_normalise_data(wav_file_path, sr = 16000):
    audio, sr = librosa.core.load(wav_file_path, sr=sr)
    audio = audio * (0.7079 / np.max(np.abs(audio)))
    maxv = np.iinfo(np.int16).max
    audio = (audio * maxv).astype(np.int16)
    return audio

def normalise_and_overwrite_wavfiles_recursively(directory, sr = 16000):
    wav_file_list = glob(directory + "*.wav")
    for wav_file in wav_file_list:
        audio, sr = librosa.core.load(wav_file, sr=sr)
        audio = audio * (0.7079 / np.max(np.abs(audio)))
        maxv = np.iinfo(np.int16).max
        audio = (audio * maxv).astype(np.int16)
        write(wav_file, sr, audio)
    dir_list = glob(directory + "*/")
    for dir in dir_list:
        normalise_and_overwrite_wavfiles_recursively(dir)

def read_wav_file_librosa(wav_file_path):
    """
    :param wav_file_path:
    :return: sample rate, data
    """
    return librosa.load(wav_file_path)

def read_wav_file_audiofile(wav_file_path):
    """
    :param wav_file_path:
    :return: sample rate, data
    """
    signal, sampling_rate = audiofile.read(wav_file_path)
    return signal, sampling_rate

def read_wav_file(wav_file_path):
    """
    :param wav_file_path:
    :return: sample rate, data
    """
    return read(wav_file_path)

def wav_file_length(wav_file_path):
    sr, data = read_wav_file(wav_file_path)
    return len(data)/sr

def chunk_audio_file(in_file, out_dir, chunk_size, zero_pad = True, prefix=""):
    """
    chunks audio data from in_file in chunk_size sec chunks (zero_padded) into separate files in out_dir.
    :param in_file: wav file to chunk
    :param out_dir: directpry where chunked data is written
    :param chunk_size: desired chunk size (in sec)
    :param zero_pad:
    :return: length of the padded audio file and number of written chunks
    """
    basename_no_ext, file_ext = splitext(basename(in_file))
    data, sample_rate = read_wav_file_librosa(in_file)
    if zero_pad:
        zeros_to_pad = np.zeros(sample_rate*chunk_size - len(data)%(sample_rate*chunk_size) )
        data = np.append(data, zeros_to_pad)
    audio_file_length = len(data) / sample_rate
    number_of_chunks = int(audio_file_length // chunk_size)
    leading_zeros_in_path = int(np.ceil(np.log10(number_of_chunks)))
    chunk_length_samples = int(chunk_size * sample_rate)
    for i in range(number_of_chunks):
        chunk_samples = data[i * chunk_length_samples: (i + 1) * chunk_length_samples]
        out_file_name = out_dir + prefix + "_" + basename_no_ext + "_" + str(i).zfill(leading_zeros_in_path) + file_ext
        write(out_file_name, sample_rate, chunk_samples)
    return audio_file_length, number_of_chunks

class Frame(object):
    """
    From Maurice Gerczuk

    Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """
    From Maurice Gerczuk

    Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """
    From Maurice Gerczuk

    Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    start = None
    end = None
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        # sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                start = ring_buffer[0][0].timestamp
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                end = frame.duration + frame.timestamp
                triggered = False
                # yield b''.join([f.bytes for f in voiced_frames])
                yield (start, end)
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        end = frame.timestamp + frame.duration
    # sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        # yield b''.join([f.bytes for f in voiced_frames])
        yield (start, end)
