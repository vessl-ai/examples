import struct
import warnings
from io import BytesIO

import librosa
import numpy as np
import parselmouth
import pyloudnorm as pyln
import resampy
import torch
import torchcrepe
import webrtcvad
from scipy.ndimage.morphology import binary_dilation
from skimage.transform import resize
from utils import audio
from utils.pitch_utils import f0_to_coarse

warnings.filterwarnings("ignore")
PUNCS = "!,.?;:"

int16_max = (2**15) - 1


def trim_long_silences(
    path, sr=None, return_raw_wav=False, norm=True, vad_max_silence_length=12
):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.
    :param wav: the raw waveform as a numpy array of floats
    :param vad_max_silence_length: Maximum number of consecutive silent frames a segment can have.
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """

    ## Voice Activation Detection
    # Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
    # This sets the granularity of the VAD. Should not need to be changed.
    sampling_rate = 16000
    wav_raw, sr = librosa.core.load(path, sr=sr)

    if norm:
        meter = pyln.Meter(sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav_raw)
        wav_raw = pyln.normalize.loudness(wav_raw, loudness, -20.0)
        if np.abs(wav_raw).max() > 1.0:
            wav_raw = wav_raw / np.abs(wav_raw).max()

    wav = librosa.resample(wav_raw, sr, sampling_rate, res_type="kaiser_best")

    vad_window_length = 30  # In milliseconds
    # Number of frames to average together when performing the moving average smoothing.
    # The larger this value, the larger the VAD variations must be to not get smoothed out.
    vad_moving_average_width = 8

    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[: len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack(
        "%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16)
    )

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(
            vad.is_speech(
                pcm_wave[window_start * 2 : window_end * 2], sample_rate=sampling_rate
            )
        )
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate(
            (np.zeros((width - 1) // 2), array, np.zeros(width // 2))
        )
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1 :] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    audio_mask = resize(audio_mask, (len(wav_raw),)) > 0
    if return_raw_wav:
        return wav_raw, audio_mask, sr
    return wav_raw[audio_mask], audio_mask, sr


def process_utterance(
    wav_path,
    fft_size=1024,
    hop_size=256,
    win_length=1024,
    window="hann",
    num_mels=80,
    fmin=80,
    fmax=7600,
    eps=1e-6,
    sample_rate=22050,
    loud_norm=False,
    min_level_db=-100,
    return_linear=False,
    trim_long_sil=False,
    vocoder="pwg",
):
    if isinstance(wav_path, str) or isinstance(wav_path, BytesIO):
        if trim_long_sil:
            wav, _, _ = trim_long_silences(wav_path, sample_rate)
        else:
            wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path
    if loud_norm:
        meter = pyln.Meter(sample_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -22.0)
        if np.abs(wav).max() > 1:
            wav = wav / np.abs(wav).max()

    # get amplitude spectrogram
    x_stft = librosa.stft(
        wav,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="constant",
    )
    spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(sample_rate, fft_size, num_mels, fmin, fmax)
    mel = mel_basis @ spc

    if vocoder == "pwg":
        mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    else:
        assert False, f'"{vocoder}" is not in ["pwg"].'

    l_pad, r_pad = audio.librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode="constant", constant_values=0.0)
    wav = wav[: mel.shape[1] * hop_size]

    if not return_linear:
        return wav, mel
    else:
        spc = audio.amp_to_db(spc)
        spc = audio.normalize(spc, {"min_level_db": min_level_db})
        return wav, mel, spc


def get_pitch_parselmouth(wav_data, mel, hparams):
    """

    :param wav_data: [T]
    :param mel: [T, 80]
    :param hparams:
    :return:
    """
    time_step = hparams["hop_size"] / hparams["audio_sample_rate"]
    f0_min = hparams["f0_min"]
    f0_max = hparams["f0_max"]

    f0 = (
        parselmouth.Sound(wav_data, hparams["audio_sample_rate"])
        .to_pitch_ac(
            time_step=time_step,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        )
        .selected_array["frequency"]
    )

    pad_size = (int(len(wav_data) // hparams["hop_size"]) - len(f0) + 1) // 2
    f0 = np.pad(f0, [[pad_size, len(mel) - len(f0) - pad_size]], mode="constant")
    pitch_coarse = f0_to_coarse(f0, hparams)
    return f0, pitch_coarse


def get_pitch_crepe(wav_data, mel, hparams, threshold=0.05):
    device = torch.device("cuda")
    wav16k = resampy.resample(wav_data, hparams["audio_sample_rate"], 16000)
    wav16k_torch = torch.FloatTensor(wav16k).unsqueeze(0).to(device)

    f0_min = hparams["f0_min"]
    f0_max = hparams["f0_max"]

    f0, pd = torchcrepe.predict(
        wav16k_torch,
        16000,
        80,
        f0_min,
        f0_max,
        pad=True,
        model="full",
        batch_size=1024,
        device=device,
        return_periodicity=True,
    )

    pd = torchcrepe.filter.median(pd, 3)
    pd = torchcrepe.threshold.Silence(-60.0)(pd, wav16k_torch, 16000, 80)
    f0 = torchcrepe.threshold.At(threshold)(f0, pd)
    f0 = torchcrepe.filter.mean(f0, 3)

    f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)

    nzindex = torch.nonzero(f0[0]).squeeze()
    f0 = torch.index_select(f0[0], dim=0, index=nzindex).cpu().numpy()
    time_org = 0.005 * nzindex.cpu().numpy()
    time_frame = (
        np.arange(len(mel)) * hparams["hop_size"] / hparams["audio_sample_rate"]
    )
    if f0.shape[0] == 0:
        f0 = torch.FloatTensor(time_frame.shape[0]).fill_(0)
        print("f0 all zero!")
    else:
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
    pitch_coarse = f0_to_coarse(f0, hparams)
    return f0, pitch_coarse


def is_sil_phoneme(p):
    return not p[0].isalpha()
