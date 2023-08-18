import io
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile
from infer_tools import infer_tool, slicer
from infer_tools.infer_tool import Svc
from utils.hparams import hparams

chunks_dict = infer_tool.read_temp("./infer_tools/new_chunks_temp.json")


def run_clip(
    svc_model,
    key,
    acc,
    use_pe,
    use_crepe,
    thre,
    use_gt_mel,
    add_noise_step,
    project_name="",
    f_name=None,
    file_path=None,
    out_path=None,
    slice_db=-40,
    **kwargs,
):
    # print(f"code version:2022-12-04")
    use_pe = use_pe if hparams["audio_sample_rate"] == 24000 else False
    use_pe = False
    if file_path is None:
        raw_audio_path = f"./raw/{f_name}"
        clean_name = f_name[:-4]
    else:
        raw_audio_path = file_path
        clean_name = str(Path(file_path).name)[:-4]
    infer_tool.format_wav(raw_audio_path)
    wav_path = Path(raw_audio_path).with_suffix(".wav")
    global chunks_dict
    audio, sr = librosa.load(wav_path, mono=True, sr=None)
    wav_hash = infer_tool.get_md5(audio)
    if wav_hash in chunks_dict.keys():
        chunks = chunks_dict[wav_hash]["chunks"]
    else:
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
    chunks_dict[wav_hash] = {"chunks": chunks, "time": int(time.time())}
    infer_tool.write_temp("./infer_tools/new_chunks_temp.json", chunks_dict)
    audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

    count = 0
    f0_tst = []
    f0_pred = []
    audio = []
    for slice_tag, data in audio_data:
        length = int(np.ceil(len(data) / audio_sr * hparams["audio_sample_rate"]))
        raw_path = io.BytesIO()
        soundfile.write(raw_path, data, audio_sr, format="wav")
        raw_path.seek(0)
        if slice_tag:
            _f0_tst, _f0_pred, _audio = (
                np.zeros(int(np.ceil(length / hparams["hop_size"]))),
                np.zeros(int(np.ceil(length / hparams["hop_size"]))),
                np.zeros(length),
            )
        else:
            _f0_tst, _f0_pred, _audio = svc_model.infer(
                raw_path,
                key=key,
                acc=acc,
                use_pe=use_pe,
                use_crepe=use_crepe,
                thre=thre,
                use_gt_mel=use_gt_mel,
                add_noise_step=add_noise_step,
            )
        fix_audio = np.zeros(length)
        fix_audio[:] = np.mean(_audio)
        fix_audio[: len(_audio)] = _audio[
            0 if len(_audio) < len(fix_audio) else len(_audio) - len(fix_audio) :
        ]
        f0_tst.extend(_f0_tst)
        f0_pred.extend(_f0_pred)
        audio.extend(list(fix_audio))
        count += 1

    soundfile.write(
        out_path,
        audio,
        hparams["audio_sample_rate"],
        "PCM_16",
        format=out_path.split(".")[-1],
    )
    return np.array(f0_tst), np.array(f0_pred), audio


def infer_on_target(project_name, model_path, target_file, out_file):
    config_path = f"./training/config.yaml"

    accelerate = 20
    hubert_gpu = True
    format = "flac"
    model = Svc(project_name, config_path, hubert_gpu, model_path)
    pitch_shift = 0
    speedup = 5
    wav_output = out_file
    add_noise_step = 500
    threshold = 0.05
    use_crepe = False
    use_pe = False
    use_gt_mel = True

    f0_test, f0_pred, audio = run_clip(
        model,
        file_path=target_file,
        key=pitch_shift,
        acc=speedup,
        use_crepe=use_crepe,
        use_pe=use_pe,
        thre=threshold,
        use_gt_mel=use_gt_mel,
        add_noise_step=add_noise_step,
        project_name=project_name,
        out_path=wav_output,
    )
