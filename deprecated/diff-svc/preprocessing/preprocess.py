import argparse
import math
import os
import wave
from pathlib import Path

from pydub import AudioSegment
from tqdm import tqdm


class SplitWavAudioMubin:
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + "/" + filename

        self.audio = AudioSegment.from_wav(self.filepath)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_sec, to_sec, split_filename, out_dir):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(out_dir + "/" + split_filename, format="wav")

    def multiple_split(self, sec_per_split, out_dir):
        total_sec = self.get_duration()
        for i in range(0, math.ceil(total_sec), sec_per_split):
            split_fn = str(i) + "_" + self.filename
            self.single_split(i, i + sec_per_split, split_fn, out_dir)
            # if i == total_sec - sec_per_split:
            #     print('All splited successfully')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("preprocess manager")
    parser.add_argument("--exp_name", help="experiment name", type=str)
    parser.add_argument(
        "--infer",
        help="whether preprocessing is for inference - inference do not require splitting",
        action="store_true",
    )
    args = parser.parse_args()

    # select target directory
    exp_name = args.exp_name
    sr_conversion = []
    dataset_path = "/input/vessl-diff-svc"
    if args.infer:
        sr_conversion = [f"/infer/{e}" for e in os.listdir(f"/infer")]
        print(f"===== conversion target: {sr_conversion} =====")
    else:
        voice_dir = f"/voice"

        logging_dir = f"{dataset_path}/assets/logging"
        for file in os.listdir(logging_dir):
            audio_path = f"{logging_dir}/{file}"
            sr_conversion.append(audio_path)

        processed_dir = f"./raw/{exp_name}/processed"
        Path(processed_dir).mkdir(parents=True, exist_ok=True)

        for voice_file in tqdm(os.listdir(voice_dir)):
            # Check frame rate. Convert if necessary.
            file_path = f"{voice_dir}/{voice_file}"
            with wave.open(file_path, "rb") as wave_file:
                frame_rate = wave_file.getframerate()
                if frame_rate > 24000:
                    print(f"===== Convert Framerate from {frame_rate}Hz to 24kHz =====")
                    sound = AudioSegment.from_file(
                        file_path, format="wav", frame_rate=frame_rate
                    )
                    sound = sound.set_frame_rate(24000)
                    sound.export(file_path, format="wav")
                elif frame_rate == 24000:
                    print(f"===== Audio sample rate is 24kHz, skip =====")
                    pass
                else:
                    print("===== Audio sample rate is smaller than 24kHz =====")
                    raise NotImplementedError
            # Split voice files: each split should be about 15 seconds.
            split_wav = SplitWavAudioMubin(voice_dir, voice_file)
            split_wav.multiple_split(sec_per_split=15, out_dir=f"{processed_dir}")

    for tmp_file in sr_conversion:
        with wave.open(tmp_file, "rb") as wave_file:
            frame_rate = wave_file.getframerate()
            if frame_rate > 24000:
                print(f"===== Convert Framerate from {frame_rate}Hz to 24kHz =====")
                sound = AudioSegment.from_file(tmp_file, format="wav", frame_rate=44100)
                sound = sound.set_frame_rate(24000)
                sound.export(tmp_file, format="wav")
            elif frame_rate == 24000:
                print(f"===== Audio sample rate is 24kHz, skip =====")
                pass
            else:
                print("===== Audio sample rate is smaller than 24kHz =====")
                raise NotImplementedError
