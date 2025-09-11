from cog import BasePredictor, Input, Path
import subprocess, torchaudio, os

class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def predict(
        self,
        audio: Path = Input(description="Input audio file (MP3, WAV, etc.)"),
        media: Path = Input(description="Input face image OR video"),
        input_type: str = Input(
            description="Choose input type",
            choices=["image-to-video", "video-to-video"],
            default="image-to-video"
        ),
        mode: str = Input(
            description="Generation mode",
            choices=["streaming", "clip"],
            default="streaming"
        ),
        resolution: str = Input(
            description="Output resolution",
            choices=["480p", "720p"],
            default="480p"
        ),
        persons: str = Input(
            description="Single or Multi person",
            choices=["single", "multi"],
            default="single"
        ),
        fps: int = Input(description="Frames per second", default=25, ge=1, le=60),
        duration_policy: str = Input(
            description="If audio >30min: cut or skip?",
            choices=["cut", "skip"],
            default="cut"
        ),
    ) -> Path:

        # ---- คำนวณความยาวเสียง ----
        info = torchaudio.info(str(audio))
        duration_sec = info.num_frames / info.sample_rate

        if duration_sec > 1800:  # 30 นาที
            if duration_policy == "skip":
                raise ValueError("Audio length exceeds 30 minutes, skipping as requested.")
            duration_sec = 1800  # cut
        max_frames = int(duration_sec * fps)

        # ---- mapping resolution ----
        size_flag = "infinitetalk-480" if resolution == "480p" else "infinitetalk-720"

        # ---- เลือก checkpoint ----
        if persons == "multi":
            infinitetalk_ckpt = "weights/InfiniteTalk/multi/infinitetalk.safetensors"
        else:
            infinitetalk_ckpt = "weights/InfiniteTalk/single/infinitetalk.safetensors"

        # ---- output path ----
        output_base = "/tmp/output"
        output_path = output_base + ".mp4"

        # ---- เตรียม input_json ----
        input_json = "/tmp/input.json"
        with open(input_json, "w") as f:
            f.write(f"""{{
                "audio": "{str(audio)}",
                "{'video' if input_type=='video-to-video' else 'image'}": "{str(media)}"
            }}""")

        # ---- สั่งรัน InfiniteTalk ----
        cmd = [
            "python", "generate_infinitetalk.py",
            "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
            "--wav2vec_dir", "weights/chinese-wav2vec2-base",
            "--infinitetalk_dir", infinitetalk_ckpt,
            "--input_json", input_json,
            "--size", size_flag,
            "--sample_steps", "40",
            "--mode", mode,
            "--motion_frame", "9",
            "--max_frame_num", str(max_frames),
            "--save_file", output_base
        ]

        print("Running command:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        return Path(output_path)
