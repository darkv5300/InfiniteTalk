from cog import BasePredictor, Input, Path
import tempfile
import subprocess
import json
import os

class Predictor(BasePredictor):
    def setup(self):
        """โหลดโมเดล (เตรียม path + เช็ค lib เสริม)"""
        self.ckpt_dir = "weights/Wan2.1-I2V-14B-480P"
        self.wav2vec_dir = "weights/chinese-wav2vec2-base"
        self.infinitetalk_dir = "weights/InfiniteTalk/single/infinitetalk.safetensors"

        # optional speed-up libs
        try:
            import xformers  # noqa
            print("✅ xformers loaded")
        except ImportError:
            print("⚠️ xformers not found, running slower")

        try:
            import flash_attn  # noqa
            print("✅ flash-attn loaded")
        except ImportError:
            print("⚠️ flash-attn not found, continuing without it")

    def predict(
        self,
        input_media: Path = Input(description="รูปภาพหรือวิดีโอ (.jpg, .png, .mp4)"),
        audio_file: Path = Input(description="ไฟล์เสียง (สูงสุด 10 นาที)"),
        video_quality: str = Input(
            choices=["480p", "720p"], default="720p", description="คุณภาพวิดีโอ"
        ),
        mode: str = Input(
            choices=["streaming", "clip"],
            default="streaming",
            description="โหมดการสร้างวิดีโอ",
        ),
        sample_steps: int = Input(
            description="จำนวน sampling steps", default=40
        ),
    ) -> Path:
        """สร้างวิดีโอจาก InfiniteTalk"""

        # เตรียม input.json สำหรับโมเดล
        input_json = tempfile.mktemp(suffix=".json")
        with open(input_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prompt": "",
                    "cond_video": str(input_media),
                    "cond_audio": str(audio_file),
                },
                f,
                ensure_ascii=False,
            )

        # เลือก resolution
        size = "infinitetalk-720" if video_quality == "720p" else "infinitetalk-480"

        # output path
        output_path = tempfile.mktemp(suffix=".mp4")

        # run InfiniteTalk script
        cmd = [
            "python", "generate_infinitetalk.py",
            "--ckpt_dir", self.ckpt_dir,
            "--wav2vec_dir", self.wav2vec_dir,
            "--infinitetalk_dir", self.infinitetalk_dir,
            "--input_json", input_json,
            "--size", size,
            "--sample_steps", str(sample_steps),
            "--mode", mode,
            "--motion_frame", "9",
            "--save_file", output_path,
        ]

        print(f"🚀 Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"❌ Model execution failed: {e}")

        return Path(output_path)
