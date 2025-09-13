from cog import BasePredictor, Input, Path
import tempfile
import subprocess
import json
import os
import sys

class Predictor(BasePredictor):
    def setup(self):
        """โหลดโมเดล (ตรวจสอบ weights และ lib)"""
        self.ckpt_dir = "weights/Wan2.1-I2V-14B-480P"
        self.wav2vec_dir = "weights/chinese-wav2vec2-base"
        self.infinitetalk_dir = "weights/InfiniteTalk/single/infinitetalk.safetensors"
        
        # ตรวจสอบ weights
        for path in [self.ckpt_dir, self.wav2vec_dir, self.infinitetalk_dir]:
            if not os.path.exists(path):
                raise RuntimeError(f"❌ Weight not found: {path}. Download from Hugging Face first.")
        
        # ตรวจสอบ xfuser
        try:
            import xfuser
            print(f"✅ xfuser loaded (v{getattr(xfuser, '__version__', 'unknown')})")
        except ImportError:
            raise RuntimeError("❌ xfuser not found. Check cog.yaml and build.")
        
        # ตรวจสอบ xformers และ flash_attn (optional)
        try:
            import xformers
            print("✅ xformers loaded")
        except ImportError:
            print("⚠️ xformers not found, may run slower")
        try:
            import flash_attn
            print("✅ flash_attn loaded")
        except ImportError:
            print("⚠️ flash_attn not found, continuing")

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
            description="โหมดการสร้างวิดีโอ"
        ),
        sample_steps: int = Input(
            description="จำนวน sampling steps", default=40
        ),
    ) -> Path:
        """สร้างวิดีโอจาก InfiniteTalk"""
        # ตรวจสอบ input files
        for f in [input_media, audio_file]:
            if not os.path.exists(f):
                raise RuntimeError(f"❌ Input file not found: {f}")
        
        # สร้าง input.json
        input_json = tempfile.mktemp(suffix=".json")
        with open(input_json, "w", encoding="utf-8") as f:
            json.dump(
                {"prompt": "", "cond_video": str(input_media), "cond_audio": str(audio_file)},
                f,
                ensure_ascii=False,
            )
        print(f"📝 Input JSON: {input_json}")
        
        # เลือก resolution และ low VRAM
        size = "infinitetalk-720" if video_quality == "720p" else "infinitetalk-480"
        extra_params = ["--num_persistent_param_in_dit", "0"] if video_quality == "480p" else []
        
        # สร้าง output path
        output_path = tempfile.mktemp(suffix=".mp4")
        
        # สร้างคำสั่งรัน
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
            "--dit_fsdp", "--t5_fsdp", "--use_teacache",  # optimize
        ] + extra_params
        
        print(f"🚀 Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd="/src")
            print(f"✅ Success: {result.stdout}")
            if result.stderr:
                print(f"⚠️ Warning: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e.stdout}\n{e.stderr}")
            raise RuntimeError(f"❌ Failed: {e}")
        
        if not os.path.exists(output_path):
            raise RuntimeError(f"❌ No output at {output_path}")
        
        print(f"✅ Video at: {output_path}")
        return Path(output_path)
