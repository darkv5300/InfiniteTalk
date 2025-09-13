from cog import BasePredictor, Input, Path
import tempfile
import subprocess
import json
import os
import sys

class Predictor(BasePredictor):
    def setup(self):
        """โหลดโมเดล (เตรียม path + เช็ค lib เสริม + ติดตั้ง dependencies)"""
        self.ckpt_dir = "weights/Wan2.1-I2V-14B-480P"
        self.wav2vec_dir = "weights/chinese-wav2vec2-base"
        self.infinitetalk_dir = "weights/InfiniteTalk/single/infinitetalk.safetensors"
        
        # ติดตั้ง dependencies สำหรับ InfiniteTalk/Wan2.1 (จาก repo GitHub)
        # xfuser เป็นหลัก สำหรับ distributed และ DiT inference
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "xfuser>=0.4.1"])
            print("✅ xfuser installed and ready")
        except Exception as e:
            print(f"⚠️ Failed to install xfuser: {e}. Continuing without it may cause errors.")
        
        # ติดตั้ง dependencies อื่นๆ ที่จำเป็น (จาก InfiniteTalk setup)
        required_packages = [
            "torch==2.4.1",  # หรือเวอร์ชันที่ตรงกับ environment ของท่าน (cu121 สำหรับ GPU)
            "torchvision==0.19.1",
            "torchaudio==2.4.1",
            "--index-url", "https://download.pytorch.org/whl/cu121",  # ถ้าใช้ CUDA 12.1
            "xformers==0.0.28",  # สำหรับ attention optimization
            "--index-url", "https://download.pytorch.org/whl/cu121",
            "transformers",  # สำหรับ Wav2Vec2
            "librosa",
            "pyloudnorm",
            "einops",
            "soundfile",
            "modelscope",  # สำหรับดาวน์โหลดโมเดลจาก Hugging Face
        ]
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + required_packages)
            print("✅ All dependencies installed")
        except Exception as e:
            print(f"⚠️ Dependency installation warning: {e}")
        
        # optional speed-up libs (หลังติดตั้ง xformers แล้ว)
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
        
        # เช็ค xfuser โดยตรง
        try:
            import xfuser
            print(f"✅ xfuser version {xfuser.__version__} loaded successfully")
        except ImportError:
            raise RuntimeError("❌ xfuser installation failed. Cannot proceed with InfiniteTalk.")

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
        # เตรียม input.json สำหรับโมเดล (ปรับให้รองรับ prompt ถ้าต้องการ)
        input_json = tempfile.mktemp(suffix=".json")
        with open(input_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prompt": "",  # สามารถเพิ่ม prompt ได้ถ้าต้องการ text guidance
                    "cond_video": str(input_media),
                    "cond_audio": str(audio_file),
                },
                f,
                ensure_ascii=False,
            )
        # เลือก resolution (ปรับให้ตรงกับ Wan2.1 sizes)
        size = "infinitetalk-720" if video_quality == "720p" else "infinitetalk-480"
        # output path
        output_path = tempfile.mktemp(suffix=".mp4")
        # run InfiniteTalk script (เพิ่ม error handling เพิ่มเติม)
        cmd = [
            "python", "generate_infinitetalk.py",
            "--ckpt_dir", self.ckpt_dir,
            "--wav2vec_dir", self.wav2vec_dir,
            "--infinitetalk_dir", self.infinitetalk_dir,
            "--input_json", input_json,
            "--size", size,
            "--sample_steps", str(sample_steps),
            "--mode", mode,
            "--motion_frame", "9",  # สำหรับ motion consistency ใน talking head
            "--save_file", output_path,
            # เพิ่ม params สำหรับ low VRAM ถ้าจำเป็น (optional)
            # "--num_persistent_param_in_dit", "0",  # ถ้า VRAM น้อย
        ]
        print(f"🚀 Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Command succeeded: {result.stdout}")
            if result.stderr:
                print(f"⚠️ Warnings: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error details: {e.stderr}")
            raise RuntimeError(f"❌ Model execution failed: {e}. Check if xfuser is properly loaded.")
        
        # เช็คว่า output file มีจริง
        if not os.path.exists(output_path):
            raise RuntimeError("❌ Output video not generated. Check logs.")
        
        return Path(output_path)
