# predict.py
from cog import BasePredictor, Input, Path
import subprocess, torchaudio, os, mimetypes, json

# ==== Weights ====
WAN_CKPT_DIR = "weights/Wan2.1-I2V-14B-480P"
WAV2VEC_DIR  = "weights/chinese-wav2vec2-base"
IT_SINGLE    = "weights/InfiniteTalk/single/infinitetalk.safetensors"
IT_MULTI     = "weights/InfiniteTalk/multi/infinitetalk.safetensors"

# ==== Policy ====
CLIP_THRESHOLD_SEC = 60    # < 1 นาที → ใช้ clip
MAX_DURATION_SEC   = 600   # จำกัด 10 นาที
DEFAULT_FPS        = 25

def is_video(path_str: str) -> bool:
    mt, _ = mimetypes.guess_type(path_str)
    return (mt or "").startswith("video")

class Predictor(BasePredictor):
    def setup(self):
        print("[setup] InfiniteTalk predictor ready")

    def predict(
        self,
        audio: Path = Input(description="ไฟล์เสียง (mp3/wav)"),
        media: Path = Input(description="รูปภาพหรือวิดีโอ"),
        persons: str = Input(
            description="เลือกโหมด คนเดียว หรือ หลายคน",
            choices=["single", "multi"],
            default="single"
        ),
        resolution: str = Input(
            description="ความละเอียดวิดีโอ",
            choices=["480p", "720p"],
            default="480p"  # default = 480p
        ),
        fps: int = Input(description="Frames per second", default=DEFAULT_FPS, ge=1, le=60),
    ) -> Path:

        # ---- ความยาวเสียง ----
        info = torchaudio.info(str(audio))
        duration_sec = info.num_frames / info.sample_rate

        if duration_sec > MAX_DURATION_SEC:
            print(f"[warn] Audio {duration_sec:.1f}s เกิน {MAX_DURATION_SEC}s → จะตัดที่ 10 นาที")
            duration_sec = MAX_DURATION_SEC

        max_frames = int(duration_sec * fps)

        # ---- auto mode ----
        mode = "clip" if duration_sec < CLIP_THRESHOLD_SEC else "streaming"

        # ---- dynamic steps ----
        if duration_sec < 60:       # สั้น < 1 นาที
            steps = "20"
        elif duration_sec < 300:    # กลาง 1–5 นาที
            steps = "40"
        else:                       # ยาว 5–10 นาที
            steps = "60"

        # ---- resolution ----
        size_flag = "infinitetalk-720" if resolution == "720p" else "infinitetalk-480"

        # ---- single vs multi ----
        ckpt = IT_MULTI if persons == "multi" else IT_SINGLE

        # ---- input type ----
        input_key = "video" if is_video(str(media)) else "image"

        # ---- input.json ----
        input_json = "/tmp/input.json"
        with open(input_json, "w") as f:
            f.write(json.dumps({
                "audio": str(audio),
                input_key: str(media)
            }))

        output_base = "/tmp/output"
        output_path = output_base + ".mp4"

        # ---- run InfiniteTalk ----
        cmd = [
            "python", "generate_infinitetalk.py",
            "--ckpt_dir", WAN_CKPT_DIR,
            "--wav2vec_dir", WAV2VEC_DIR,
            "--infinitetalk_dir", ckpt,
            "--input_json", input_json,
            "--size", size_flag,
            "--sample_steps", steps,
            "--mode", mode,
            "--motion_frame", "9",
            "--sample_audio_guide_scale", "4",
            "--sample_text_guide_scale", "5",
            "--max_frame_num", str(max_frames),
            "--save_file", output_base
        ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        return Path(output_path)
