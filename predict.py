# predict.py
from cog import BasePredictor, Input, Path
import subprocess, torchaudio, mimetypes, json

# ==== Weights ====
WAN_CKPT_DIR = "weights/Wan2.1-I2V-14B-480P"
WAV2VEC_DIR  = "weights/chinese-wav2vec2-base"
IT_SINGLE    = "weights/InfiniteTalk/single/infinitetalk.safetensors"
IT_MULTI     = "weights/InfiniteTalk/multi/infinitetalk.safetensors"

# ==== Policy ====
CLIP_THRESHOLD_SEC = 60        # < 1 นาที → clip
MAX_DURATION_SEC   = 600       # จำกัด 10 นาที
DEFAULT_FPS        = 25

def is_video(path_str: str) -> bool:
    mt, _ = mimetypes.guess_type(str(path_str))
    return (mt or "").startswith("video")

def nearest_4n_plus_1_leq(x: int) -> int:
    if x < 5:
        return 5
    n = (x - 1) // 4
    return 4 * n + 1

class Predictor(BasePredictor):
    def setup(self):
        print("[setup] InfiniteTalk predictor ready")

    def predict(
        self,
        audio1: Path = Input(description="ไฟล์เสียงคนที่ 1 (mp3/wav)"),
        audio2: Path = Input(description="ไฟล์เสียงคนที่ 2 (เฉพาะโหมด multi)", default=None),
        media: Path = Input(description="รูปภาพหรือวิดีโอ"),
        persons: str = Input(
            description="เลือกโหมด คนเดียว หรือ หลายคน",
            choices=["single", "multi"],
            default="single"
        ),
        resolution: str = Input(
            description="ความละเอียดวิดีโอ (ค่าเริ่มต้น 480p)",
            choices=["480p", "720p"],
            default="480p"
        ),
        fps: int = Input(description="Frames per second", default=DEFAULT_FPS, ge=1, le=60),
        multi_mode: str = Input(
            description="(เฉพาะ multi) เลือกลำดับการพูด",
            choices=["left-right", "right-left", "together"],
            default="together"
        ),
        prompt: str = Input(description="ข้อความกำกับ (optional)", default=""),
    ) -> Path:

        # ---- ความยาวเสียง ----
        info = torchaudio.info(str(audio1))
        duration_sec = info.num_frames / info.sample_rate
        if duration_sec > MAX_DURATION_SEC:
            print(f"[warn] Audio {duration_sec:.1f}s > {MAX_DURATION_SEC}s → จะตัดที่ 10 นาที")
            duration_sec = MAX_DURATION_SEC

        max_frames = int(duration_sec * fps)

        # ---- auto mode ----
        mode = "clip" if duration_sec < CLIP_THRESHOLD_SEC else "streaming"

        # ---- dynamic steps ----
        if duration_sec < 60:
            steps = "20"
        elif duration_sec < 300:
            steps = "40"
        else:
            steps = "60"

        # ---- resolution ----
        size_flag = "infinitetalk-480" if resolution == "480p" else "infinitetalk-720"

        # ---- checkpoint ----
        ckpt = IT_MULTI if persons == "multi" else IT_SINGLE

        # ---- สร้าง input.json ----
        cond_audio = {"person1": str(audio1)}
        input_payload = {
            "prompt": prompt,
            "cond_video": str(media),
            "cond_audio": cond_audio
        }

        if persons == "multi":
            if audio2 is None:
                raise ValueError("ต้องใส่ไฟล์เสียงของคนที่ 2 ด้วยเมื่อเลือกโหมด multi")

            if multi_mode == "together":
                input_payload["audio_type"] = "para"
                cond_audio["person2"] = str(audio2)

            elif multi_mode == "left-right":
                input_payload["audio_type"] = "add"
                cond_audio["person1"] = str(audio1)
                cond_audio["person2"] = str(audio2)

            elif multi_mode == "right-left":
                input_payload["audio_type"] = "add"
                cond_audio["person1"] = str(audio2)
                cond_audio["person2"] = str(audio1)

        input_json = "/tmp/input.json"
        with open(input_json, "w", encoding="utf-8") as f:
            json.dump(input_payload, f, ensure_ascii=False)

        # ---- output ----
        output_base = "/tmp/output"
        output_path = output_base + ".mp4"

        # ---- args สำหรับ clip / streaming ----
        clip_args, streaming_args = [], []
        if mode == "clip":
            frame_num = nearest_4n_plus_1_leq(max_frames)
            clip_args = ["--frame_num", str(frame_num)]
        else:
            streaming_args = ["--max_frame_num", str(max_frames)]

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
            "--save_file", output_base,
        ] + clip_args + streaming_args

        print("Running:", " ".join(cmd))
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("[ERROR] generate_infinitetalk.py failed")
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)
            raise

        return Path(output_path)
