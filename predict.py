# predict.py
from cog import BasePredictor, Input, Path
import subprocess, torchaudio, os, mimetypes, json, math

WAN_CKPT_DIR = "weights/Wan2.1-I2V-14B-480P"
WAV2VEC_DIR  = "weights/chinese-wav2vec2-base"
IT_SINGLE    = "weights/InfiniteTalk/single/infinitetalk.safetensors"
IT_MULTI     = "weights/InfiniteTalk/multi/infinitetalk.safetensors"

CLIP_THRESHOLD_SEC = 60        # < 1 นาที → clip
MAX_DURATION_SEC   = 600       # จำกัด 10 นาที
DEFAULT_FPS        = 25

def is_video(path_str: str) -> bool:
    mt, _ = mimetypes.guess_type(str(path_str))
    return (mt or "").startswith("video")

def nearest_4n_plus_1_leq(x: int) -> int:
    # คืนค่าใกล้สุดที่เป็น 4n+1 และไม่เกิน x (อย่างน้อย 5 เฟรม)
    if x < 5: 
        return 5
    n = (x - 1) // 4
    return 4 * n + 1

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
            description="ความละเอียดวิดีโอ (ค่าเริ่มต้น 480p)",
            choices=["480p", "720p"],
            default="480p"
        ),
        fps: int = Input(description="Frames per second", default=DEFAULT_FPS, ge=1, le=60),
        audio_type: str = Input(
            description="(เฉพาะหลายคน) 'para'=พูดพร้อมกัน, 'add'=ต่อคิว",
            choices=["para", "add"],
            default="para"
        ),
        prompt: str = Input(description="ข้อความกำกับ (optional)", default=""),
    ) -> Path:

        # ---- คำนวณความยาวเสียง ----
        info = torchaudio.info(str(audio))
        duration_sec = info.num_frames / info.sample_rate
        if duration_sec > MAX_DURATION_SEC:
            print(f"[warn] Audio {duration_sec:.1f}s > {MAX_DURATION_SEC}s → ตัดที่ 10 นาที")
            duration_sec = MAX_DURATION_SEC

        # เฟรมตามเสียง
        max_frames = int(duration_sec * fps)

        # ---- โหมด gen อัตโนมัติ ----
        mode = "clip" if duration_sec < CLIP_THRESHOLD_SEC else "streaming"

        # ---- steps แบบเน้นบาลานซ์คุณภาพ/ความเร็ว ----
        if duration_sec < 60:
            steps = "20"
        elif duration_sec < 300:
            steps = "40"
        else:
            steps = "60"

        # ---- ความละเอียด ----
        size_flag = "infinitetalk-480" if resolution == "480p" else "infinitetalk-720"

        # ---- เลือก checkpoint ----
        ckpt = IT_MULTI if persons == "multi" else IT_SINGLE

        # ---- เตรียม input.json ตามสเปค generate_infinitetalk.py ----
        input_json = "/tmp/input.json"
        cond_audio = {"person1": str(audio)}  # single เป็นค่าเริ่มต้น

        # NOTE: ถ้าจะรองรับอัพโหลดเสียงคนที่ 2 ผ่าน Cog ให้เพิ่มพารามฯ อีกตัวแล้วเติม cond_audio["person2"] ตรงนี้
        input_payload = {
            "prompt": prompt,
            "cond_video": str(media),      # รับได้ทั้ง image หรือ video
            "cond_audio": cond_audio
        }
        if persons == "multi":
            input_payload["audio_type"] = audio_type  # "para" หรือ "add"

        with open(input_json, "w", encoding="utf-8") as f:
            json.dump(input_payload, f, ensure_ascii=False)

        # ---- output path ----
        output_base = "/tmp/output"     # ไม่ใส่ .mp4 ที่ argument
        output_path = output_base + ".mp4"

        # ---- อาร์กิวเมนต์เฉพาะโหมด ----
        clip_args = []
        streaming_args = []
        if mode == "clip":
            frame_num = nearest_4n_plus_1_leq(max_frames)
            clip_args = ["--frame_num", str(frame_num)]
        else:
            streaming_args = ["--max_frame_num", str(max_frames)]

        # ---- รัน InfiniteTalk พร้อม log error เต็ม ๆ ----
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
            # hint เช็คไฟล์/โฟลเดอร์หลัก ๆ
            print("\n[HINT] ตรวจสอบว่า weights/ ทั้งหมดมีจริงไหม:")
            print(" -", WAN_CKPT_DIR)
            print(" -", WAV2VEC_DIR)
            print(" -", ckpt)
            print("[HINT] และดูว่า /tmp/input.json มีคีย์ 'cond_video' และ 'cond_audio' ตามตัวอย่างหรือไม่")
            raise

        return Path(output_path)
