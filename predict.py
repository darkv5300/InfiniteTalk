from cog import BasePredictor, Input, Path
import torch
import cv2
import librosa
import numpy as np
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
import tempfile
import os
from PIL import Image

class Predictor(BasePredictor):
    def setup(self):
        """โหลด models"""
        # โหลด InfiniteTalk models เดิมที่มี
        pass
    
    def predict(
        self,
        input_media: Path = Input(description="รูปภาพหรือวิดีโอ (.jpg, .png, .mp4)"),
        audio_file: Path = Input(description="ไฟล์เสียง (สูงสุด 10 นาที)"),
        second_input: Path = Input(description="รูปภาพ/วิดีโอที่สอง (สำหรับโหมด 2 คน)", default=None),
        speaker_mode: str = Input(choices=["single", "dual"], default="single", description="โหมดผู้พูด"),
        dual_layout: str = Input(choices=["left_to_right", "right_to_left", "simultaneous"], default="left_to_right", description="การจัดเรียงสำหรับ 2 คน"),
        video_quality: str = Input(choices=["480p", "720p"], default="720p", description="คุณภาพวิดีโอ"),
        optimization: str = Input(choices=["good", "auto"], default="auto", description="โหมดออปติไมเซชัน")
    ) -> Path:
        
        # ตรวจสอบเสียงไม่เกิน 10 นาที
        audio_duration = self.validate_audio(audio_file)
        
        # ตรวจสอบประเภทไฟล์
        input_type = self.get_media_type(input_media)
        
        # กำหนด resolution
        width, height = self.get_resolution(video_quality)
        
        if speaker_mode == "single":
            output_video = self.generate_single_video(input_media, audio_file, input_type, width, height, optimization)
        else:
            if second_input is None:
                raise ValueError("ต้องใส่รูป/วิดีโอที่สองสำหรับโหมด dual")
            output_video = self.generate_dual_video(input_media, second_input, audio_file, dual_layout, width, height, optimization)
        
        return Path(output_video)
    
    def validate_audio(self, audio_file):
        """ตัดเสียงให้ไม่เกิน 10 นาที"""
        audio, sr = librosa.load(str(audio_file))
        duration = len(audio) / sr
        
        if duration > 600:  # 10 minutes
            audio = audio[:600 * sr]
            temp_audio = tempfile.mktemp(suffix='.wav')
            sf.write(temp_audio, audio, sr)  # ใช้ soundfile แทน librosa.output
            return temp_audio
        
        return str(audio_file)
    
    def get_media_type(self, media_path):
        """ตรวจสอบประเภทไฟล์"""
        ext = str(media_path).lower().split('.')[-1]
        if ext in ['jpg', 'jpeg', 'png']:
            return 'image'
        elif ext in ['mp4', 'mov', 'avi']:
            return 'video'
        else:
            raise ValueError(f"ไม่รองรับไฟล์ .{ext}")
    
    def get_resolution(self, quality):
        """กำหนด resolution"""
        if quality == "480p":
            return 854, 480
        else:  # 720p
            return 1280, 720
    
    def generate_single_video(self, input_media, audio_file, input_type, width, height, optimization):
        """สร้างวิดีโอคนเดียว"""
        output_path = tempfile.mktemp(suffix='.mp4')
        
        # TODO: ใส่โค้ด InfiniteTalk model ตรงนี้
        # ตอนนี้แค่ mock ตัวรวมเสียง
        video_clip = VideoFileClip("path_to_generated_video")
        audio_clip = AudioFileClip(audio_file)
        
        final_video = video_clip.set_audio(audio_clip)
        final_video.write_videofile(output_path, 
                                  codec='libx264', 
                                  preset='medium' if optimization == 'auto' else 'slow')
        
        return output_path
    
    def generate_dual_video(self, input1, input2, audio_file, layout, width, height, optimization):
        """สร้างวิดีโอสองคน"""
        
        # แยกเสียงสำหรับสองคน (แบบง่าย: แบ่งครึ่ง)
        audio, sr = librosa.load(str(audio_file))
        mid_point = len(audio) // 2
        
        audio1 = audio[:mid_point]
        audio2 = audio[mid_point:]
        
        # บันทึกเสียงแยก
        temp_audio1 = tempfile.mktemp(suffix='.wav')
        temp_audio2 = tempfile.mktemp(suffix='.wav')
        sf.write(temp_audio1, audio1, sr)
        sf.write(temp_audio2, audio2, sr)
        
        # สร้างวิดีโอแต่ละคน
        video1 = self.generate_single_video(input1, temp_audio1, self.get_media_type(input1), width//2, height, optimization)
        video2 = self.generate_single_video(input2, temp_audio2, self.get_media_type(input2), width//2, height, optimization)
        
        # รวมวิดีโอตาม layout
        output_path = tempfile.mktemp(suffix='.mp4')
        
        clip1 = VideoFileClip(video1)
        clip2 = VideoFileClip(video2)
        
        if layout == "simultaneous":
            final_video = CompositeVideoClip([
                clip1.set_position(('left')),
                clip2.set_position(('right'))
            ], size=(width, height))
        elif layout == "left_to_right":
            final_video = CompositeVideoClip([
                clip1.set_position(('left')).set_duration(clip1.duration),
                clip2.set_position(('right')).set_start(clip1.duration)
            ], size=(width, height))
        else:  # right_to_left
            final_video = CompositeVideoClip([
                clip2.set_position(('right')).set_duration(clip2.duration),
                clip1.set_position(('left')).set_start(clip2.duration)
            ], size=(width, height))
        
        final_video.write_videofile(output_path, codec='libx264')
        
        return output_path
