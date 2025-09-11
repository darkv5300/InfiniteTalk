import cog
from infinitetalk import InfiniteTalk # ต้องเช็ค module ที่ repo ใช้จริง

class Predictor(cog.BasePredictor):
    def setup(self):
        self.model = InfiniteTalk.load_default()

    def predict(self, audio: cog.Path = None, text: str = "Hello world") -> str:
        # ตรงนี้มึงต้องปรับตามฟังก์ชันจริง
        output_path = "output.mp4"
        self.model.generate(audio, text, output_path)
        return output_path
