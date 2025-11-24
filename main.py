import modal
from model import ACNN,ResBk
import sys
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import torchaudio
import torch
import torch.nn as nn
import numpy as np
import torchaudio.transforms as T
import torch.optim as optim
from pydantic import BaseModel,Base64Str,Base64Bytes
from tqdm import tqdm
from torch.optim.lr_scheduler  import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import base64
import soundfile as sf
import io
import librosa
import requests

app  = modal.App("audio-cnn-inference")
image = (modal.Image.debian_slim()
         .apt_install(['libsndfile1'])
         .pip_install_from_requirements("requirements.txt")
         .add_local_python_source("model"))
modal_volume = modal.Volume.from_name("esc-model")

class Audio_processor():
    def __init__(self):
        self.transform = nn.Sequential(
            T.MelSpectrogram(
            sample_rate= 22050*2,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_max=11025,
            f_min=0,
        ))
    def process_audio_cnk(self,data):
        waveform = torch.from_numpy(data).float()
        waveform=waveform.unsqueeze(0)
        spectogram = self.transform(waveform)
        return spectogram.unsqueeze(0)

class Inference_req(BaseModel):
    audio_data :str


@app.cls(image=image,
    gpu="A10G",
    volumes={ "/models": modal_volume},
    timeout= 86400,
    scaledown_window = 15)
class Audio_classifier():
    @modal.enter()
    def load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load('/models/best_model.pth',
                          map_location=self.device)
        self.classes = ckpt['classes']
        self.model = ACNN(num_classes=len(self.classes))
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.ap = Audio_processor() 
        print("MOdel_loaded")
    @modal.fastapi_endpoint(method ="POST")    
    def inference(self, request: Inference_req):
        audio_bit = base64.b64decode(request.audio_data)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bit), dtype='float32')
        
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        if sample_rate != 22050*2:
            audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=22050*2)  
        
        spectogram = self.ap.process_audio_cnk(audio_data)
        spectogram = spectogram.to(self.device)

        with torch.no_grad():
            output = self.model(spectogram)
            output = torch.nan_to_num(output)
            probab = torch.softmax(output, dim=1)
            top_prob, top_index = torch.topk(probab[0], 3)

            predict = [{"class": self.classes[idx.item()], "confidence": pro.item()} for pro, idx in zip(top_prob, top_index)]
        
        response = {"predictions": predict}    
        return response  
@app.local_entrypoint()
def main():
    audio_data,_= sf.read('1-1791-A-26.wav')
    buffer=io.BytesIO()
    sf.write(buffer,audio_data,22050,format="WAV")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    payload ={"audio_data":audio_b64}
   
    server = Audio_classifier()
    url = server.inference.get_web_url()
    response = requests.post(url,json =payload)
    response.raise_for_status()
    result =response.json()
    for pred in result.get("predictions"):
        print(pred['class'] , pred['confidence'])

