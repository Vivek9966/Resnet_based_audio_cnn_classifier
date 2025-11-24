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
from tqdm import tqdm
from torch.optim.lr_scheduler  import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
app = modal.App("Audio_cnn_v2.2")


images = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .apt_install(["wget", "ffmpeg", "unzip", "libsndfile1"])
    .run_commands([
        "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
        "cd /tmp && unzip esc50.zip",
        "mkdir -p /opt/esc50-data",
        "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
        "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
    ])
    .add_local_python_source("model")
)

volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
modal_volume = modal.Volume.from_name("esc-model", create_if_missing=True)

class ESC50Dataset(Dataset):
    def __init__(self, data_dir ,metadeta_file , split="train" , transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadeta_file)
        self.split = split
        self.transform = transform
        
        if split == 'train':
            self.metadata = self.metadata[self.metadata['fold'] !=5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] ==5 ]
        self.classes = sorted(self.metadata['category'].unique())
        self.classes_to_idx = {cls:idx for idx,cls in enumerate(self.classes)}
        self.metadata["label"] = self.metadata["category"].map(self.classes_to_idx)
    def __len__(self): 
        return len(self.metadata)
    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        audio_path = self.data_dir / "audio" / row['filename']
        waveform,sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] >1:
            waveform = torch.mean(waveform,dim=0,keepdim=True)
        if self.transform:
            spectogram =self.transform(waveform)
        else:
            spectogram =waveform
        return spectogram , row['label']    
def mixup_data(x,y):
        lam = np.random.beta(.2,.2)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam *x +(1-lam)*x[index,:]
        y_a,y_b = y, y[index]
        return mixed_x ,y_a,y_b ,lam
    
def mixup_criterion(critetion, pred,y_a,y_b,lam):
        
        return lam * critetion(pred,y_a) + (1-lam)*critetion(pred,y_b)

@app.function(
    image=images,
    gpu="A10G",
    volumes={"/data": volume, "/models": modal_volume},
    timeout= 86400
)
def train():
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'/models/tensorboard_logs/run_{timestamp}'
    writer = SummaryWriter(log_dir,)

    print("training")
    esc50_dir = Path("/opt/esc50-data")
    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate= 22050*2,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_max=11025,
            f_min=0,
        ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)

    )
    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate= 22050*2,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_max=11025,
            f_min=0,
        ),
        T.AmplitudeToDB()
        #T.FrequencyMasking(freq_mask_param=30),
       # T.TimeMasking(time_mask_param=80)

    )



    train_dataset = ESC50Dataset(data_dir=esc50_dir, metadeta_file=esc50_dir / "meta" / "esc50.csv" ,split ="train",transform=train_transform)
    validation_dataset = ESC50Dataset(data_dir=esc50_dir, metadeta_file=esc50_dir / "meta" / "esc50.csv" ,split ="val",transform=val_transform)
    print(f"tranning samples: {len(train_dataset)}")
    print(f"validatiojn samples: {len(validation_dataset)}")
    train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    val_dataloader = DataLoader(validation_dataset,batch_size=32,shuffle=False)

    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model =ACNN(num_classes = len(train_dataset.classes))
    model.to(device)

    num_epoch = 170
    criterian = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),lr = 0.0005,weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=.002,
        epochs=num_epoch,
        steps_per_epoch = len(train_dataloader),
        pct_start=.1)
    best_acc =  0.0
   # print("starting trainnning")
    for epoch in range(num_epoch):
        model.train()
        epoch_loss = 0.0
        progress_bat = tqdm(train_dataloader,desc=f'Epoch {epoch+1}/{num_epoch}')
        for data,target in progress_bat:
            data,target = data.to(device) , target.to(device)

            if np.random.random() >0.7 :
                data,target_a ,target_b, lam =  mixup_data(data,target)
                output = model(data)
                loss = mixup_criterion(criterian,output,target_a,target_b,lam)
            else:
                 output = model(data)
                 loss = criterian(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss+=loss.item()
            progress_bat.set_postfix({"loss":f'{loss.item():.2f}'})
        avg_epoch_loss= epoch_loss /len(train_dataloader)
        writer.add_scalar("Loss/Train",avg_epoch_loss,epoch)
        writer.add_scalar("Learnig Rate",optimizer.param_groups[0]['lr'],epoch)
        model.eval()
        corr =0
        tot=0
        val_los = 0
        with torch.no_grad(): 
             for data,target in val_dataloader:
                  data ,target = data.to(device),target.to(device)
                  outputs = model(data)
                  loss = criterian(outputs,target)
                  val_los += loss.item()

                  _ ,predicted = torch.max(outputs.data , 1)
                  tot+= target.size(0)
                  corr += (predicted==target).sum().item()
                
        accuracy = 100*corr/tot
        avg_val_loss = val_los/len(val_dataloader)
        
        writer.add_scalar("Loss/Val",avg_val_loss,epoch)
        writer.add_scalar("accuracy/val",accuracy,epoch)

        print(f"EPOCH-> {epoch+1} LOSS: {avg_epoch_loss:.2f} VAL_LOSS ->{avg_val_loss:.2f}   Accuracy-> {accuracy:.2f}")

        if accuracy>best_acc:
             best_acc=accuracy
             torch.save(
                  {
                       'model_state_dict':model.state_dict(),
                       "accuracy" : accuracy,
                       "epoch":epoch,
                       "classes":train_dataset.classes

                  }, '/models/best_model.pth'
             )
             print(f'New Best Model Saved Acc -> {accuracy:.2f}')
    print(f"Trainning Complete best accuracy -> {best_acc}")
    writer.close()

             

@app.local_entrypoint()
def main():
    train.remote()
