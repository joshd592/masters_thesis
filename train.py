#%% NN Imports and definitions
import matplotlib
matplotlib.use('tkAgg')
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import h5py
import matplotlib.pyplot as plt
import time
from IPython.display import display
import h5py
import itertools
torch.manual_seed(time.time())
import sys
from globals import printt

class ScenarioDataset(Dataset):
    def __init__(self, hdf5_path, sequence_length=1, local=False):
        self.hdf5_path = hdf5_path
        self.sequence_length = sequence_length
        self.local = local
        self.sequences = []
        self.start_steps = []
        self.cutoffs = {}
        
        with h5py.File(hdf5_path, 'r') as f:
            for key in sorted(f.keys()):
                if key.startswith('scenario_'):
                    pos_ric = torch.FloatTensor(f[key]['chaser_pos_ric'][:])
                    norms = torch.norm(pos_ric, dim=1)
                    diffs = norms[1:] - norms[:-1]
                    cutoff = (diffs > 0).nonzero()
                    cutoff = cutoff[0].item() + 6 if len(cutoff) > 0 else len(norms)
                    self.cutoffs[key] = cutoff
                    num_steps = len(f[key]['chaser_pos'])
                    max_start = num_steps - sequence_length
                    for start_idx in range(min(cutoff, max_start)):
                        self.sequences.append(key)
                        self.start_steps.append(start_idx)
        
        state_type = "local (7D)" if local else "absolute (12D)"
        print(f"Dataset: {len(self)} samples, sequence_length={sequence_length}, state={state_type}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        scenario_key = self.sequences[idx]
        start_step = self.start_steps[idx]
        end_step = start_step + self.sequence_length
        
        with h5py.File(self.hdf5_path, 'r') as f:
            group = f[scenario_key]
            
            
            tback = torch.FloatTensor(group['tback'][start_step:end_step])
            target_pos = torch.FloatTensor(group['target_pos'][start_step:end_step])
            
            if self.local:
                # Relative coordinates: (seq_len, 3)
                chaser_eci_pos = torch.FloatTensor(group['chaser_pos_ric'][start_step:end_step])
                chaser_eci_vel = torch.FloatTensor(group['chaser_vel_ric'][start_step:end_step])
                
                # Magnitude of target position: (seq_len, 1)
                target_mag = torch.norm(target_pos, dim=1, keepdim=True)
                
                # Stack: (seq_len, 7)
                state = torch.cat([chaser_eci_pos, chaser_eci_vel, target_mag], dim=1)
            else:
                # Extract sequence
                chaser_pos = torch.FloatTensor(group['chaser_pos'][start_step:end_step])
                chaser_vel = torch.FloatTensor(group['chaser_vel'][start_step:end_step])
                target_vel = torch.FloatTensor(group['target_vel'][start_step:end_step])
                # Absolute coordinates: (seq_len, 12)
                state = torch.cat([chaser_pos, target_pos, chaser_vel, target_vel], dim=1)
            
            # Squeeze if sequence length is 1
            if self.sequence_length == 1:
                state = state.squeeze(0)  # (7,) or (12,)
                tback = tback.squeeze(0)  # (1,)
        
        return state, tback
class FromSequenceModel(nn.Module):
    #needs updates with normalizaitons
    def __init__(self, input_size=12, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, steps, 12)
        lstm_out, _ = self.lstm(x)  # (batch, steps, hidden_size)
        output = self.fc(lstm_out)   # (batch, steps, 1)
        return output  
class FromStateModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_hidden_layers=3):
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()
        
        layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        return self.sig(self.output_layer(x))
class NotTooLowLoss(nn.Module):
    def __init__(self, nu = 0):
        super().__init__()
        self.nu = nu
        self.relu = nn.ReLU()
    
    def forward(self, predictions, targets):
        # Your loss computation here
        error = predictions - targets
        loss = error**2
        reluloss = self.relu(-error)**2
        return (loss.sum() + (self.nu * reluloss.sum())) / loss.numel()

blnPlot = False
if blnPlot:
    plt.ioff()
    fig, ax = plt.subplots()
    train_losses, val_losses = [], []  
    ax.set_ylabel('Loss')
    plt.show(block=False)
    
#Hyperparameters
seqLength = 1 #Anything over 1 switches models
localState = True
val_size = 1500
batch_size = 2 ** 9
# hidden_size = 2 ** 11
# num_hidden_layers = 8
learning_rate = .0005
tooLowPenalty = .01
withSched = True
nepochs = 5

hidden_sizes = [2**p for p in range(8,12)]
num_hidden_layerss = range(3,8)

datapath = "gridsweep.h5"
print("Saving log as: ", time.strftime('locallogs/sweep_log_%b%d_%H%M.txt'))
with open(time.strftime('locallogs/sweep_log_%b%d_%H%M.txt'), "w") as f:
    for (hidden_size, num_hidden_layers) in itertools.product(hidden_sizes, num_hidden_layerss):
        printt(hidden_size, num_hidden_layers, withSched, file=f)
        # Training
        dims = 7 if localState else 12
            
        dataset = ScenarioDataset(datapath,sequence_length=seqLength,local=localState)
        total_size = len(dataset)
        train_size = int(0.6 * total_size)
        #val_size = int(val_perc * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        normFromFile = torch.load('normsfor'+datapath[:datapath.find(".")]+'.pth', weights_only=True)
        tback_min = normFromFile['tback_min']
        tback_max = normFromFile['tback_max']
        state_min = normFromFile['state_min_loc'] if localState else normFromFile['state_min_abs']
        state_max = normFromFile['state_max_loc'] if localState else normFromFile['state_max_abs']
        state_range = torch.tensor([ma - mi if torch.abs(ma - mi) > 1e-3 else np.inf
                                     for ma, mi in zip(state_max, state_min)])
        
        
        if seqLength == 1:
            model = FromStateModel(input_size=dims, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
        else:
            model = FromSequenceModel(input_size=dims, hidden_size=hidden_size, num_layers=seqLength)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = NotTooLowLoss(tooLowPenalty)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)
        
        min_val_loss=np.inf
        update = False
        model_path = time.strftime('localsaves/model_%b%d_%H%M.pth')
        printt(f"Model Saving as: {model_path}", file=f)
        printt(f"{len(train_loader)} Batches Per Epoch; {train_loader.batch_size} Samples per Batch", file=f)
        try:
            for epoch in range(nepochs):
                # Training
                train_loss = 0
                for batchidx, (state, tback) in enumerate(train_loader):
                    state = (state - state_min) / state_range
                    optimizer.zero_grad()
                    outputs = model(state)
                    tback_norm = (tback - tback_min) / (tback_max - tback_min)
                    loss = criterion(outputs, tback_norm)
                    loss.backward()
                    optimizer.step()
                    train_loss = (batchidx/(batchidx+1))*train_loss + (loss.item() / (batchidx+1))
                    
                     # Validate every N batches
                    if (batchidx + 1) % 5 == 0:
                        val_loss = 0
                        with torch.no_grad():
                            for state_val, tback_val in val_loader:
                                state_val = (state_val - state_min) / state_range
                                outputs_val = model(state_val)
                                tback_val_norm = (tback_val - tback_min) / (tback_max - tback_min)
                                loss_val = criterion(outputs_val, tback_val_norm)
                                val_loss += loss_val.item()
                        val_loss /= len(val_loader)
                        scheduler.step(val_loss)
                        if val_loss < min_val_loss:
                            min_val_loss = val_loss
                            torch.save({'model': model.state_dict(), 'tback_min': tback_min, 'tback_max': tback_max}, model_path)
                            print(f"Epoch {epoch+1}, Batch {batchidx+1} | sqrtTrain: {np.sqrt(loss.item()):.4f}, sqrtVal: {np.sqrt(val_loss):.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} saved")
                        else:
                            print(f"Epoch {epoch+1}, Batch {batchidx+1} | sqrtTrain: {np.sqrt(loss.item()):.4f}, sqrtVal: {np.sqrt(val_loss):.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
                        if blnPlot:
                            train_losses.append(train_loss)
                            val_losses.append(val_loss)
                            ax.clear()
                            ax.plot(range(len(train_losses)), train_losses, label='Train', alpha=0.7)
                            ax.plot(range(len(train_losses)), val_losses, label='Val', alpha=0.7)
                            ax.legend()
                            ax.set_ylabel('Loss')
                            fig.show()
                            #plt.pause(.05)
                            fig.canvas.flush_events()
        except:
            if blnPlot:
                plt.close()    
                plt.pause(.1)    
                plt.switch_backend('module://matplotlib_inline.backend_inline')
                display(fig)
            print(f"Model Saved as: {model_path}")
            sys.exit()
        printt("Lowest sqrt of Val Loss: ", np.sqrt(min_val_loss), file=f)
        # Test on holdout set
        printt("Testing...", file=f)
        #model.load_state_dict(torch.load(model_path))
        model.eval()
        
        test_loss = 0
        all_preds = []
        all_actuals = []
        
        with torch.no_grad():
            for state, tback in test_loader:
                state = (state - state_min) / state_range
                tback_norm = (tback - tback_min) / (tback_max - tback_min)
                outputs = model(state)
                loss = criterion(outputs, tback_norm)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        printt(f"Test Loss: {np.sqrt(test_loss):.4f}", file=f)
        printt(f"Model Saved as: {model_path}\n", file=f)
        f.flush()

#%%
import time

checkpoint = torch.load('localsaves/model_Jun04_2247.pth', weights_only=False)
state_dict = checkpoint['model']

# Infer architecture from weights
input_layer_key = next(k for k in state_dict if 'weight' in k)
dims = state_dict[input_layer_key].shape[1]
localState = True if dims == 7 else False
hidden_size = state_dict[input_layer_key].shape[0]
num_layers = sum([1 for k in state_dict if k.endswith('.weight') and 'out' not in k])

bestmodel = FromStateModel(input_size=dims, hidden_size=hidden_size, num_hidden_layers=num_layers)
bestmodel.load_state_dict(state_dict)
bestmodel.eval()

# checkpoint = torch.load('model_Jun04_2334.pth', weights_only=False)
# bestmodel = FromStateModel(input_size=dims, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
# bestmodel.load_state_dict(checkpoint['model'])
# bestmodel.eval()
normFromFile = torch.load('norm.pth', weights_only=True)
tback_min = normFromFile['tback_min']
tback_max = normFromFile['tback_max']
state_min = normFromFile['state_min_loc'] if localState else normFromFile['state_min_abs']
state_max = normFromFile['state_max_loc'] if localState else normFromFile['state_max_abs']


# Get some test samples
dataset_test = ScenarioDataset('big2.h5', sequence_length=1, local=localState)
torch.manual_seed(time.time())
dataloader_test = DataLoader(dataset_test, batch_size=5, shuffle=True)

with torch.no_grad():
    for state, actual_tback in dataloader_test:
        state = (state - state_min) / (state_max - state_min)
        predicted_tback = bestmodel(state)
        predicted_tback_denorm = predicted_tback * (tback_max - tback_min) + tback_min
        print(predicted_tback)
        print(state)
        print("Sample | Actual tback | Predicted tback | Error")
        print("-" * 70)
        
        for i in range(state.shape[0]):
            state_sample = state[i, :3]  # First 3 dimensions
            actual = actual_tback[i].item()
            predicted = predicted_tback_denorm[i].item()
            error = abs(actual - predicted)
            print(f"{i+1:6d} | {actual:11.4f} | {predicted:15.4f} | {error:6.4f}")
        
        break  # Just show first batch
