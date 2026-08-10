#----------------------------------------------------------------
# Global variables and functions
#----------------------------------------------------------------

import numpy as np
import matplotlib.colors as mcolors
from numpy.linalg import norm
from torch.utils.data import Dataset
from torch import nn

mu = 398600.4418 #km^3/s^2
workingDir = 'C:\\Users\\Owner\\Desktop\\thesis'

def rainbow_plot2(ax, x, y, linewidth=2, **kwargs):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) < 2:
        raise ValueError("At least 2 points are required to draw a line.")

    # Rainbow spans hue 0.0 (red) → ~0.83 (purple/violet) in HSV space
    hues = np.linspace(0.0, 0.83, len(x) - 1)

    for i, hue in enumerate(hues):
        color = mcolors.hsv_to_rgb([hue, 1.0, 1.0])
        ax.plot(
            [x[i], x[i + 1]],
            [y[i], y[i + 1]],
            color=color,
            linewidth=linewidth,
            **kwargs,
        )
def rainbow_plot(ax, x=[], y=[]):
    if len(x)!=0 and len(y)!=0:
      rainbow_plot2(ax, x, y)
    else:
      rainbow_plot2(ax, range(len(x)), x)

def ECIprop(pos, vel, step, accel=None):
    if accel is None:
        accel = np.zeros((3,1))
    accel = np.asarray(accel).reshape(3,)

    halfvel = vel + (0.5*(-mu*pos*step/norm(pos)**3)*step) + .5*accel*step
    newpos = pos + (halfvel*step)
    newvel = halfvel + (0.5*(-mu*newpos*step/norm(newpos)**3)*step) + .5*accel*step
    return newpos, newvel
def ECIPropWithSpringDamper(pos,vel,step,masspos, massvel, mkc, drymass, accel=None):
    stepsize = .1
    if accel is None:
        accel = np.zeros((3,1))
    accel = np.asarray(accel).reshape(3,)
    newpos = np.asarray(pos).reshape(3,)
    newvel = np.asarray(vel).reshape(3,)
    newmasspos = np.asarray(masspos).reshape(3,)
    newmassvel = np.asarray(massvel).reshape(3,)
    times = np.append(np.arange(0,step,stepsize),step) #should stop at step-stepsize
    partialStepSizes = times[1:] - times[:-1]
    for partialStep in partialStepSizes:
        satNonGravAcell =  (( mkc[1]/drymass)*newmasspos) + (( mkc[2]/drymass)*newmassvel) + accel
        satGravAcell = mu*newpos/norm(newpos)**3
        massAccel = ((-mkc[1]/mkc[0])*newmasspos) + ((-mkc[2]/mkc[0])*newmassvel) - satNonGravAcell

        
        newmassvel = newmassvel + partialStep*massAccel
        newmasspos = newmasspos + partialStep*newmassvel
    
        newvel = newvel + (partialStep*satNonGravAcell) + (partialStep*satGravAcell)
        newpos = newpos + (partialStep*newvel)
    return newpos, newvel, newmasspos, newmassvel
def gravity_gradient(pos):
    pmag = np.linalg.norm(pos)
    return (mu / pmag**5) * (3 * np.outer(pos, pos) - pmag**2 * np.eye(3))
def indexInterp(arr, dblIndex,axis):
    if dblIndex>np.size(arr, axis=axis):
        raise ValueError("Index must be less than array axis max index.")
    if dblIndex<0:
        raise ValueError("Index must non-negative")
    i = int(dblIndex)
    d = dblIndex - i
    j=i+1
    return ((1-d) * np.take(arr, i, axis=axis)) + ((d) * np.take(arr, j, axis=axis))
def ECI2RIC(pos, vel, eps=1e-10):
    if norm(pos) < eps or norm(vel) < eps:
        return np.eye(3)
    radial = pos / norm(pos)
    cross = np.cross(pos, vel)
    cross = cross / norm(cross)
    intrack = np.cross(cross, radial)
    return np.vstack((radial, intrack, cross))

def printt(*args, file):
    print(*args)
    print(*args, file=file)

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
                chaser_ric_pos = torch.FloatTensor(group['chaser_pos_ric'][start_step:end_step])
                chaser_ric_vel = torch.FloatTensor(group['chaser_vel_ric'][start_step:end_step])

                # Magnitude of target position: (seq_len, 1)
                target_mag = torch.norm(target_pos, dim=1, keepdim=True)

                # Stack: (seq_len, 7)
                state = torch.cat([chaser_ric_pos, chaser_ric_vel, target_mag], dim=1)
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
    def __init__(self, input_size=12, hidden_size=64, num_hidden_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_hidden_layers, batch_first=True)
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