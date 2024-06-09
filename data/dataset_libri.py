import os
import sys
import pdb
import json
import random
import numpy as np
from six.moves import cPickle
from tqdm import tqdm
import warnings
import librosa
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


class LibriTrainPathSet(Dataset):
    """
    Dataset of paths of train files (.pkl)
    self.fpath_list     : (list) Path (str) to .pkl files
    """
    def __init__(self, json_path, root_dir):
        
        # Load path list
        with open(json_path, 'r') as f:
            json_list = json.load(f)
        # Add root dirctory in front of files
        self.path_list = []
        for f in json_list:
            self.path_list.append(os.path.join(root_dir, f))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        return self.path_list[idx] 

# ----------------------------------------------------------------------------

class LibriTrainFrameSet(Dataset):
    """
    Dataset of frames in multiple files
    """
    def __init__(self, path_list, noise_loader, frame_runner, h):
    #def __init__(self, path_tuple, num_frame, init_block_idx, shift_size):
        """
        self.metadata       : (list) Info of the first frame of every num_frame_inter frames
                                        as (str) 'meta_idx, fname, frame_idx'
        self.waveform_dict  : (dict) Reference waveform of each file 
                                            (key: fname (str), value: waveform (np.array))
        self.frames_dict    : (dict) Reference frames of each file 
                                            (key: fname (str), value: frames (np.array))
        """
        
        # Load parameters
        snr_list = h.mix.snr_list

        # Construct noisy mixture
        fr_idx = 0
        file_dict = {}
        path_list = sorted(path_list)
        for i, cpath in enumerate(path_list):
            # Read file
            clean, csr = librosa.load(cpath, sr=None, mono=False)

            # Select noise and load file
            if noise_loader != None:
                npath = next(iter(noise_loader))[0]
                noise_full, nsr = librosa.load(npath, sr=None, mono=False)

                assert csr == nsr, "Should be the same sampling rate"

                # Select noise time range
                assert len(clean) < len(noise_full)
                siglen = len(clean)
                nidx = np.random.randint(0, len(noise_full)-siglen)
                noise = noise_full[nidx: nidx+siglen]
                
                # Add noise with random noise level
                snr = random.choice(snr_list)
                if snr != 'clean':
                    E_clean = np.mean(clean**2)**(1/2)
                    E_noise = np.mean(noise**2)**(1/2)
                    
                    noisescalar = E_clean / (10**(snr/20)) / (E_noise + 1e-8)
                    noisenewlevel = noise * noisescalar
                    mix = clean + noisenewlevel
                else:
                    mix = clean.copy()
            
            else:
                mix = clean.copy()


            # Extract and stack frames
            frames, _ = frame_runner.extract_windows(mix, drop_last=True)   # (B, frame_size)
            c_frames, _ = frame_runner.extract_windows(clean, drop_last=True)
            
            if i == 0:
                self.data = frames.copy()
                self.c_data = c_frames.copy()
            else:
                self.data = np.append(self.data, frames, axis=0)
                self.c_data = np.append(self.c_data, c_frames, axis=0)

            # Update info
            fname = cpath.split('/')[-1]
            spk = fname.split('-')[0]
            if spk not in file_dict.keys():
                file_dict[spk] = {}
                file_dict[spk]['start_idx'] = fr_idx
                file_dict[spk]['length'] = 0
            fr_idx += len(frames)
            file_dict[spk]['length'] += len(frames)

        self.file_dict = file_dict



    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        """ Get current frames and candidate previous samples (size: num_previous_samples) """
        # idx == meta_idx

        p_x, n_x = self._pick_pairs(idx)

        return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.c_data[idx]),\
                torch.FloatTensor(p_x), torch.FloatTensor(n_x)


    def _pick_pairs(self, idx):

        for key in self.file_dict.keys():
            st = self.file_dict[key]['start_idx']
            ln = self.file_dict[key]['length']

            if idx >= st and idx < st + ln:
                anchor_st = st
                anchor_ln = ln
                break

        # same speaker, diff utt
        r_list = list(range(anchor_st, idx))+list(range(idx+1, anchor_st+anchor_ln))
        if len(r_list)==0:
            r_list = [anchor_st]
        
        r_idx = np.random.choice(r_list)
        p_x = self.c_data[r_idx]

        # diff speaker, random utt
        r_list = list(range(0, anchor_st))+list(range(anchor_st+anchor_ln, len(self.data)))
        r_idx = np.random.choice(r_list)
        n_x = self.c_data[r_idx]

        return p_x, n_x



# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

class LibriInferSet():
    """
    Dataset of inference dataset (valid, test)
    validation mode: valid_all, valid_small, valid_vis
    """
    def __init__(self, split, num_file, noise_path_set, noise_indices, frame_runner, seed, h):
        """
        self.fpath_list : (list) File paths to inference files (include extension)
        self.frame_dict : (dict) Frames (np.array) extracted from inference files
        
        We don't use drop_last here, but clip the last frame to the length of original waveform
        Therefore, we need to save the length of original waveform
        """

        # Set parameters
        self.snr_list = h.mix.snr_list

        # Set data path
        if split == 'valid':
            data_dir = os.path.join(h.root_dir, 'dev-clean')
        elif split == 'test':
            data_dir = os.path.join(h.root_dir, 'test-clean')
        else:
            raise NotImplementedError("Not implemented split:", split)
        
        # Define fpath list
        with open(h.clean.json_path.replace('train', split), 'r') as f:
            json_list = json.load(f)
        cpath_list = []
        for f in json_list:
            cpath_list.append(os.path.join(h.root_dir, f))
        
        if not split.endswith('all'):
            cpath_list = sorted(cpath_list)
            rng = np.random.default_rng(seed)
            rng.shuffle(cpath_list)
            cpath_list = cpath_list[:num_file]

        self.snr_subset_dict = {}
        for snr in self.snr_list:
            self.snr_subset_dict[str(snr)+'dB'] = LibriInferSubSet()
        self.snr_subset_dict['clean'] = LibriInferSubSet()
        
        # Construct noisy mixture
        print("Creating inference dataset of [ {} {} ]".format(split, num_file))
        t = tqdm(total=len(cpath_list), ncols=100)
        for i, (cpath, ni) in enumerate(zip(cpath_list, noise_indices)):
            # Read file
            clean, csr = librosa.load(cpath, sr=None, mono=False)

            # Select noise and load file
            npath = noise_path_set[ni]
            noise_full, nsr = librosa.load(npath, sr=None, mono=False)

            cname = cpath.split('/')[-1][:-5]
            nname = '_'.join(npath.split('/')[-2:])[:-4]
            fname = 'C_{}_N_{}'.format(cname, nname)

            # Select noise time range
            assert len(clean) < len(noise_full)
            siglen = len(clean)
            nidx = np.random.randint(0, len(noise_full)-siglen)
            noise = noise_full[nidx: nidx+siglen]

            
            # Add noise with random noise level
            E_clean = np.mean(clean**2)**(1/2)
            E_noise = np.mean(noise**2)**(1/2)
           
            for snr in self.snr_list:
                if snr == 'clean':
                    continue

                noisescalar = E_clean / (10**(snr/20)) / (E_noise + 1e-8)
                noisenewlevel = noise * noisescalar
                mix = clean + noisenewlevel

                # Extract and frames
                frames, _ = frame_runner.extract_windows(mix, drop_last=True)   # (B, frame_size)
            
                # Update info
                self.snr_subset_dict[str(snr)+'dB'].frame_dict[fname] = frames.copy()
                self.snr_subset_dict[str(snr)+'dB'].fname_list.append(fname)
                self.snr_subset_dict[str(snr)+'dB'].wavlen_dict[fname] = siglen

            # clean
            frames, _ = frame_runner.extract_windows(clean, drop_last=True)   # (B, frame_size)
            # Update info
            self.snr_subset_dict['clean'].frame_dict[cname] = frames.copy()
            self.snr_subset_dict['clean'].fname_list.append(cname)
            self.snr_subset_dict['clean'].wavlen_dict[cname] = siglen

            
            t.update(1)
        t.close()



class LibriInferSubSet(Dataset):
    def __init__(self):

        self.fname_list = []
        self.frame_dict = {}
        self.wavlen_dict = {}


    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, idx):

        fname = self.fname_list[idx]

        data = torch.FloatTensor(self.frame_dict[fname])
        # data: torch.Tensor (1, #frame, frame_size)

        wavlen = int(self.wavlen_dict[fname])

        return fname, data, wavlen




