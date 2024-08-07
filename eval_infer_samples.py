import os
import pdb
import random
import argparse
import numpy as np
import json 
import importlib
from tqdm import tqdm
from scipy.io import wavfile

import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.define_dataset import load_dataset
from utils.frame_runner import FrameRunner 
from utils.utils import AttrDict

# GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
print("Is GPU available:\t", torch.cuda.is_available())
print("#Available GPU:\t", torch.cuda.device_count())



# Configuration --------------------------------------------------------------

parser = argparse.ArgumentParser(description="Inference")

parser.add_argument('--exp_name',
                        default='00_2350bps_L350_S1000_E1000',
                        #default='01_2850bps_L350_S1000_E1500',
                            type=str, help="(str) Experiment name")
parser.add_argument('--q_tag',     
                        default='LSE',
                        #default='LS',
                        #default='L',
                            type=str)

parser.add_argument('--ckpt_num',
                        default=600000,
                            type=int, help="(int) Corresponding iteration")

parser.add_argument('--frame_size',
                        default=16000,
                            type=int, help="(int) Window length in inference")

parser.add_argument('--batch_size',     
                        default=64,
                            type=int)

parser.add_argument('--eval_seed',     
                        default=526,
                            type=int)

parser.add_argument('--data_name',
                        default='LibriSpeech',
                            type=str, help="(str) Dataset name")
parser.add_argument('--data_mode',
                        default='test_all',
                        #default='test_10',
                            type=str, help="(str) Dataset mode")



def save_wav(waveform, fname, save_wav_dir, data_name, sr):
    """ Save wavform as wav file """
    # waveform: (T)
    # fname: (1)

    # Save files
    if data_name.startswith('LibriSpeech'):
        save_spk_dir = os.path.join(save_wav_dir, fname.split('-')[0])
        os.makedirs(save_spk_dir, exist_ok=True)
        save_path = os.path.join(save_spk_dir, fname)

    waveform = waveform * 32768
    waveform = np.clip(waveform, -32768, 32767)
    wavfile.write(save_path, sr, waveform.astype(np.int16))


def save_wav_num(waveform, fname, save_wav_dir, data_name, sr):
    """ Save wavform as wav file """
    # waveform: (T)
    # fname: (1)

    # Save files
    os.makedirs(save_wav_dir, exist_ok=True)
    save_path = os.path.join(save_wav_dir, fname)

    waveform = waveform * 32768
    waveform = np.clip(waveform, -32768, 32767)
    wavfile.write(save_path, sr, waveform.astype(np.int16))


@torch.no_grad()
def _infer_full_frame_multiple(f_frame_list, f_feature_list, f_q_feature_list, wavlen_list, model, frame_runner, q_mode):
    """ Infer full frames of multiple files """

    # full_frame_list: list of (1, #frame, T) torch.Tensor (list size=num_file=B)
    num_file = len(f_frame_list)
    
    """ (1) Prepare for inference """

    # Reconstruct desired waveform and search max number of frames
    desired_wav_list = []
    num_frame_list = []
    for full_frame, wavlen in zip(f_frame_list, wavlen_list):
        desired_wav = frame_runner.reconstruct_windows(full_frame.squeeze(0).numpy())
        desired_wav = desired_wav[:wavlen]
        desired_wav_list.append(desired_wav)
        num_frame_list.append(full_frame.size(1))
    
    # Pad each file with zeros to have save num_frame for all files
    max_num_frame = max(num_frame_list)
    for i, (f_frame, f_feature, f_q_feature, num_frame) in enumerate(zip(f_frame_list, f_feature_list, f_q_feature_list, num_frame_list)):
        # f_frame: (1, #frame, T)
        pad_len = max_num_frame - f_frame.size(1)
        f_frame_list[i] = F.pad(f_frame, pad=(0, 0, 0, pad_len, 0, 0), mode='constant', value=0)
        f_feature_list[i] = F.pad(f_feature, pad=(0, 0, 0, 0, 0, pad_len, 0, 0), mode='constant', value=0)
        f_q_feature_list[i] = F.pad(f_q_feature, pad=(0, 0, 0, 0, 0, pad_len, 0, 0), mode='constant', value=0)
        # padded f_frame: (1, max_num_frame, frame_len)
        # padded f_feature: (1, max_num_frame, feature_len, feature_dim)
        assert f_frame_list[i].size(1) == max_num_frame

    """ (2) Infer """

    # Create batch of files and inference
    for fr_i in range(max_num_frame):
    
        """ (2-1) Stack input frames """
        # Stack a frame of each file 
        stacked_frame = torch.cat([f_frame_list[file_i][0, fr_i, :].unsqueeze(0) \
                                    for file_i in range(num_file)], dim=0)
        stacked_feature = torch.cat([f_feature_list[file_i][0, fr_i, :, :].unsqueeze(0) \
                                    for file_i in range(num_file)], dim=0)
        stacked_q_feature = torch.cat([f_q_feature_list[file_i][0, fr_i, :, :].unsqueeze(0) \
                                    for file_i in range(num_file)], dim=0)

        # stacked_frame: (B=num_file, T)
        stacked_frame = stacked_frame.unsqueeze(1).cuda()           # (B, 1, T)
        stacked_feature = stacked_feature.transpose(1, 2).cuda()    # (B, C, T)
        stacked_q_feature = stacked_q_feature.transpose(1, 2).cuda()    # (B, C, T)
    
        """ (2-2) Infer input frames """
        # Infer
        with torch.no_grad():
            if q_mode == 'disabled':
                stacked_gen_frame, _ = model(stacked_feature, stacked_q_feature)
            elif q_mode == 'rvq':
                stacked_gen_frame, _ = model(stacked_feature, stacked_q_feature, True)
            elif q_mode == 'sq':
                stacked_gen_frame, frame_dict = model(stacked_feature, stacked_q_feature, 'round')
            # stacked_gen_frame: (B=num_file, 1, T) 

        """ (2-3) Stack output and intermediate frames """
        # Get output frames
        stacked_gen_frame_np = stacked_gen_frame.detach().cpu().numpy() # (B, 1, T)
        
        # Stack output frames
        if fr_i == 0:
            stacked_recon_frame = stacked_gen_frame_np.copy()  # (B, 1, T)
        else:
            stacked_recon_frame = np.append(stacked_recon_frame, stacked_gen_frame_np, axis=1)
        # stacked_recon_frame: (B, fr_i+1, T) => (B, max_num_frame, T)

        if q_mode == 'sq':
            """ (2-4) Stack bitrate """
            info_value = frame_dict['info_res'].unsqueeze(1).detach().cpu().numpy()      # (B, 1, ???)
            entropy_frame_i = np.sum(info_value, axis=tuple(range(2, info_value.ndim))) # (B, 1)

            if fr_i == 0:
                entropy_frame = entropy_frame_i.copy()
            else:
                entropy_frame = np.append(entropy_frame, entropy_frame_i, axis=1)
            # entropy_frame: (B, max_num_frame)

   
    """ (3-1) Reconstruct waveforms"""
    # Reconstruct waveform after removing redundant (padded) frames
    recon_wav_list = []
    bitrate_list = []
    for i, (num_frames, wavlen) in enumerate(zip(num_frame_list, wavlen_list)):
        # Reconstructed signal
        recon_frame_i = stacked_recon_frame[i, :num_frames, :]   # (num_frames, T)
        recon_wav = frame_runner.reconstruct_windows(recon_frame_i)
        recon_wav = recon_wav[:wavlen]
        assert len(recon_wav) == len(desired_wav_list[i])
        recon_wav_list.append(recon_wav)

        if q_mode == 'sq':
            sec_wav = num_frames * frame_runner.frame_size/ 16000
            bitrate_wav = np.sum(entropy_frame[i, :num_frames]) / sec_wav / 1000
            bitrate_list.append(bitrate_wav)
        else:
            bitrate_list.append(0.)


    return desired_wav_list, recon_wav_list, bitrate_list




def main(args):
   
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    # Load trained model -----------------------------------------------------
    # Set load path
    ckpt_dir = os.path.join('checkpoint_infer', args.exp_name)
    load_path = os.path.join(ckpt_dir, '{}_iteration_{:08d}.ckpt'.format(args.exp_name, args.ckpt_num))

    # Load trained into
    package = torch.load(load_path)
    cfg = package['cfg']

    # Load vocoder model
    for name in cfg.model.module_list:
        if name.startswith("M0"):
            coder_name = name
            break

    models = importlib.import_module('models')    
    model = models.all_model_dict[coder_name](cfg.model[coder_name]).cuda()
    
    model.load_state_dict(package['coder'])
    model.eval()
    print()

    

    # Create test dataset ----------------------------------------------------

    # Define frame runner
    frame_runner = FrameRunner(cfg.data.ola)

    pdb.set_trace()
    # Define dataset
    noise_path_set = load_dataset('noise', cfg.data)
    if args.data_name == 'LibriSpeech':
        if args.data_mode.startswith('test'):  
            if args.data_mode.endswith('all'):
                eval_num = 2620
            else:
                eval_num = int(args.data_mode.split('_')[-1])
            eval_noise_rng = np.random.default_rng(seed=args.eval_seed)
            eval_noise_indices = eval_noise_rng.integers(0, len(noise_path_set), size=eval_num)

            eval_set = load_dataset('test', cfg.data, num_file=cfg.record.eval_num,
                                noise_path_set=noise_path_set, 
                                noise_indices=eval_noise_indices,
                                frame_runner=frame_runner,
                                seed=cfg.record.eval_seed)

    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False)
    
    eval_loader_dict = {}
    for snr in cfg.data.mix.snr_list:
        key = str(snr)+'dB' if snr != 'clean' else 'clean'
        subset = eval_set.snr_subset_dict[key]
        eval_loader_dict[key] = DataLoader(subset, batch_size=1, shuffle=False)
    print()


    # Load centroids
    centroid_path = os.path.join(cfg.data.centroid.root_dir,
                                'wavlm_layer_{}_vocabsize_{}.pkl'.format(
                                        cfg.data.centroid.layer, cfg.data.centroid.vocabsize))
    with open(centroid_path, 'rb') as cPickle_file:
        centroid = cPickle.load(cPickle_file)

    centroid_dict = {}
    centroid = torch.Tensor(centroid).cuda()
    centroid_t = centroid.transpose(0, 1)
    centroid_dict['centroid'] = centroid          # (#clusters, feature_dim)
    centroid_dict['centroid_t'] = centroid_t      # (feature_dim, #clusters)
    centroid_dict['c_t_norm'] = (centroid_t**2).sum(0, keepdim=True)   # (1, #clusters)

    # Evaluate with metric and save result -----------------------------------

    # Prepare directories to save waveforms and bitrate files
    data_dir = '{}_{}_{}'.format(args.data_name, args.data_mode, args.frame_size)
    ckpt_name = 'iteration_{:08d}'.format(args.ckpt_num)
    for q_name in ['LSE', 'LS']:
        save_dir = './eval_samples/{}/{}/{}/{}'.format(data_dir, args.exp_name, q_name, ckpt_name)
        os.makedirs(save_dir, exist_ok=True)
    desired_dir = './eval_samples/{}/{}'.format(data_dir, 'desired')
    
    pbar = tqdm(eval_loader, desc="Infer...", ncols=80)
    for i, (fname, f_frame, f_feature, f_q_feature, wavlen) in enumerate(pbar):
        # all_frames: (1, #frame, T)
        # fname: (1,)
        # wavlen: (1, )

        # Create list of files (size: batch size)
        if i % args.batch_size == 0:
            f_frame_list = []
            f_feature_list = []
            f_q_feature_list = []
            wavlen_list = []
            fname_list = []
        f_frame_list.append(f_frame)
        f_feature_list.append(f_feature)
        f_q_feature_list.append(f_q_feature)
        wavlen_list.append(wavlen)

        if args.data_mode == 'tb_samples':
            fname_list.append('wav_{:02d}.wav'.format(i))
        elif args.data_mode.startswith('valid'):
            fname_list.append('wav_{:03d}.wav'.format(i))
        else:
            fname_list.append(fname[0])
        
        # Infer when len(full_frame_list) == num_file or the end of pbar
        if (i+1) % args.batch_size == 0 or (i+1) == len(pbar):
            desired_wav_list, recon_wav_list, bitrate_list = _infer_full_frame_multiple(f_frame_list, f_feature_list, f_q_feature_list, wavlen_list, model, frame_runner, args.q_mode)

            # Save waveforms and bitrate files
            for fname, desired_wav, recon_wav, bitrate in zip(fname_list, desired_wav_list, recon_wav_list, bitrate_list):
                if args.data_mode.startswith('test'):   
                    save_wav(recon_wav, fname, save_dir, data_dir, sampling_rate)
                    #save_wav(desired_wav, fname, desired_dir, data_dir, sampling_rate)
                else:
                    save_wav_num(recon_wav, fname, save_dir, data_dir, sampling_rate)
                with open(bitrate_path, 'a') as f:
                    f.write('{},{:4.0f}\n'.format(fname, bitrate*1000))




if __name__ == '__main__':
    
    args = parser.parse_args()

    print("Args:\n", args)
    print()
    print("Experiment name:     ", args.exp_name)
    print("Iteration:           ", args.ckpt_num)
    print("q_tag:              ", args.q_tag)
    print("Data mode:           ", args.data_mode)
    print()
    
    main(args)
    print("Finished!")
    print()







