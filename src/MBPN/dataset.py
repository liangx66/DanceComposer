import torch
import torch.utils.data
import torch.nn.functional as F

from pathlib import Path
import numpy as np
import random
import os

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


class AISTDataset(torch.utils.data.Dataset):
    """
    This is the main class to load video data.
    """        

    def __init__(self, motion_files, music_beat_path, duration = 12.0, fps = 60.0, split = 'train'):
        self.fps = fps
        self.num_frames =int(duration * fps)            
        self.music_beat_path = music_beat_path
        self.motion_files = files_to_list(motion_files)
        self.motion_files = [Path(motion_files).parent / x for x in self.motion_files]
        self.split = split

    def __len__(self):
        return len(self.motion_files)

    def __getitem__(self, index):
        # Read motion
        motion_filename = self.motion_files[index]
        motion = self.load_motion_to_torch(motion_filename, self.num_frames, part=25)  # [C, T, V, M]

        # Read audio
        motion_id = motion_filename.name.split(".")[0].split("_")   # e.g.: gBR_sBM_c06_d05_mBR0_ch01
        genre = motion_id[0]      
        situation = motion_id[1]  
        dancer_id = motion_id[3]  
        music_id =  motion_id[4]  
        num = motion_id[-1]
        if situation == 'sBM':
            music_beat_filename = genre+'_'+situation+'_'+dancer_id+'_'+music_id+'.npy'   # gBR_sBM_d05_mBR0.mp3
        if situation == 'sFM':
            music_beat_filename = genre+'_'+situation+'_'+dancer_id+'_'+music_id+'_'+num+'.npy'   # gBR_sFM_d05_mBR0_0.mp3
        music_beat_file = os.path.join(self.music_beat_path, music_beat_filename)
        music_beats = self.get_music_beat(music_beat_file, self.num_frames)    # music_beats:[T]
        id_ = motion_filename.name.split(".")[0] # +'.npy'
        return music_beats, motion, id_
    

    def get_music_beat(self, infile_path, num_frames = 360):
        data = np.load(infile_path)
        l = data.shape[0]
        if l < num_frames:
            data = np.concatenate((data, np.zeros((int(num_frames) - l))))
        if l > num_frames:
            data = data[:num_frames]
        return torch.from_numpy(data).long()


    def load_motion_to_torch(self, full_path: str, length: int, part=25):
        data = np.load(full_path)           
        T, V_C = data.shape
        data = data.reshape(T, -1, 3)[:, :, :-1]    # [T, V, C]
        if T < length+1: 
            V = data.shape[1]
            data = np.concatenate((data, np.zeros((int(length+1) - T, V, 2))))
        if T > length+1:
            data = data[:length+1, :, :]
        data = torch.from_numpy(data).float()   
        # data = self.pixel_to_normalized_coordinates(data, (288, 512))  
        data[:length, :, :] = data[1:length+1, :, :] - data[:length, :, :] # first order difference
        data = data[:length, :, :]                 
        if part == 25:
            data = data.reshape(length, part, 2, 1)     # [T, V, C] -> [T, V, C, 1]
            data = data.permute(2, 0, 1, 3)             # [T, V, C, 1] -> [C, T, V, 1]
        # if self.split == 'train':
        #     data = random_move(data)
        return data


    def pixel_to_normalized_coordinates(self, coords, size):
        """Convert from pixel coordinates to normalized coordinates.

        Args:
        coords: Coordinate tensor, where elements in the last dimension are ordered as (x, y, ...). B, V, C
        size: Number of pixels in each spatial dimension, ordered as (..., height, width).  (288, 512)

        Returns:
        `coords` in normalized coordinates.
        """
        if torch.is_tensor(coords):
            size = coords.new_tensor(size).flip(-1)
        return ((2 * coords + 1) / size) - 1


    def random_move(self,
                    data_numpy,
                    angle_candidate=[-10., -5., 0., 5., 10.],
                    scale_candidate=[0.9, 1.0, 1.1],
                    transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                    move_time_candidate=[1]):
        # input: C,T,V,M
        C, T, V, M = data_numpy.shape
        move_time = random.choice(move_time_candidate)
        node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
        node = np.append(node, T)
        num_node = len(node)

        A = np.random.choice(angle_candidate, num_node)
        S = np.random.choice(scale_candidate, num_node)
        T_x = np.random.choice(transform_candidate, num_node)
        T_y = np.random.choice(transform_candidate, num_node)

        a = np.zeros(T)
        s = np.zeros(T)
        t_x = np.zeros(T)
        t_y = np.zeros(T)

        # linspace
        for i in range(num_node - 1):
            a[node[i]:node[i + 1]] = np.linspace(
                A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
            s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                                node[i + 1] - node[i])
            t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                                node[i + 1] - node[i])
            t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                                node[i + 1] - node[i])

        theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                        [np.sin(a) * s, np.cos(a) * s]])

        # perform transformation
        for i_frame in range(T):
            xy = data_numpy[0:2, i_frame, :, :]
            new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
            new_xy[0] += t_x[i_frame]
            new_xy[1] += t_y[i_frame]
            data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

        return data_numpy

