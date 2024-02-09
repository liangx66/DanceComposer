import torch
import torch.utils.data
import torch.nn.functional as F

from pathlib import Path
import numpy as np
import random
from PIL import Image
import os
from audio_processor import compute_melgram

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

class JointDataset(torch.utils.data.Dataset):
    """
    This is the main class to load music data and dance video data.
    """
    def __init__(self, motion_files, music_path, duration = 12.0, fps = 60.0, split = 'train'):
        self.fps = fps
        self.num_frames =int(duration * fps)
        self.motion_files = files_to_list(motion_files)
        self.motion_files = [Path(motion_files).parent / x for x in self.motion_files]
        self.split = split
        self.music_path = music_path
        self.label_to_index = { 'gBR':0,
                                'gHO':1,
                                'gJB':2,
                                'gJS':3,
                                'gKR':4,
                                'gLH':5,
                                'gLO':6,
                                'gMH':7,
                                'gPO':8,
                                'gWA':9,
                            }

    def __len__(self):
        return len(self.motion_files)

    def __getitem__(self, index):
        # Read motion
        motion_filename = self.motion_files[index]
        motion = self.load_motion_to_torch(motion_filename, self.num_frames, part=25)  # [C, T, V, M]
        # Read label
        file_name = motion_filename.name.split(".")[0]
        label_name = file_name.split('_')[0]
        label_index_onehot = self.label_to_index.get(label_name)
        # Read music
        music_name = file_name.split('_')[4]
        music_file = os.path.join(self.music_path, music_name+'.mp3')
        melgram = compute_melgram(music_file)
        melgram = torch.from_numpy(melgram).float()
        return motion, melgram, label_index_onehot

    def load_motion_to_torch(self, full_path: str, length: int, part=25):
        data = np.load(full_path)
        # downsample
        stride = int(60.0/self.fps)
        data = data[::stride, :]
        T = data.shape[0]
        data = data.reshape(T, -1, 3)[:, :, :-1]
        if T < length:
            V = data.shape[1]
            data = np.concatenate((data, np.zeros((int(length) - T, V, 2))))
        if T > length:
            data = data[:length, :, :]
        data = torch.from_numpy(data).float()   
        data = self.pixel_to_normalized_coordinates(data, (288, 512))        
        if part == 25:                             
            data = data.reshape(length, part, 2, 1)     # [T, V, C] -> [T, V, C, 1] 
            data = data.permute(2, 0, 1, 3)     # [T, V, C, 1] -> [C, T, V, 1]
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

class MotionDataset(torch.utils.data.Dataset):
    """
    This is the main class to load video data.
    """        
    def __init__(self, motion_files, duration = 12.0, fps = 60.0, split = 'train'):
        self.fps = fps
        self.num_frames =int(duration * fps)
        self.motion_files = files_to_list(motion_files)
        self.motion_files = [Path(motion_files).parent / x for x in self.motion_files]
        self.split = split
        self.label_to_index = { 'gBR':0,
                                'gHO':1,
                                'gJB':2,
                                'gJS':3,
                                'gKR':4,
                                'gLH':5,
                                'gLO':6,
                                'gMH':7,
                                'gPO':8,
                                'gWA':9,
                            }

    def __len__(self):
        return len(self.motion_files)

    def __getitem__(self, index):
        # Read motion
        motion_filename = self.motion_files[index]
        motion = self.load_motion_to_torch(motion_filename, self.num_frames, part=25)
        # Read label
        file_name = motion_filename.name.split(".")[0]
        label_name = file_name.split('_')[0]
        label_index_onehot = self.label_to_index.get(label_name)
        return motion, label_index_onehot, file_name

    def load_motion_to_torch(self, full_path: str, length: int, part=25):
        data = np.load(full_path)
        # downsample
        stride = int(60.0/self.fps)
        data = data[::stride, :]
        T = data.shape[0]
        data = data.reshape(T, -1, 3)[:, :, :-1]    # [T, V, C]
        if T < length:
            V = data.shape[1]
            data = np.concatenate((data, np.zeros((int(length) - T, V, 2))))
        if T > length:
            data = data[:length, :, :]
        data = torch.from_numpy(data).float()   
        data = self.pixel_to_normalized_coordinates(data, (288, 512))
        if part == 25:                             
            data = data.reshape(length, part, 2, 1)     # [T, V, C] -> [T, V, C, 1]
            data = data.permute(2, 0, 1, 3)     # [T, V, C, 1] -> [C, T, V, 1]
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



class MusicDataset(torch.utils.data.Dataset):
    """
    This is the main class to load music melgram data.
    """        

    def __init__(self, melgram_files, split = 'train'):
        self.melgram_files = melgram_files
        self.melgram_files = files_to_list(melgram_files)
        # self.melgram_files = [Path(melgram_files).parent / x for x in self.melgram_files]
        self.split = split
        self.label_to_index = {'blues':0,
                                'classical':1, 
                                'country':2,
                                'disco':3,
                                'hiphop':4, 
                                'jazz':5,
                                'metal':6,
                                'pop':7,
                                'reggae':8,
                                'rock':9, 
                            }

    def __len__(self):
        return len(self.melgram_files)

    def __getitem__(self, index):
        # Read input melgram image
        melgram_filename = self.melgram_files[index]
        data = np.load(melgram_filename)
        data = torch.from_numpy(data).float()

        label_name = melgram_filename.split('/')[-1].split('.')[0] 
        label_index_onehot = self.label_to_index.get(label_name)

        return data, label_index_onehot