import torch
import torch.utils.data
import torch.nn.functional as F

from pathlib import Path
import numpy as np
import random
import os

DECODER_DIMENSION = {
    'type'      : 0,
    'beat'      : 1,
    'density'   : 2,
    'pitch'     : 3,
    'duration'  : 4,
    'velocity'  : 5,
    'strength'  : 6,
    'i_beat'    : 7,
    'n_beat'    : 8,
}

class DrumDataset(torch.utils.data.Dataset):
    """
    This is the main class to load drum data.
    """        
    def __init__(self, npz_file_path, split = 'train'):
        self.split = split
        self.data = np.load(npz_file_path, allow_pickle=True)
        self.encoder = self.data['encoder']
        self.decoder = self.data['decoder'][:, :, [0, 1, 2, 3, 4, 5, 6, 7]]
        self.decoder_mask = self.data['decoder_mask'][:, :]
        self.decoder_n_class = np.max(self.decoder, axis=(0, 1)) + 1

    def __len__(self):
        return len(self.encoder)

    def __getitem__(self, index):
        # to tensor
        en_x = torch.from_numpy(self.encoder[index, :]).long()
        de_x = torch.from_numpy(self.decoder[index, :-1]).long()
        de_y = torch.from_numpy(self.decoder[index, 1:]).long()
        de_mask = torch.from_numpy(self.decoder_mask[index, 1:]).float()
        if(self.split != 'train'):
            init_density = np.sum(self.data['encoder_mask'][index][:17])
            id_ = self.data['metadata'][index]['id']
            bpm = self.data['metadata'][index]['tempo']
            return en_x, de_x, de_y, de_mask, id_, bpm, init_density
        else:
            return en_x, de_x, de_y, de_mask
        
    def get_decoder_n_class(self):
        return self.decoder_n_class
    
    def get_len(self, index):
        return self.data['metadata'][index]['en_len'], self.data['metadata'][index]['de_len']