import torch
import torch.utils.data
import torch.nn.functional as F

from pathlib import Path
import numpy as np
import os

DECODER_DIMENSION = {
    'type'      : 0,
    'beat'      : 1,
    'density'   : 2,
    'pitch'     : 3,
    'duration'  : 4,
    'instr_type': 5,
    'velocity'  : 6,
    'strength'  : 7,
    'i_beat'    : 8,
    'n_beat'    : 9,
    'p_beat'    : 10,
}

class CPDataset(torch.utils.data.Dataset):
    """
    This is the main class to load multi-track data.
    """        
    def __init__(self, npz_file_path, embedding_path, split = 'train'):
        self.split = split
        self.embedding_path = embedding_path
        self.data = np.load(npz_file_path, allow_pickle=True)
        if self.split == 'val':
            length = len(self.data['encoder'])
            if length > 1000:
                length = 1000
            self.encoder = self.data['encoder'][:length, :, [1, 0, 2, 3, 4, 5, 6, 7, 10, 8]]
            self.decoder = self.data['decoder'][:length, :, [1, 0, 2, 3, 4, 5, 6, 7, 10, 8]]
            self.decoder_mask = self.data['decoder_mask'][:length, :]
            self.decoder_n_class = np.max(self.decoder, axis=(0, 1)) + 1
            self.metadata = self.data['metadata'][:length]
        else:
            self.encoder = self.data['encoder'][:, :, [1, 0, 2, 3, 4, 5, 6, 7, 10, 8]]
            self.decoder = self.data['decoder'][:, :, [1, 0, 2, 3, 4, 5, 6, 7, 10, 8]]
            self.decoder_mask = self.data['decoder_mask'][:, :]
            self.decoder_n_class = np.max(self.decoder, axis=(0, 1)) + 1
            self.metadata = self.data['metadata']

    def __len__(self):
        return len(self.encoder)

    def __getitem__(self, index):
        # to tensor
        en_x = torch.from_numpy(self.encoder[index, :]).long() 
        de_x = torch.from_numpy(self.decoder[index, :-1]).long()
        de_y = torch.from_numpy(self.decoder[index, 1:]).long()
        de_mask = torch.from_numpy(self.decoder_mask[index, 1:]).float()
        id_ = self.metadata[index]['id']
        style_embedding = np.load(os.path.join(self.embedding_path, id_+'.npy'))
        style_embedding = torch.from_numpy(style_embedding).float()
        return en_x, de_x, de_y, de_mask, style_embedding

    def get_decoder_n_class(self):
        return self.decoder_n_class
    
    def get_len(self, index):
        return self.data['metadata'][index]['en_len'], self.data['metadata'][index]['de_len']