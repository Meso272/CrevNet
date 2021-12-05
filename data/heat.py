import socket
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
class HEAT(Dataset):
    
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, data_path, start_idx=0,end_idx=6000,seq_len=20, image_height=64,image_width=64,gmax=100,gmin=0,norm_to_tanh=False):
        #path = data_root
        self.seq_len = seq_len
        
        self.image_height = image_height
        self.image_width=image_width
        
        
        self.channels = 1
 
        #self.data =np.fromfile(data_path,dtype=np.float32).reshape((-1,1,image_height,image_width))[start_idx:end_idx]
        self.data=np.zeros((end_idx-start_idx,1,image_height,image_width))
        for i in range(start_idx,end_idx):
            filename="%d.dat" % i
            filepath=os.path.join(data_path,filename)
            self.data[i-start_idx][0]=np.fromfile(filepath,dtype=np.float32).reshape((image_height,image_width))
        self.data=(self.data-gmin)/(gmax-gmin)
        if norm_to_tanh:
            self.data=self.data*2-1
        self.N = self.data.shape[0]

   
          
    def __len__(self):
        return self.N-self.seq_len+1

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len]
