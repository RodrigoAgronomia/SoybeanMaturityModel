import os
import cv2
import torch
import numpy as np
import pandas as pd


class MyDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.df = self._get_img_df()
    
    def __len__(self):
        return len(self.df)

    def _get_img_df(self):
        img_names = os.listdir(self.img_dir)
        df = pd.DataFrame(img_names, columns=['Filename'])
        df['Plot_ID'] = df.Filename.str[:-8]
        df['DOY'] = df.Filename.str[-7:-4].astype('int')
        df = pd.pivot_table(df, values = 'Filename', index='Plot_ID', columns='DOY', aggfunc='first').reset_index()
        return(df)

    def _transform(self, x):
        x = np.moveaxis(x, -1, -3)
        x = torch.tensor(x, dtype = torch.float)
        x = (x - 128) / 35
        return(x)

    def __getitem__(self, idx):
        Plot_ID = self.df.Plot_ID.iloc[idx]
        sdf = self.df.iloc[idx, 1:]
        img_dates = 0.1 * torch.tensor(sdf.index, dtype = torch.float32)
        imgs = np.stack([cv2.imread(os.path.join(self.img_dir, f)) for f in sdf.values])
        imgs = self._transform(imgs)
        return (Plot_ID, imgs, img_dates)