import torch
import torch.nn as nn
import torch.nn.functional as F


class MaturityPrediction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        cnn_dim = [3, 4, 6, 8, 10, 32]
        fc_dim =  [32, 16, 8, 5]
        pred_dim = [30, 15, 5]

        # Convolutional layers
        cnn_layers = nn.ModuleList()
        for i in range(len(cnn_dim) - 1):
            cnn_layer = nn.Sequential(
                nn.Conv2d(in_channels=cnn_dim[i],
                         out_channels=cnn_dim[i + 1],
                         kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size = 2),
                nn.Dropout2d(0.1),
                nn.LeakyReLU(),
            )
            cnn_layers.append(cnn_layer)
        cnn_layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.cnn_layers = nn.Sequential(*cnn_layers)

        
        # Fully conected layers
        fc_layers = nn.ModuleList()
        for i in range(len(fc_dim) - 1):
            fc_layer = nn.Sequential(
                nn.Conv2d(fc_dim[i], fc_dim[i + 1], 1),
                nn.Dropout2d(0.1),
            )
            fc_layers.append(fc_layer)
        self.fc_layers = nn.Sequential(*fc_layers)
        self.fc_pred_layer = nn.Conv2d(fc_dim[-1], 1, 1)
        
        # Prediciton layers
        pred_layers = nn.ModuleList()
        for i in range(len(pred_dim) - 1):
            pred_layer = nn.Sequential(
                nn.Conv2d(pred_dim[i], pred_dim[i + 1], 1),
                nn.Dropout2d(0.1),
            )
            pred_layers.append(pred_layer)
        pred_layers.append(nn.Conv2d(pred_dim[-1], 1, 1))
        self.pred_layers = nn.Sequential(*pred_layers)

            
    def extract_img_features(self, imgs):
        feats = self.cnn_layers(imgs)
        feats = self.fc_layers(feats)
        preds = self.fc_pred_layer(feats)
        return([preds, feats])

    
    def extract_features(self, imgs):
        shp = imgs.shape
        imgs = imgs.reshape(-1, *shp[2:])
        preds, feats = self.extract_img_features(imgs)
        preds = preds.reshape(shp[0], shp[1], *preds.shape[1:])
        feats = feats.reshape(shp[0], shp[1], *feats.shape[1:])
        return([preds, feats])
   

    def calc_best_features(self, preds, feats, n = 5):
        idx = preds.abs().argmin(1).cpu() - 2
        idx = torch.clamp(idx[:,None], 0, preds.shape[1] - n)
        img_idx = idx + torch.arange(n)
        cidx = torch.arange(idx.shape[0])[:,None]
        best_feats = feats[cidx, img_idx]
        return(best_feats, img_idx)
    
    
    def forward(self, imgs, img_dates):
        
        preds, feats = self.extract_features(imgs)
        best_feats, img_idx = self.calc_best_features(preds.mean((2,3,4)), feats)
        best_feats = best_feats.reshape(best_feats.shape[0], -1, *best_feats.shape[3:])
        imgd = img_dates[img_idx]
        data_imgd = 0.1 * (imgd - imgd[:,[2]])
        data_imgd = data_imgd[:,:, None,None].repeat(1, 1, *best_feats.shape[-2:])
        feats = torch.cat([data_imgd, best_feats], 1)
        predf = self.pred_layers(feats)
        predf = imgd[:,[2],None,None] - predf
        return(predf)
    

    
if __name__ == "__main__":
    model = MaturityPrediction([3, 4, 6, 8, 10, 32], [32, 16, 8, 5], [30, 15, 5])
    model.load_state_dict(torch.load('../data/model_multi_UIUC.pth'))
    torch.save(model, '../data/model_2019_vf.pth')
