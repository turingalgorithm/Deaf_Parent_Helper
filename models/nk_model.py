import torch.nn as nn
import torchvision.models as models
# model.py
import torch

class nkModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        #self.model = models.densenet201(pretrained=True)
        self.model = models.mobilenet_v3_small(pretrained=True)
        self.model.fc = nn.Linear(1000, args.class_number)

    def forward(self, inputs):
        #print("inputs shape", inputs.shape)
        pred = self.model(inputs)
        pred = self.model.fc(pred)

        return pred


class Dvector(nn.Module):
    def __init__(self, args):
        super(Dvector, self).__init__()
        self.n_spks = args.n_spks
        indim = args.indim
        outdim = args.outdim
        self.linears = nn.Sequential(nn.Linear(indim, outdim),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     nn.Linear(outdim, outdim),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     nn.Linear(outdim, 128),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     nn.Linear(128, 128),
                                     nn.LeakyReLU(negative_slope=0.2))

        self.clf = nn.Linear(128, self.n_spks)

    def forward(self, x, extract=False):
        # Normalize input features.
        x_mean = torch.mean(x, -1)
        x_var = torch.std(x, -1)
        x_var[x_var < 0.01] = 0.01
        x = (x - x_mean[:, :, None]) / x_var[:, :, None]

        x = self.linears(x)

        x = x.mean(dim=1)
        if extract:
            x = self.clf(x)

        return x