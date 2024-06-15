from functools import wraps
import torch
from torch import nn
from vit_pytorch.cross_vit import CrossViT
from models.binae import BinModel
from vit_pytorch.cross_vit import Attention

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

class Recorder(nn.Module):
    def __init__(self, BinModel, device = None):
        super().__init__()
        self.CrossViT = BinModel

        self.data = None
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

    def _hook(self, _, input, output):
        self.recordings.append(output.clone().detach())

    def _register_hook(self):
        modules = find_modules(self.CrossViT.encoder, Attention)
        for module in modules:
            handle = module.attend.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.CrossViT

    def clear(self):
        self.recordings.clear()

    def record(self, attn):
        recording = attn.clone().detach()
        self.recordings.append(recording)

    def forward(self, img, gt_img):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        pred = self.CrossViT(img, gt_img)

        # move all recordings to one device before stacking
        target_device = self.device if self.device is not None else img.device
        recordings = tuple(map(lambda t: t.to(target_device), self.recordings))

        attns = torch.stack(recordings, dim = 1) if len(recordings) > 0 else None
        return pred, attns