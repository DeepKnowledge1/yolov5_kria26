import numpy as np
import torch
import torch.nn as nn

from utils.general import (
    check_version,
)


class Post_Process(nn.Module):
    def __init__(self, anchors=(), nc=80):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        # self.anchors = anchors
        self.no = self.nc + 5  # +180  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1).to("cpu")] * self.nl  # init grid
        self.anchors = torch.tensor(anchors).float().to("cpu").view(self.nl, -1, 2)
        self.anchor_grid = [torch.zeros(1).to("cpu")] * self.nl  # init anchor grid
        self.stride = torch.tensor([8, 16, 32])

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        d = self.anchors[i].device
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d), torch.arange(nx, device=d)
        yv, xv = torch.meshgrid([y, x])

        grid = torch.stack((xv, yv), 2).expand(shape).float()
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape).float()

        return grid, anchor_grid

    def post_process(self, x):
        z = []  # inference output
        for i in range(self.nl):
            bs, ny, nx, _ = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].numpy()
            #  Rearrange the tensors to restore the original order (B, C, H, W), as the DPU operates with a different tensor order (B, H, W, C).
            x[i] = np.transpose(x[i], (0, 3, 1, 2))
            x[i] = np.reshape(x[i], (bs, self.na, self.no, ny, nx))
            x[i] = np.transpose(x[i], (0, 1, 3, 4, 2))
            x[i] = torch.from_numpy(x[i])
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
            # Detect (boxes only)
            xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
            xy = (xy * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
            y = torch.cat((xy, wh, conf), 4)

            z.append(y.view(bs, self.na * nx * ny, self.no))
        return (torch.cat(z, 1), x)
