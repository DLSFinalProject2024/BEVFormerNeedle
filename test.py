import torch
import math
import torch.nn.functional as F

seed = 32
torch.manual_seed(seed)
def grid_sample(A, grid, mode = 'bilinear', padding_mode = 'zeros', align_corners = False):
    h, w = A.shape[-2:]
    out = torch.zeros_like(A)
    A_pad = F.pad(A, (1,1,1,1), "constant", 0)
    offset_x = (w + 1) / 2
    offset_y = (h + 1) / 2
    xx = [0, 1, 0, 1]
    yy = [0, 0, 1, 1]
    for i in range(h):
        for j in range(w):
            x, y = grid[0,i,j,:]
            x_ = x * w / 2 + offset_x
            y_ = y * h / 2 + offset_y
            x_floor = math.floor(x_)
            y_floor = math.floor(y_)
            dx = x_ - x_floor
            dy = y_ - y_floor
            for k in range(4):
                out[0,:,i,j] += A_pad[0, :, y_floor + yy[k], x_floor + xx[k]] * (dx * ((xx[k] << 1) - 1) + 1 - xx[k]) * (dy * ((yy[k] << 1) - 1) + 1 - yy[k])
    return out
norm_tot = 0.0
for s1 in range(1, 10):
    for s2 in range(1, 10):
        for c in range(1,3):
            A_shape = (1, c, s1, s2)
            grid_shape = (1, s1, s2, 2)
            A = torch.rand(A_shape, dtype=float)
            grid = torch.rand(grid_shape, dtype=float)
            grid = grid * 2 - 1
            sample_ = grid_sample(A, grid)
            sample = F.grid_sample(A, grid, mode = 'bilinear', padding_mode = 'zeros', align_corners = False)
            norm_tot += torch.norm(sample - sample_)
print(norm_tot)