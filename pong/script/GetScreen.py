import numpy as np
import torch

def get_screen(env, resize, device):
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))  # transpose into torch order (CHW)
    screen = np.ascontiguousarray(screen, dtype=np.float32)  # / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    screen = screen[0, 34:194, :]
    screen[screen == 144] = 0
    screen = resize(screen)
    screen = screen / 255
    screen[screen != 0] = 1
    return screen.unsqueeze(0).to(device)
