import numpy as np
import torch

def get_screen(env, resize):
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))  # transpose into torch order (CHW)
    screen = np.ascontiguousarray(screen, dtype=np.float32)  # /255
    screen = screen[0, 34:194, :]
    screen[screen == 144] = 0
    # Resize, and add a batch dimension (BCHW)
    screen = resize(screen)
    screen[screen != 0] = 1

    return screen.unsqueeze(0)

def update_state(env, resize, state, done=False):
    current_screen = get_screen(env, resize)
    next_state_update = torch.zeros(state.shape)

    if not done:
        next_state_update[:, 1:4, :, :] = state[:, 0:3, :, :]
        next_state_update[:, 0, :, :] = current_screen
    else:
        next_state_update = None

    return next_state_update