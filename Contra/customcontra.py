import gym
# import ctypes
from gym.wrappers import TimeLimit, ResizeObservation, GrayScaleObservation
from ray.rllib.env.wrappers.atari_wrappers import MaxAndSkipEnv
from gym.spaces import MultiDiscrete, Box
import numpy as np
import logging
# import sys
# import io
# import os
# import tempfile
# from contextlib import contextmanager


# libc = ctypes.CDLL(None)
# c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
log = logging.getLogger(__name__)


# @contextmanager
# def stdout_redirector(stream):
#     # The original fd stdout points to. Usually 1 on POSIX systems.
#     # ray_stdout = sys.stdout
#     # sys.stdout = sys.__stdout__
#     original_stdout_fd = sys.stdout.fileno()

#     def _redirect_stdout(to_fd):
#         """Redirect stdout to the given file descriptor."""
#         # Flush the C-level buffer stdout
#         libc.fflush(c_stdout)
#         # Flush and close sys.stdout - also closes the file descriptor (fd)
#         sys.stdout.close()
#         # Make original_stdout_fd point to the same file as to_fd
#         os.dup2(to_fd, original_stdout_fd)
#         # Create a new sys.stdout that points to the redirected fd
#         sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

#     # Save a copy of the original stdout fd in saved_stdout_fd
#     saved_stdout_fd = os.dup(original_stdout_fd)
#     try:
#         # Create a temporary file and redirect stdout to it
#         tfile = tempfile.TemporaryFile(mode='w+b')
#         _redirect_stdout(tfile.fileno())
#         # Yield to caller, then redirect stdout back to the saved fd
#         yield
#         _redirect_stdout(saved_stdout_fd)
#         # Copy contents of temporary file to the given stream
#         tfile.flush()
#         tfile.seek(0, io.SEEK_SET)
#         stream.write(tfile.read())
#     finally:
#         tfile.close()
#         os.close(saved_stdout_fd)
#         # sys.stdout = ray_stdout



class CustomContra(gym.Env):
    def __init__(self, config):
        import Contra
        logging.debug("Initializing custom contra env with config: " + str(config))
        self.env = TimeLimit(
            MaxAndSkipEnv(
                GrayScaleObservation(
                    ResizeObservation(
                        gym.make('Contra-v0'), 
                        shape=(240, 320), 
                    ),
                    keep_dim=True,
                ),
                skip=config['skip_frames'],
            ), 
            max_episode_steps=config['max_episode_steps']
        )
        self.action_space = MultiDiscrete([3, 3, 2, 2])
        # self.observation_space = self.env.observation_space
        self.observation_space = Box(
            low=0,
            high=1,
            shape=(240, 320, 1),
            dtype=np.float32
        )
        self.current_step = None
        self.total_reward = None

    def step(self, action):
        nes_action = 0
        # region X-Axis
        if action[0]==0:
            nes_action += 64
        elif action[0]==1:
            nes_action += 0
        elif action[0]==2:
            nes_action += 128
        else:
            raise Exception('Invalid x-axis direction')
        # endregion 

        # region Y-Axis
        if action[1]==0:
            nes_action += 32
        elif action[1]==1:
            nes_action += 0
        elif action[1]==2:
            nes_action += 16
        else:
            raise Exception('Invalid y-axis direction')
        # endregion
        
        # region A
        if action[2]==0:
            nes_action += 0
        elif action[2]==1:
            nes_action += 1
        else:
            raise Exception('Invalid A button')
        # endregion

        # region B
        if action[3]==0:
            nes_action += 0
        elif action[3]==1:
            nes_action += 2
        else:
            raise Exception('Invalid B button')
        # endregion

        # if ((self.current_step > self.highest_step_count) and (self.current_step % 60 == 0)) or (self.current_step > 120 and self.current_step == self.last_restore_step_count - 120) or (self.current_step > 180 and self.current_step == self.last_death_step_count - 180):
        #     if (self.current_step > self.highest_step_count) and (self.current_step % 60 == 0):
        #         logging.info('Saving state at step ' + str(self.current_step) + ' due to progress from previous highest step count ' + str(self.highest_step_count))
        #         self.highest_step_count = self.current_step
        #     elif self.current_step > 120 and self.current_step == self.last_restore_step_count - 120:
        #         logging.info('Saving state at step ' + str(self.current_step) + ' due to backtracking from previous restore step count ' + str(self.last_restore_step_count))
        #     else:
        #         logging.info('Saving state at step ' + str(self.current_step) + ' due to death at step count ' + str(self.last_death_step_count))
            
        #     self.env.backup()
        #     self.last_restore_step_count = self.current_step
        #     self.state_total_reward = self.total_reward
        #     self.times_restored = 0
        
        # f = io.BytesIO()
        # with stdout_redirector(f):
        state, reward, done, info = self.env.step(nes_action)
        # state = self.env.ram/255.0
        if action[2]==1:
            reward -= 0.1
        if action[3]==1:
            reward -= 0.1
        # result = f.getvalue().decode('utf-8')
        # if 'failed to execute opcode' in result:
        #     reward = 0
        #     done = True
        #     self.times_restored = 3 * 3 * 2 * 2
        #     logging.warning('failed to execute opcode')
        
        logging.debug('Step: {0:,.0f}, Action: {1}, Reward: {2:,.0f}, Total Reward: {3:,.0f}'.format(self.current_step, str(action), reward, self.total_reward, self.env.horz_scroll_offset, self.env.x_position))
        self.current_step += 1
        self.total_reward += reward
        # if self.total_reward < -5:
            # done = True
        
        return np.float32(state) / 255.0, reward, done, info

    def reset(self):
        # reset base environment
        state = self.env.reset()
        # state = self.env.ram / 255.0
        self.total_reward = 0
        self.current_step = 0
        # if self.env.has_backup:
        #     p_restore = np.clip(1 - np.exp((self.times_restored * 10 / (3 * 3 * 2 * 2)) - 10), 0, 1)
        #     logging.debug('Probability of restore: {0:,.2%} at count {1:,.0f}'.format(p_restore, self.times_restored))
        #     if np.random.choice([True, False], p=[p_restore, 1 - p_restore]):
        #         logging.info('Restoring state at step: ' + str(self.last_restore_step_count) + '. Highest step count: ' + str(self.highest_step_count) + '. Last restore step count: ' + str(self.last_restore_step_count) + '. Last death step count: ' + str(self.last_death_step_count))
        #         self.env.restore()
        #         self.times_restored += 1
        #         self.current_step = self.last_restore_step_count
        #         self.total_reward = self.state_total_reward

        logging.debug('Reset complete')
        return np.float32(state) / 255.0
    
    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        self.env.close()