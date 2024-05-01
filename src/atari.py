import os
import numpy as np
import ale_py
from ale_py import ALEInterface
import time
import cv2
import sys

class Atari:
  def __init__(self, rom_dir, frame_skip=4):
    self.ale = ALEInterface()

    # Set Settings
    self.ale.setInt("random_seed", 123)
    self.frame_skip = frame_skip
    self.ale.setInt("frame_skip", self.frame_skip)
    self.ale.setBool("display_screen", False)
    self.ale.setFloat('repeat_action_probability', 0.0)  # Disable sticky actions
    # self.ale.setBool("sound", True)
    self.record_sound_for_user = False
    # self.ale.setBool("record_sound_for_user", self.record_sound_for_user) 


    self.ale.loadROM(rom_dir)
    self.screen_width, self.screen_height = self.ale.getScreenDims()
    self.legal_actions = self.ale.getLegalActionSet()
    self.n_actions = len(self.legal_actions)

    self.reset()
  
  def get_observation(self):
    observation = np.zeros(self.screen_width*self.screen_height*3, dtype=np.uint8)
    observation = self.ale.getScreenRGB()
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    return np.reshape(observation, (self.screen_height, self.screen_width, 1))
  
  def reset(self):
    self.ale.reset_game()
    return self.get_observation(), None

  def step(self, action):
    reward = self.ale.act(action)
    observation = self.get_observation()
    terminated = self.ale.game_over()
    return observation, reward, terminated, None, None

  # Not being used currently
  def get_image_and_audio(self):
    np_data_image = np.zeros(self.screen_width*self.screen_height*3, dtype=np.uint8)
    if self.record_sound_for_user:
        np_data_audio = np.zeros(self.ale.getAudioSize(), dtype=np.uint8)
        self.ale.getScreenRGBAndAudio(np_data_image, np_data_audio)
        np_data_image = cv2.cvtColor(np_data_image, cv2.COLOR_BGR2GRAY)

        # Also supports independent audio queries if user desires:
        #  self.ale.getAudio(np_data_audio)
        return np.reshape(np_data_image, (self.screen_height, self.screen_width)), np.asarray(np_data_audio)
    else:
        self.ale.getScreenRGB(np_data_image)
        np_data_image = cv2.cvtColor(np_data_image, cv2.COLOR_BGR2GRAY)
        return np.reshape(np_data_image, (self.screen_height, self.screen_width))
