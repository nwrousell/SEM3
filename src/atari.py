import os
import numpy as np
from ale_python_interface import ALEInterface
import time
import cv2
import shutil
import scipy.io.wavfile as wavfile
import scipy.misc
from python_speech_features import mfcc
import matplotlib.pyplot as plt
import subprocess as sp
import imageio
from skimage import transform

class Atari:
  def __init__(self, rom_dir, frame_skip=4):
    self.ale = ALEInterface()

    # Set Settings
    self.ale.setInt(b"random_seed", 123)
    self.frame_skip = frame_skip
    self.ale.setInt(b"frame_skip", self.frame_skip)
    self.ale.setBool(b"display_screen", False)
    self.ale.setFloat(b"repeat_action_probability", 0.0)  # Disable sticky actions
    self.ale.setBool(b"sound", True)
    self.record_sound_for_user = True
    self.ale.setBool(b"record_sound_for_user", self.record_sound_for_user) 


    self.ale.loadROM(str.encode(rom_dir))
    self.screen_width, self.screen_height = self.ale.getScreenDims()
    self.legal_actions = self.ale.getLegalActionSet()
    self.n_actions = len(self.legal_actions)

    self.reset()
  
  def get_observation(self):
    observation = np.zeros(self.screen_width*self.screen_height*3, dtype=np.uint8)
    observation = self.ale.getScreenGrayscale()
    # audio = self.ale.getAudio()
    # print(audio)
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

        # Also supports independent audio queries if user desires:
        #  self.ale.getAudio(np_data_audio)
    else:
        #  np_data_audio = 0
        np_data_audio = np.zeros(self.ale.getAudioSize(), dtype=np.uint8)
        self.ale.getAudio(np_data_audio)
        self.ale.getScreenRGB(np_data_image)

    return np.reshape(np_data_image, (self.screen_height, self.screen_width, 3)), np.asarray(np_data_audio)

# class AtariMonitor:
    # def __init__(self, env, video_dir='videos'):
    #     self.env = env
    #     self.video_dir = video_dir
    #     video_path = f"{self.video_dir}.mp4"
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (self.env.screen_width, self.env.screen_height))

    # def record_frame(self):
    #     frame = self.env.getScreenRGB()
    #     self.video_writer.write(frame)

    # def reset(self):
    #     observation, _ = self.env.reset()
    #     self.video_writer.release()
    #     return observation

    # def step(self, action):
    #     observation, reward, terminated, _, _ = self.env.step(action)
    #     self.record_frame()
    #     return observation, reward, terminated

class AtariMonitor:
  def __init__(self, env, video_dir='../videos/'):
    self.env = env
    self.video_dir = video_dir
    self.framerate = 60 # Should read from ALE settings technically
    self.samples_per_frame = 512 # Should read from ALE SoundExporter class technically
    self.audio_freq = self.framerate*self.samples_per_frame #/ self.env.frame_skip
    self.action_count = 0
    self.episode_count = 0
    self.all_audio = np.zeros((0, ), dtype=np.uint8)
    
    
    self.save_dir_av = video_dir + '/logs_av_seq' # Save png sequence and audio wav file here
    self.save_dir_movies = video_dir + '/logs_movies'
    self.save_image_prefix = 'im_frames_'
    self.save_audio_filename = 'audio.wav'
    self.create_save_dir(self.save_dir_av)
  
  def reset(self):
    self.save_audio(self.all_audio)
    self.save_movie("run_" + str(time.time()))
    return self.env.reset()

  def step(self, action):
    observation, reward, terminated, _, _ = self.env.step(action)
    self.action_count += 1
    image, audio = self.env.get_image_and_audio()
    self.all_audio = np.append(self.all_audio, audio)
    audio_mfcc = self.audio_to_mfcc(self.env.ale.getAudio())
    image = self.concat_image_audio(self.env.ale.getScreenRGB(), audio_mfcc)
    # audio = self.env.ale.getAudio()
    # image = self.env.ale.getScreenRGB()
    # self.save_audio(audio)
    self.save_image(image)
    return observation, reward, terminated, None, None
  
  def create_save_dir(self, directory):
      # Remove previous img/audio image logs
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
        

  def audio_to_mfcc(self, audio):
    mfcc_data = mfcc(signal=audio, samplerate=self.audio_freq, winlen=0.002, winstep=0.0006)
    mfcc_data = np.swapaxes(mfcc_data, 0 ,1) # Time on x-axis

    # Normalization 
    min_data = np.min(mfcc_data.flatten())
    max_data = np.max(mfcc_data.flatten())
    mfcc_data = (mfcc_data-min_data)/(max_data-min_data)
      
    return mfcc_data

  def save_image(self, image):
    number = str(self.action_count).zfill(6)
    imageio.imwrite(os.path.join(self.save_dir_av, self.save_image_prefix+number+'.png'), image)
    # scipy.misc.imsave(os.path.join(self.save_dir_av, self.save_image_prefix+number+'.png'), image)

  def save_audio(self, audio):
    wavfile.write(os.path.join(self.save_dir_av, self.save_audio_filename), self.audio_freq, audio)

  def save_movie(self, movie_name):
    # Use ffmpeg to convert the saved img sequences and audio to mp4

    # Video recording
    command = [ "ffmpeg",
                  '-y', # overwrite output file if it exists
                  '-r', str(self.framerate), # frames per second
                  '-i', os.path.join(self.save_dir_av, self.save_image_prefix+'%6d.png') # Video input comes from pngs
              ]

    # Audio if available
    if self.env.record_sound_for_user:
      command.extend(['-i', os.path.join(self.save_dir_av, self.save_audio_filename)]) # Audio input comes from wav

    # Codecs and output
    command.extend(['-c:v', 'libx264', # Video codec
                  '-c:a', 'mp3', # Audio codec
                  os.path.join(self.save_dir_movies, movie_name+'.mp4') # Output dir
                    ])

    # Make movie dir and write the mp4
    if not os.path.exists(self.save_dir_movies):
        os.makedirs(self.save_dir_movies)
    sp.call(command) # NOTE: needs ffmpeg! Will throw 'dir doesn't exist err' otherwise.
    
  def concat_image_audio(self, image, audio_mfcc):
    # Concatenates image and audio to test sync'ing in saved .mp4
    # print(audio_mfcc.shape)
    audio_mfcc = transform.resize(audio_mfcc, (image.shape[0], image.shape[1])) # Resize MFCC image to be same size as screen image
    # print(audio_mfcc.shape)
    cmap = plt.get_cmap('viridis') # Apply a colormap to spectrogram
    # print(cmap(audio_mfcc).shape)
    audio_mfcc = (np.delete(cmap(audio_mfcc), 3, 2)*255.).astype(np.uint8) # Gray MFCC -> 4 channel colormap -> 3 channel colormap
    image = np.concatenate((image, audio_mfcc), axis=1) # Concat screen image and MFCC image
    return image

  # def plot_mfcc(self, audio_mfcc):
  #   plt.clf()
  #   plt.imshow(audio_mfcc, interpolation='bilinear', cmap=plt.get_cmap('viridis'))
  #   plt.pause(0.001)