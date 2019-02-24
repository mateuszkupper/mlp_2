'''
saves ~ 200 episodes generated from a random policy
'''

import numpy as np
import random
import os
import gym
from gym_torcs import TorcsEnv
import snakeoil3_gym as snakeoil3
from model import make_model
import argparse
import time

MAX_FRAMES = 1000 # max length of carracing
MAX_TRIALS = 200 # just use this to extract one trial. 

render_mode = False # for debugging.

parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                'using pepg, ses, openes, ga, cma'))
parser.add_argument('--port', type=int, default=1, help='port')
args = parser.parse_args()

DIR_NAME = 'record'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

port= 3001
client = snakeoil3.Client(p=port, vision=False)  # Open new UDP in vtorcs
client.MAX_STEPS = np.inf

if port<3006:
  time.sleep(5)

client.get_servers_input(0)  # Get the initial input from torcs

model = make_model(load_model=False)

total_frames = 0
model.make_env(client, render_mode=render_mode, full_episode=True)
#obs = client.S.d  # Get the current full-observation from torcs
#ob = env.make_observation(obs)
for trial in range(MAX_TRIALS): # 200 trials per worker
  try:
    random_generated_int = random.randint(0, 2**31-1)
    filename = DIR_NAME+"/"+str(random_generated_int)+".npz"
    recording_obs = []
    recording_action = []
    #np.random.seed(random_generated_int)
    #model.env.seed(random_generated_int)

    # random policy
    model.init_random_model_params(stdev=np.random.rand()*0.01)

    model.reset()
    #obs, _ = model.env.reset(client) # pixels
    obs={}
    while obs=={}:    
      obs = client.S.d  # Get the current full-observation from torcs
    
    dictlist = []
    for key, value in obs.items():
        if key == 'opponents' or key == 'track' or key == 'wheelSpinVel' or key == 'focus':
            dictlist = dictlist + value
        else:
            dictlist.append(value)
   
    #ob = model.env.make_observation(obs)
    for frame in range(MAX_FRAMES):
      '''
      if render_mode:
        model.env.render("human")
      else:
        model.env.render("rgb_array")
      '''
      print(port)
      recording_obs.append(dictlist)
      z, mu, logvar = model.encode_obs(obs)
      action = model.get_action(z)
      action[1] = action[1] + 0.2
      action[0] = action[0] - 0.1
      recording_action.append(action)
      obs, reward, done, info = model.env.step(frame, client, action, 1)
      dictlist = []
      for key, value in obs.items():
        if key == 'opponents' or key == 'track' or key == 'wheelSpinVel' or key == 'focus':
          dictlist = dictlist + value
        else:
          dictlist.append(value)
      if done:
        break

    total_frames += (frame+1)
    print("dead at", frame+1, "total recorded frames for this worker", total_frames)
    recording_obs = np.array(recording_obs, dtype=np.uint8)
    recording_action = np.array(recording_action, dtype=np.float16)
    np.savez_compressed(filename, obs=recording_obs, action=recording_action)
    if 'termination_cause' in info.keys() and info['termination_cause']=='hardReset':
        print('Hard reset by some agent')
        #ob, client = model.env.reset(client=client) 
    else:
        ob, client = model.env.reset(client=client, relaunch=True)
  except gym.error.Error:
    print("stupid gym error, life goes on")
    model.env.close()
    model.make_env(render_mode=render_mode)
    continue
model.env.close()
