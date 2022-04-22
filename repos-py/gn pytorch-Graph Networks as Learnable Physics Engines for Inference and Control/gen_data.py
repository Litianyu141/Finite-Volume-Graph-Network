from dm_control import suite
import myswimmer as swimmer
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy import interpolate
from tqdm import tqdm
import sys


# Load one task:
env = swimmer.swimmer6(time_limit = 4) #suite.load(domain_name="swimmer", task_name="random")

# Iterate over a task set:
#for domain_name, task_name in suite.BENCHMARKING:
#  env = suite.load(domain_name, task_name)

# Step through an episode and print out reward, discount and observation.

max_frame = 200
max_episodes = int(sys.argv[1])

width = 480
height = 480

action_spec = env.action_spec()
time_step = env.reset()

actions = np.zeros((201,5))
x = np.arange(0,201,20)

dataset = np.zeros((max_episodes, max_frame+1, 5 + 5 + 18 + 18))
len_dataset = dataset.shape[0]

for idx in tqdm(range(len_dataset)):
    time_step = env.reset()
    video = np.zeros((max_frame, height, width, 3), dtype=np.uint8)
    i = 0
    for j in range(5):
        y = np.random.uniform(-1,1,x.shape)
        tck = interpolate.splrep(x, y, s=0)
        xnew = np.arange(0,201)
        ynew = interpolate.splev(xnew, tck, der=0)

        actions[:,j] = ynew

    actions = np.clip(actions, -1, 1)
    record = False

    while not time_step.last():
        action = actions[i]
        time_step = env.step(action)
        #from IPython import embed; embed()
        obs = time_step.observation
        #print(obs)
        dataset[idx,i,:5] = action
        dataset[idx, i, 5:10] = obs['joints']
        dataset[idx,i,10:28] = obs['abs']
        dataset[idx,i,28:] = obs['body_velocities']

        if record:
            if i < max_frame:
                video[i] = env.physics.render(height, width, camera_id=0)
        i += 1

    if record:
        writer = imageio.get_writer('test_{}.gif'.format(idx), fps=60)
        for j in range(max_frame):
            writer.append_data(video[j])
        writer.close()

np.save(sys.argv[2], dataset)

'''
plt.plot(actions)
plt.savefig('actions.png')
'''
