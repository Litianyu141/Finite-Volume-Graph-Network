import torch.utils.data as data
import numpy as np
import torch


class SwimmerDataset(data.Dataset):
    def __init__(self, path):
        self.data = np.load(path)

    def __len__(self):
        return self.data.shape[0] * (self.data.shape[1] - 2)

    def __getitem__(self, idx):
        episode = idx // (self.data.shape[1] - 2)
        frame = idx % (self.data.shape[1] - 2) + 1
        #print(episode, frame)

        last_state = self.data[episode, frame - 1,5:]
        this_state = self.data[episode, frame,5:]
        action = self.data[episode, frame, :5]

        pos = last_state[5:5 + 18].reshape(6, 3)
        #pos += np.random.normal(scale = 0.001, size = pos.shape)
        last_state[5:5 + 18] = pos.reshape(18,)

        delta_state = this_state - last_state
        delta_state[delta_state > np.pi] -= np.pi * 2
        delta_state[delta_state < -np.pi] += np.pi * 2

        return action, delta_state, last_state
    
    
    def __get_episode__(self, idx):
        episode = idx 
        #print(episode, frame)
        
        actions = []
        delta_states = []
        last_states = []
        
        for frame in range(10,110):
        
            last_state = self.data[episode, frame - 1,5:]
            this_state = self.data[episode, frame,5:]
            action = self.data[episode, frame, :5]

            pos = last_state[5:5 + 18].reshape(6, 3)
            #pos += np.random.normal(scale = 0.001, size = pos.shape)
            last_state[5:5 + 18] = pos.reshape(18,)

            delta_state = this_state - last_state
            delta_state[delta_state > np.pi] -= np.pi * 2
            delta_state[delta_state < -np.pi] += np.pi * 2

            actions.append(action)
            delta_states.append(delta_state)
            last_states.append(last_state)
            
        
        actions = np.array(actions)
        delta_states = np.array(delta_states)
        last_states = np.array(last_states)

        return actions, delta_states, last_states