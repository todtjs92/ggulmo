import torch
import pickle

model = torch.load('/home/jungeui/Documents/workspace/ggulmo/models/FM/FM_best_model.pt')


print(model['linear.fc.weight'])
print(len(model['linear.fc.weight']))

# 
model['linear.fc.weight'][-3] = model['linear.fc.weight'][-1]
print(model['linear.fc.weight'])

with open('../../data/state_dict.pickle','rb') as f:
    state_dict = pickle.load(f)
print(state_dict)