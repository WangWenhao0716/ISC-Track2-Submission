import torch
mod = torch.load('logs/baseline/50/checkpoint_24.pth.tar',map_location='cpu')
mod['state_dict'].pop('classifier_0.weight')
mod['state_dict'].pop('classifier_1.weight')
mod['state_dict'].pop('classifier_2.weight')
mod['state_dict'].pop('classifier_3.weight')
torch.save(mod['state_dict'], '/dev/shm/baseline_distill_50.pth.tar')
