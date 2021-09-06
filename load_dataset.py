import cv2
import numpy as np
import torch
from glob import glob

from agents.cilrs.cilrs_wrapper import CilrsWrapper
from agents.cilrs.models.utils.dataset import CilrsDataset
from agents.cilrs.cilrs_agent import CilrsAgent


env_wrapper = CilrsWrapper(
    acc_as_action=True,
    input_states=["speed", 'vec', 'cmd'],
    view_augmentation=False,
    value_as_supervision=True,
    value_factor=1.0,
    action_distribution='beta_shared',
    dim_features_supervision=256,
    im_mean=[0.485, 0.456, 0.406],
    im_std=[0.229, 0.224, 0.225],
    im_stack_idx=[-1],
#    speed_factor= 12.0,
)

h5_path = ['/scratch3/datasets/roach/lb_data/expert/0003.h5']

dataset = CilrsDataset(list_expert_h5=h5_path, list_dagger_h5=[], env_wrapper=env_wrapper)
