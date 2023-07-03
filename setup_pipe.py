import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import numpy

from diffusers import StableDiffusionPipeline
import torch

from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.refcoco import RefcocoTask
from OFA.models.ofa import OFAModel
from PIL import Image

from shapely.geometry import Polygon
import glob
from torch import nn
import torchvision
from PIL import Image

from grasp_detect_multibox import *

from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.refcoco import RefcocoTask
from models.ofa import OFAModel
from PIL import Image
from matplotlib import cm

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


# Setup Stable Diffusion
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Setup SAM
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)