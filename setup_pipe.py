import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import numpy

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from torchvision import transforms

from fairseq import utils, tasks
from fairseq import checkpoint_utils
from OFA.utils.eval_utils import eval_step
from OFA.tasks.mm_tasks.refcoco import RefcocoTask
from OFA.models.ofa import OFAModel
from PIL import Image

from shapely.geometry import Polygon
import glob
from torch import nn
import torchvision
from PIL import Image

from grasp_detect_multibox import *
from matplotlib import cm

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


# Step 1: Setup Stable Diffusion
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Step 2: Setup SAM
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# Step 3: Setup OFA
# Register refcoco task
tasks.register_task('refcoco', RefcocoTask)

# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False
# Load pretrained ckpt & config
overrides={"bpe_dir":"OFA/utils/BPE"}
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths('OFA/checkpoints/refcocog.pt'),
        arg_overrides=overrides
    )

cfg.common.seed = 7
cfg.generation.beam = 5
cfg.generation.min_len = 4
cfg.generation.max_len_a = 0
cfg.generation.max_len_b = 4
cfg.generation.no_repeat_ngram_size = 3

# Fix seed for stochastic decoding
if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

# Image transform
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()
def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text.lower()),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

# Construct input for refcoco task
patch_image_size = cfg.task.patch_image_size
def construct_sample(image: Image, text: str):
    w, h = image.size
    w_resize_ratio = torch.tensor(patch_image_size / w).unsqueeze(0)
    h_resize_ratio = torch.tensor(patch_image_size / h).unsqueeze(0)
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(' which region does the text " {} " describe?'.format(text), append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        },
        "w_resize_ratios": w_resize_ratio,
        "h_resize_ratios": h_resize_ratio,
        "region_coords": torch.randn(1, 4)
    }
    return sample

# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

width = 416
height = 416
dim = (width, height)

# Step 4: Setup Grasp Generator
def draw_multi_box(img, box_coordinates):
    point_color1 = (255, 255, 0)  # BGR
    point_color2 = (255, 0, 255)  # BGR
    thickness = 2
    lineType = 4
    for i in range(box_coordinates.shape[0]):
        center = (box_coordinates[i, 1].item(), box_coordinates[i, 2].item())
        size = (box_coordinates[i, 3].item(), box_coordinates[i, 4].item())
        angle = box_coordinates[i, 5].item()
        box = cv2.boxPoints((center, size, angle))
        box = np.int64(box)
        cv2.line(img, box[0], box[3], point_color1, thickness, lineType)
        cv2.line(img, box[3], box[2], point_color2, thickness, lineType)
        cv2.line(img, box[2], box[1], point_color1, thickness, lineType)
        cv2.line(img, box[1], box[0], point_color2, thickness, lineType)

transform = torchvision.transforms.Compose([
    transforms.ToTensor(),
])
ragt_weights_path = 'pretrained_weights/RAGT-3-3.pth'
inference_multi_image = DetectMultiImage(device=device, weights_path=ragt_weights_path)


print("Successfully loaded!")


def generate_a_sample(prompt, query, fn):
    """
    prompt: str - natural language to describe the synthesized image
    query: str - object needed to be captured
    fn: str - name of the sample
    """
    # Step 1: Generate the image
    image = pipe(prompt).images[0]
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(fn, image)

    # Step 2: Grounding image
    image = Image.open(fn)
    sample = construct_sample(image, query)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

    # Run eval step for refcoco
    with torch.no_grad():
        result, scores = eval_step(task, generator, models, sample)

    # Step 3: Segmentation
    input_box = np.array(list(map(int, result[0]["box"])))

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    # Step 4: Maskout the image
    masks = np.squeeze(masks).astype('uint8')
    result = cv2.bitwise_and(image, image, mask=masks)

    # Step 5: Defining the grasping pose
    img = torch.from_numpy(result).permute(2, 0, 1).float().unsqueeze(0).to(device)
    boxes = inference_multi_image(img, 0.95)

    return boxes

    # Visualize (optional)
    # img = cv2.imread(fn)
    # draw_multi_box(img, boxes.data)


if __name__ == '__main__':
    generate_a_sample("A keyboard", "keyboard", "test.png")
    print("Done")