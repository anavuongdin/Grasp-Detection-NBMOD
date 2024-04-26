import numpy as np
import os
import torch
from torch import nn
from model import get_model
import cv2
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

from grasp_detect_multibox import *

from shapely.geometry import Polygon
from shapely.geometry import LineString, Point
from shapely.affinity import rotate


def grasp_quality(grasp, convex_boundary):
    _, x, y, h, w, angle = grasp

    # Define the line segment and rotate the line segment by theta degrees around the given point
    line = LineString([(x-h/2, y), (x+h/2, y)])
    rotated_line = rotate(line, angle, origin=(x, y))

    # Get intersection points
    intersection_points = convex_boundary.intersection(rotated_line)
    quality = 0
    found = 0
    points = convex_boundary.exterior.coords

    for point in intersection_points.coords:
        edge_line = None
        for i,j in zip(points, points[1:]):
            if LineString((i,j)).distance(Point(point)) < 0.001:
                edge_line = LineString((i,j))
                break

        if edge_line is not None:
            found += 1
            vector1 = np.array([intersection_points.coords[1][0] - intersection_points.coords[0][0], intersection_points.coords[1][1] - intersection_points.coords[0][1]])
            vector2 = np.array([edge_line.coords[1][0] - edge_line.coords[0][0], edge_line.coords[1][1] - edge_line.coords[0][1]])

            # Normalize the vectors
            vector1 /= np.linalg.norm(vector1)
            vector2 /= np.linalg.norm(vector2)

            # Calculate the dot product of the vectors
            dot_product = np.dot(vector1, vector2)

            # Calculate the angle in radians
            angle_rad = np.arccos(dot_product)
            quality += np.absolute(np.sin(angle_rad))

    if found == 1:
        quality = 0
    elif found == 0:
        quality = -1

    # Calculate the distance between the center point
    center = convex_boundary.centroid
    distance = center.distance(rotated_line) + 1e-5
    quality = quality / distance

    return quality

def get_best_grasp(masks, boxes):
    boxes = boxes.detach().cpu()
    indices = np.argwhere(masks.T == 1)
    polygon = Polygon(indices.tolist())
    convex_boundary = polygon.convex_hull

    for grasp in boxes:
        quality = grasp_quality(grasp, convex_boundary)
        grasp[0] = quality


    negative_grasps = list(filter(lambda x: x[0] <= 1e-5, boxes))
    positive_grasps = list(filter(lambda x: x[0] > 0, boxes))
    positive_grasps = sorted(positive_grasps, key=lambda x: x[0], reverse=True)

    if len(positive_grasps) < 1:
        return None, None
    
    if len(negative_grasps) < 1:
        negative_grasps = [torch.rand(6)]

    negative_grasps = torch.cat(negative_grasps).reshape(-1, 6)
    positive_grasps = torch.cat(positive_grasps).reshape(-1, 6)

    return negative_grasps, positive_grasps



def load_grasp_model(ragt_weights_path='weights/RAGT-3-3.pth'):
    device = "cuda"

    inference_multi_image = DetectMultiImage(device=device, weights_path=ragt_weights_path)

    print("Successfully loaded grasp detection model!")
    return inference_multi_image

def detect_grasp(inference_multi_image, image, masks):
    device = "cuda"
    width = 416
    height = 416
    dim = (width, height)

    np_image = np.array(image)
    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    np_image = cv2.resize(np_image, dim, interpolation = cv2.INTER_AREA)
    result = cv2.bitwise_and(np_image, np_image, mask=masks)
    img = torch.from_numpy(result).permute(2, 0, 1).float().unsqueeze(0).to(device)

    boxes = inference_multi_image(img, 0.99)

    negative_grasps, positive_grasps = get_best_grasp(masks, boxes)
    
    if positive_grasps is None:
        return None

    # Setup handle for saving grasp file
    # neg_grasp_fn = os.path.join(neg_grasp_dir, part_id + '.pt')
    # pos_grasp_fn = os.path.join(pos_grasp_dir, part_id + '.pt')
    # torch.save(negative_grasps, neg_grasp_fn)
    # torch.save(positive_grasps, pos_grasp_fn)

if __name__ == '__main__':
    image_fn = "sample/0ba05c786580d26941b01d3a05ae70c75272a2bdd03df13c7c696fd68f158dc8.jpg"
    image = Image.open(image_fn)

    mask_fn = "sample/0ba05c786580d26941b01d3a05ae70c75272a2bdd03df13c7c696fd68f158dc8_0_0.npy"
    masks = np.load(mask_fn)


    inference_multi_image = load_grasp_model(ragt_weights_path='weights/RAGT-3-3.pth')
    detect_grasp(inference_multi_image, image, masks)