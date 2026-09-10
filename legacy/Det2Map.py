import os
import cv2
import numpy as np
import cupy as cp
import pandas as pd
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
from typing import List, Optional
import copy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.append('../DEPTH/Depth-Anything/metric_depth')
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from torchvision.transforms import ToTensor


def plot_dets_polar(cartesian_detections, θ, FOV, classes, color_pallete):    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(10, 10))

    # Rotate the subplot by 90 degrees and invert angle display to clockwise
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
       
    # Sector     
    theta = np.linspace(np.deg2rad(θ-FOV/2), np.deg2rad(θ+FOV/2), 100)
    r = 100 * np.ones_like(theta)  
    theta = np.append(theta, [0, theta[0]])
    r = np.append(r, [0,r[0]])
    ax.plot(theta, r, color='gray', linewidth=2)  # Plotting the sector
    ax.fill(theta, r, color='lightgray', alpha=0.5)

    # Detections
    r_array = [np.linalg.norm(det[:2]) for det in cartesian_detections.xyxy]
    theta_array = [np.arctan2(det[0], det[1]) for det in cartesian_detections.xyxy]
    color_array = [color_pallete[class_id] for class_id in cartesian_detections.class_id]
    area = 0.25 * np.array(r_array)**2

    ax.scatter(theta_array, r_array, c=color_array, s=area, alpha=0.75)

    ax.set_ylim(0, 100)
    ax.set_yticks([ 20, 40, 60, 80, 100])
    ax.set_yticklabels(['20 m', '40 m', '60 m', '80 m', '100 m'])


    # legend
    plt.legend(handles=[mpatches.Patch(color=color, label=categoria) for color, categoria in zip(color_pallete, classes)],
            loc='lower right', fontsize='small', bbox_to_anchor=(1.1, -0.1))

    fig.canvas.draw()
    
    figure = np.array(fig.canvas.renderer.buffer_rgba())
    
    plt.close()
    
    return figure 


def dets2cartesian2D(detections, θ, FOV, detections_metric_depth, im_width):    
    dets = copy.deepcopy(detections)
    FOV = FOV * np.pi/180
    θ = θ * np.pi/180
    for i in range(len(dets.xyxy)):        
        x1, _, x2, _ = dets.xyxy[i][:4]
        r = detections_metric_depth[i]         
        dets.xyxy[i] = np.array([r*np.sin(FOV*x1/im_width + θ - FOV/2), r*np.cos(FOV*x1/im_width + θ - FOV/2),
                                 r*np.sin(FOV*x2/im_width + θ - FOV/2), r*np.cos(FOV*x2/im_width + θ - FOV/2)])
    return dets


def render_detections(img, detections, names, show_conf: bool = False) -> np.ndarray:
    box_annotator = sv.BoxAnnotator(text_scale=0.5, text_padding=2)
    if show_conf:
        labels = [
            f"{names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections
        ]
    else:
        labels = [
            f"{names[class_id]}"
            for _, _, _, class_id, _, _
            in detections
        ]
    annotated_frame = box_annotator.annotate(
        scene=img.copy(),
        detections=detections,
        labels=labels
    )    
    return annotated_frame


color_pallete = [   [0.6392156862745098, 0.3176470588235294, 0.984313725490196],
                    [0.9019607843137255, 0.09803921568627451, 0.29411764705882354],
                    [0.23529411764705882, 0.7058823529411765, 0.29411764705882354],
                    [1.0, 0.8823529411764706, 0.09803921568627451],
                    [0.0, 0.5098039215686274, 0.7843137254901961]]

classes = ['Militar', 'Vehículo Militar', 'Civil', 'Vehículo Civil', 'Perro']

df = pd.read_pickle('/home/gprietos/ACHILE/Gafas360/planarImgViewer_app/video_data.pkl')
video = cv2.VideoCapture("/home/gprietos/ACHILE/Gafas360/planarImgViewer_app/video_planar.mp4")
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
output_video = cv2.VideoWriter("video_planar_full_infer_diagram_2.mp4", fourcc, fps, (2160, 1080))

FOV = 90

model = YOLO("./best.pt")

depth_model_path = "local::/home/gprietos/ACHILE/DEPTH/Depth-Anything/metric_depth/checkpoints/depth_anything_metric_depth_outdoor.pt"
conf = get_config(model_name="zoedepth",
                  mode ="infer",
                  pretrained_resource="local::/home/gprietos/ACHILE/DEPTH/Depth-Anything/metric_depth/checkpoints/depth_anything_metric_depth_outdoor.pt",
                  dataset="kitti")
depth_anything_model = build_model(conf).cuda()

with tqdm(total=total_frames) as pbar:
    while True:
        ret, planarImg = video.read()
        
        if not ret:
            break  
                        
        pred = model(planarImg, conf=0.6, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(pred)
        
        if detections:
            metric_depth = depth_anything_model.infer(ToTensor()(planarImg).unsqueeze(0).cuda()).squeeze().detach().cpu().numpy()
            detections_metric_depth = [metric_depth[int((y1 + y2) / 2), int((x1 + x2) / 2)] for x1, y1, x2, y2 in detections.xyxy] 
            cartesian_detections = dets2cartesian2D(detections,
                                                    df[int(video.get(cv2.CAP_PROP_POS_FRAMES)-1)]["yaw"],
                                                    FOV,
                                                    detections_metric_depth,
                                                    planarImg.shape[1]) 

        frame_infer = render_detections(planarImg, detections,pred.names,show_conf=True) 
        
        frame_diagram = cv2.cvtColor(plot_dets_polar(cartesian_detections,
                                                     df[int(video.get(cv2.CAP_PROP_POS_FRAMES)-1)]["yaw"],
                                                     FOV, 
                                                     classes, 
                                                     color_pallete),
                                     cv2.COLOR_RGBA2BGR)

        joint_frame = np.hstack((frame_infer, cv2.resize(frame_diagram, (1080, 1080))))

        output_video.write(joint_frame)
        pbar.update(1)

    output_video.release()