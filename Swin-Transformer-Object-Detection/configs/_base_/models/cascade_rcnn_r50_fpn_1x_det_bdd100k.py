"""Cascade RCNN with ResNet50-FPN, 1x schedule."""

_base_ = [
    "../models/cascade_rcnn_r50_fpn.py",
    "../datasets/bdd100k.py",
    "../schedules/schedule_1x.py",
    "../default_runtime.py",
]
# load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/cascade_rcnn_r50_fpn_1x_det_bdd100k.pth"
# load_from = '/home/belay/Documents/0_UPPSALA/KOGNIC_research/Codes/Swin-Transformer-Object-Detection/pths/cascade_rcnn_r50_fpn_1x_det_bdd100k.pth'
load_from = ''