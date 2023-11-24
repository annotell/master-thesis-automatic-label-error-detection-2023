"""Faster RCNN with ResNet50-FPN, 1x schedule."""

_base_ = [
    "../models/faster_rcnn_r50_fpn.py",
    "../datasets/shift.py",
    "../schedules/schedule_1x.py",
    "../default_runtime.py",
]
# load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_1x_det_bdd100k.pth"
# load_from = "pths/faster_rcnn_r50_fpn_1x_det_bdd100k.pth"
load_from = ""
