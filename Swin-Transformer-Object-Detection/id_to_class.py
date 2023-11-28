bdd100k_det = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motorcycle",
    7: "bicycle",
    8: "traffic light",
    9: "traffic sign",
    10: "bg",
}


shift_det = {
    0: "pedestrian",
    1: "car",
    2: "truck",
    3: "bus",
    4: "motorcycle",
    5: "bicycle",
    6: "bg"
}


shift_det_name2RGB = {
    "pedestrian": (245, 245, 245),
    "car": (197, 202, 233),
    "truck": (155, 249, 196),
    "bus": (178, 235, 242),
    "motorcycle": (255, 205, 210),
    "bicycle": (255, 224, 178),
    "bg": (244, 67, 54)
}

shift_det_name2red = {
    "pedestrian": (255, 0, 0),
    "car": (255, 0, 0),
    "truck": (255, 0, 0),
    "bus": (255, 0, 0),
    "motorcycle": (255, 0, 0),
    "bicycle": (255, 0, 0),
    "bg": (255, 0, 0)
}

shift_det_name2white = {
    "pedestrian": (255, 255, 255),
    "car": (255, 255, 255),
    "truck": (255, 255, 255),
    "bus": (255, 255, 255),
    "motorcycle": (255, 255, 255),
    "bicycle": (255, 255, 255),
    "motorcycle": (255, 255, 255),
    "bicycle": (255, 255, 255),
    "bg": (255, 255, 255)
}

# https://materialui.co/colors/
# https://www.rapidtables.com/convert/color/hex-to-rF44336gb.html
bdd100k_det_name2RGB = {
    "pedestrian": (245, 245, 245),
    "rider": (197, 202, 233),
    "car": (155, 249, 196),
    "truck": (178, 235, 242),
    "bus": (255, 205, 210),
    "train": (255, 224, 178),
    "motorcycle": (215, 204, 200),
    "bicycle": (225, 190, 231),
    "traffic light": (200, 230, 201),
    "traffic sign": (248, 187, 208),
    "bg": (244, 67, 54)
}
bdd100k_det_name2red = {
    "pedestrian": (255, 0, 0),
    "rider": (255, 0, 0),
    "car": (255, 0, 0),
    "truck": (255, 0, 0),
    "bus": (255, 0, 0),
    "train": (255, 0, 0),
    "motorcycle": (255, 0, 0),
    "bicycle": (255, 0, 0),
    "traffic light": (255, 0, 0),
    "traffic sign": (255, 0, 0),
    "bg": (255, 0, 0)
}

bdd100k_det_name2white = {
    "pedestrian": (255, 255, 255),
    "rider": (255, 255, 255),
    "car": (255, 255, 255),
    "truck": (255, 255, 255),
    "bus": (255, 255, 255),
    "train": (255, 255, 255),
    "motorcycle": (255, 255, 255),
    "bicycle": (255, 255, 255),
    "traffic light": (255, 255, 255),
    "traffic sign": (255, 255, 255),
    "bg": (255, 255, 255)
}

bdd100k_det_name2white = {
    "pedestrian": (255, 255, 255),
    "rider": (255, 255, 255),
    "car": (255, 255, 255),
    "truck": (255, 255, 255),
    "bus": (255, 255, 255),
    "train": (255, 255, 255),
    "motorcycle": (255, 255, 255),
    "bicycle": (255, 255, 255),
    "traffic light": (255, 255, 255),
    "traffic sign": (255, 255, 255),
    "bg": (255, 255, 255)
}

bdd100k_det_name2HEX = {
    "pedestrian": 'F5F5F5',
    "rider": "C5CAE9",
    "car": "FFF9C4",
    "truck": "B2EBF2",
    "bus": "FFCDD2",
    "train": "FFE0B2",
    "motorcycle": "D7CCC8",
    "bicycle": "E1BEE7",
    "traffic light": "C8E6C9",
    "traffic sign": "F8BBD0",
    "bg": "F44336"
}

gtsrb_classes = {
    0:'Speed limit 20km',
    1:'Speed limit 30km', 
    2:'Speed limit 50km', 
    3:'Speed limit 60km', 
    4:'Speed limit 70km', 
    5:'Speed limit 80km', 
    6:'End of speed limit 80km', 
    7:'Speed limit 100km', 
    8:'Speed limit 120km', 
    9:'No passing', 
    10:'No passing veh over 3.5 tons', 
    11:'Right-of-way at intersection', 
    12:'Priority road', 
    13:'Yield', 
    14:'Stop', 
    15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 
    17:'No entry', 
    18:'General caution', 
    19:'Dangerous curve left', 
    20:'Dangerous curve right', 
    21:'Double curve', 
    22:'Bumpy road', 
    23:'Slippery road', 
    24:'Road narrows on the right', 
    25:'Road work', 
    26:'Traffic signals', 
    27:'Pedestrians', 
    28:'Children crossing', 
    29:'Bicycles crossing', 
    30:'Beware of ice snow',
    31:'Wild animals crossing', 
    32:'End speed + passing limits', 
    33:'Turn right ahead',
    34:'Turn left ahead', 
    35:'Ahead only', 
    36:'Go straight or right', 
    37:'Go straight or left', 
    38:'Keep right', 
    39:'Keep left', 
    40:'Roundabout mandatory', 
    41:'End of no passing', 
    42:'End no passing veh > 3.5 tons'
}


swedish_classes = {
    0: '100_SIGN', 
    1: '110_SIGN', 
    2: '30_SIGN', 
    3: '50_SIGN', 
    4: '60_SIGN', 
    5: '70_SIGN', 
    6: '80_SIGN', 
    7: '90_SIGN', 
    8: 'GIVE_WAY', 
    9: 'NO_PARKING', 
    10: 'NO_STOPPING_NO_STANDING', 
    11: 'OTHER', 
    12: 'PASS_EITHER_SIDE', 
    13: 'PASS_RIGHT_SIDE', 
    14: 'PEDESTRIAN_CROSSING', 
    15: 'PRIORITY_ROAD', 
    16: 'STOP'
}


coco_shiftall_det = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "person",
    4: "motorcycle",
    5: "bicycle"
}


coco_det = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
    }

coco_det_origin = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'}