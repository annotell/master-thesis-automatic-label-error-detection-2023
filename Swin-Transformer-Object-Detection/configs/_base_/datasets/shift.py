"""Dataset settings."""

dataset_type = "SHIFTDataset"
data_root = f"{data_dirpath}/SHIFT"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(1280, 800), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1280, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
anno_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=False),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1280, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            # dict(type="Collect", keys=["img"]),
        ],
    ),
    dict(
        type="Collect",
        # keys=["img", "gt_bboxes", "gt_labels"],
        keys=["img", "gt_bboxes", "gt_labels"],
    )
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=f"{data_root}/train/Two_busmo_uni1.json",
        img_prefix=f'{data_root}/train/RGB_stereo',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=f"{data_root}/val/Two_busmo_uni1.json",
        img_prefix=f'{data_root}/val/RGB_stereo',
        pipeline=anno_pipeline,
    ),
)

evaluation = dict(interval=8, metric="mAP")
