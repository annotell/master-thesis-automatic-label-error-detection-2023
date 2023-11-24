"""Dataset settings."""

dataset_type = "BDD100KDetDataset"  # pylint: disable=invalid-name
# data_root = "../data/bdd100k/"  # pylint: disable=invalid-name
k = 1
data_root = f"{data_dirpath}/bdd100k/bdd100k_images_100k/{k}/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Resize",
        img_scale=[
            (1280, 600),
            (1280, 624),
            (1280, 648),
            (1280, 672),
            (1280, 696),
            (1280, 720),
        ],
        multiscale_mode="value",
        keep_ratio=True,
    ),
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
        img_scale=(1280, 720),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"])
        ],
    )
]
anno_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    # dict(
    #     type="Resize",
    #     img_scale=(1280, 720),
    #     keep_ratio=True,
    # ),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1280, 720),
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
    # dict(type="RandomFlip", flip_ratio=0.5),
    # dict(type="Normalize", **img_norm_cfg),
    # dict(type="Pad", size_divisor=32),
    # dict(type="ImageToTensor", keys=["img"]),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
    )
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + f"fold{k}_train.json",
        img_prefix=data_root + "train",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + f"fold{k}_val.json",
        img_prefix=data_root + "val",  # should be in train dir?
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + f"fold{k}_val.json",
        img_prefix=data_root + "val", # should be in train dir?
        pipeline=test_pipeline,
    ),
    offical_val=dict(
        type=dataset_type,
        ann_file=f"{data_dirpath}/bdd100k/bdd100k_images_100k/labels_coco2/val_cocofmt.json",
        img_prefix="/media/18T/data_thesis/bdd100k/bdd100k_images_100k/images/100k/val",
        # pipeline=test_pipeline,
        pipeline=anno_pipeline,
    ),
    small=dict(
        type=dataset_type,
        ann_file=f"/media/18T/data_thesis/bdd100k/bdd100k_images_100k/0/val_small_val.json",
        img_prefix="/media/18T/data_thesis/bdd100k/bdd100k_images_100k/0/val_small/images",
        # pipeline=test_pipeline,
        pipeline=anno_pipeline,
    ),
)
evaluation = dict(interval=1, metric="bbox")
