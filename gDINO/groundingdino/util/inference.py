import pprint
from typing import Tuple, List

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchvision.ops import box_convert

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap, get_phrases_from_posmap_all


# ----------------------------------------------------------------------------------------------------------------------
# OLD API
# ----------------------------------------------------------------------------------------------------------------------


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [

            # VPTW
            # T.Resize(sizes=(1200, 602)),
            # T.Normalize([0.5102, 0.6078, 0.5801], [0.1578, 0.1895, 0.2130]),

            T.ToTensor(),
            
            # Grounding DINO Default, use for bdd100k and SHIFT
            # T.RandomResize([800], max_size=1333),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def cxcywh2xyxy_ratio(bbox, img_shape):
    img_h, img_w = img_shape
    cx, cy, w, h = bbox
    # x is row
    # y is column
    
    xmin = img_w*(cx - w/2.0)
    ymin = img_h*(cy - h/2.0)
    xmax = img_w*(cx + w/2.0)
    ymax = img_h*(cy + h/2.0)
    
    xmin = 0 if xmin < 0 else xmin
    ymin = 0 if ymin < 0 else ymin
    xmax = xmax if xmax < img_w else img_w
    ymax = ymax if ymax < img_h else img_h
    
    #print('[cxcywh2xyxy_ratio] ', xmin, ymin, xmax, ymax)
    return xmin, ymin, xmax, ymax

def xywh2ratio(img_w, img_h, bbox):
    # bbox_left, bbox_top, bbox_width, bbox_height = bbox

    # cx = (bbox_left + bbox_width/2.0)
    # cy = (bbox_top + bbox_height/2.0)

    # cx = 0 if cx < 0 else cx
    # cx = img_w if cx > 0 else cx
    # cy = 0 if cy < 0 else cy
    # cy = img_h if cy > 0 else cy

    # x_center_ratio = cx/img_w
    # y_center_ratio = cx/img_h
    # w_ratio = bbox_width/img_w
    # h_ratio = bbox_height/img_h

    x, y, w, h = bbox
    x_center = x + w/2.0
    y_center = y + h/2.0
    x_center_ratio = x_center/img_w * 1.0
    y_center_ratio = y_center/img_h * 1.0
    w_ratio = w/img_w * 1.0
    h_ratio = h/img_h * 1.0


    return [x_center_ratio, y_center_ratio, w_ratio, h_ratio]


def crop_bbox(xmin, ymin, xmax, ymax, img, img_shape, pad_ratio):
    img_xmax = img_shape[0]
    img_ymax = img_shape[1]

    w_pad = int((xmax - xmin) * pad_ratio)
    h_pad = int((ymax - ymin) * pad_ratio)

    xmin2 = int(xmin) - w_pad if int(xmin) - w_pad > 0 else 0
    ymin2 = int(ymin) - h_pad if int(ymin) - h_pad > 0 else 0
    xmax2 = int(xmax) + w_pad if int(xmin) + w_pad > 0 else img_xmax
    ymax2 = int(ymax) + h_pad if int(ymax) + h_pad > 0 else img_ymax

    # print(f'img.shape={img.shape}')
    cropped = img[xmin2:xmax2, ymin2:ymax2, :]
    # print(f'cropped.shape={cropped.shape}')
    img_w2 = xmax2 - xmin2
    img_h2 = ymax2 - ymin2

    xmin3 = xmin - xmin2
    ymin3 = ymin - ymin2
    w3 = xmax - xmin
    h3 = ymax - ymin

    bbox = xywh2ratio(img_w2, img_h2, (xmin3, ymin3, w3, h3))

    return cropped, bbox


def crop_bboxes(img, img_shape, gt_bboxes, pad):
    bbox_imgs = []
    bboxes = []

    for idx, bbox in enumerate(gt_bboxes):
        
        #xmin, ymin, xmax, ymax = cxcywh2xyxy_ratio(bbox, img_shape)
        ymin, xmin, ymax, xmax = cxcywh2xyxy_ratio(bbox, img_shape)
        
        bbox_img, bbox = crop_bbox(xmin, ymin, xmax, ymax, img, img_shape, pad)

        bbox_imgs.append(bbox_img)
        bboxes.append(bbox)

    return bbox_imgs, bboxes


def load_copped_image(image_path: str, bboxes: List[List], pad: float) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [

            # VPTW
            # T.Resize(sizes=(1200, 602)),
            # T.Normalize([0.5102, 0.6078, 0.5801], [0.1578, 0.1895, 0.2130]),

            T.ToTensor(),
            
            # Grounding DINO Default, use for bdd100k
            # T.RandomResize([800], max_size=1333),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)

    img_shape = (image.shape[0], image.shape[1])

    image_transformed, _ = transform(image_source, None)

    cropped_bboxes, bbox_coord = crop_bboxes(image, img_shape, bboxes, pad=pad)
    # croooed_bboxes = [np.swapaxes(i, 2, 0) for i in cropped_bboxes]

    # cropped_bboxes_transformed = []
    # for idx, bbox in enumerate(cropped_bboxes):
    #     bbox_transformed, _ = transform(bbox, None)
    #     cropped_bboxes_transformed.append(bbox_transformed)



    return image, image_transformed, cropped_bboxes, bbox_coord



def predict(
        model,
        bboxes,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    
    pp = pprint.PrettyPrinter(indent=4)

    process_caption = caption.replace(' ', '')
    input_classes = [preprocess_caption(i).replace('.', '') for i in caption.split(',')]

    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    if image.ndim == 4:
        with torch.no_grad():
            outputs = model(bboxes, image, captions=[caption])
    else:
        assert image.ndim == 3
        with torch.no_grad():
            outputs = model(bboxes, image[None], captions=[caption])

    #prediction_logits = outputs["pred_logits"].cpu()[0]  # prediction_logits.shape = (nq, 256)

    # Softmax
    #prediction_logits = outputs["pred_logits"].cpu().softmax(dim=1)[0]

    # Sigmoid
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    # all_prediction_logits = outputs["all_pred_logits"].cpu().sigmoid()  # prediction_logits.shape = (nq, 256)
    # all_prediction_boxes = outputs["all_pred_boxes"].cpu()  # prediction_boxes.shape = (nq, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    #print(f'[def predict] caption={caption}')
    #print(f'[def predict] tokenized={tokenized}')


    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits_masked = prediction_logits[mask]  # logits.shape = (n, 256)
    pred_bboxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    # No mask based on threshold
    #logits = prediction_logits  # logits.shape = (n, 256)
    # boxes = prediction_boxes  # boxes.shape = (n, 4)

    

    # test_token_ids = tokenized["input_ids"]
    # test_token_decoded = tokenizer.decode(test_token_ids)
    splited_captions_and_special = [tokenizer.decode(i) for i in tokenized["input_ids"]]

    special_token_idxs = []
    valid_tokens = []
    valid_tokens_idxs = []
    special_tokens = ["[CLS]", "[SEP]", ",", ".", "?"]

    for idx, input_id in enumerate(tokenized["input_ids"]):

        token_str = tokenizer.decode([input_id])
        if token_str in special_tokens:
            special_token_idxs.append(idx)
        else:
            valid_tokens.append(tokenizer.decode([input_id]))
            valid_tokens_idxs.append(idx)

        # print(f'input_id={input_id}, decode={tokenizer.decode([input_id])}')

    prediction_raw_logits = outputs["pred_logits"].cpu()[0]
    token_logits_raw_unmasked = prediction_raw_logits[:len(bboxes), valid_tokens_idxs]
    token_logits_unmasked = prediction_logits[:len(bboxes), valid_tokens_idxs]
    pred_bboxes_unmasked = prediction_boxes[:len(bboxes), :]
    
    tokenidx2class, class2tokenidx = {}, {}
    sub_token_idx_ptr = 0

    for classname_idx, classname in enumerate(input_classes):
        
        class2tokenidx[classname_idx] = []
        per_classname_tokenized = tokenizer(classname)
        splited_captions_and_special = [tokenizer.decode(i) for i in per_classname_tokenized["input_ids"]]
        sub_tokens = [i.replace('#', '') for i in splited_captions_and_special if i not in special_tokens]
        # print(f'[Token mapping] {classname}(#{classname_idx}) tokenlized to {sub_tokens}')

        for sub_token_idx, sub_token in enumerate(sub_tokens):
            sub_token_idx_ptr += sub_token_idx
            tokenidx2class[sub_token_idx_ptr] = (sub_token, classname_idx, classname)
            class2tokenidx[classname_idx].append((sub_token, sub_token_idx_ptr, classname))

            if sub_token_idx+1 == len(sub_tokens):
                sub_token_idx_ptr += 1

    # print(f'\n----- tokenidx2class -----')
    # pp.pprint(tokenidx2class)
    # print('')

    phrases = []
    for logit in logits_masked:
        # print(f'logit={logit}, logit > text_threshold({text_threshold}) = {logit > text_threshold}')
        #phrases_combine, phrases_split = get_phrases_from_posmap(logit, tokenized, tokenizer)
        phrases_combine, phrases_split = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        phrases_combine = phrases_combine.replace('.', '')
        phrases_split = [i.replace('.', '') for i in phrases_split]

        # if len(phrases_split) > 1:
        #     print(f'split={phrases_split}, combine={phrases_combine}')
        #     print(f'logit={logit}')
        #     tmp = logit > text_threshold
        #     non_zero_idx = tmp.nonzero(as_tuple=True)[0].tolist()
        #     print(f'non_zero_idx={non_zero_idx}')
        #     print('-------------')

        phrases.append(phrases_combine)
        # print('-----')

    assert logits_masked.shape[0] == len(phrases)


    # specical_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    # print(f'tokenizer.decode={test_token_decoded}')
    # print(f'type of tokenizer.decode={type(test_token_decoded)}')




    # highest_token_idx = token_logits.max(dim=1)[1]
    # tokenidx2class[highest_token_idx]
    assert token_logits_unmasked.shape[1] == len(tokenidx2class), 'number of tokenlized strings should be the same'
    #####################################################
    #         Merge token_logits to class_logits        #
    #            Filter by Highest value or ?           #
    #####################################################
    class_logits = None
    class_logits_raw = None
    for classname_idx, sub_token_list in class2tokenidx.items():
        # sub_token, sub_token_idx_ptr, classname
        classname = sub_token_list[0][2]
        sub_token_idxs = [i[1] for i in sub_token_list]
        # print(f'{classname} has sub_token_idxs={sub_token_idxs}')

        ##########################################
        #           per class logits             #
        ##########################################
        per_class_logits = token_logits_unmasked[:, sub_token_idxs]
        per_class_logits_max = per_class_logits.max(dim=1)[0].unsqueeze(1)
        #per_class_logits_mean = per_class_logits.mean(dim=1, keepdim=True)
        per_class_logits = per_class_logits_max

        if class_logits is None:
            class_logits = per_class_logits
        else:
            class_logits = torch.cat((class_logits, per_class_logits), 1)

        #############################################
        #           per class logits raw            #
        #############################################
        per_class_logits_raw = token_logits_raw_unmasked[:, sub_token_idxs]
        per_class_logits_raw_max = per_class_logits_raw.max(dim=1)[0].unsqueeze(1)
        per_class_logits_raw = per_class_logits_raw_max
        if class_logits_raw is None:
            class_logits_raw = per_class_logits_raw
        else:
            class_logits_raw = torch.cat((class_logits_raw, per_class_logits_raw), 1)


    # all return objects are in cpu
    assert pred_bboxes.shape[0] == len(phrases)
    return pred_bboxes, pred_bboxes_unmasked, logits_masked.max(dim=1)[0], phrases, valid_tokens, token_logits_unmasked, token_logits_raw_unmasked, class_logits, class_logits_raw, mask.tolist(), tokenidx2class, class2tokenidx


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    box_annotator = sv.BoxAnnotator(text_scale=0.5, thickness=1)
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


# ----------------------------------------------------------------------------------------------------------------------
# NEW API
# ----------------------------------------------------------------------------------------------------------------------


class Model:

    def __init__(
        self,
        model_config_path: str,
        model_checkpoint_path: str,
        device: str = "cuda"
    ):
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device
        ).to(device)
        self.device = device

    def predict_with_caption(
        self,
        image: np.ndarray,
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> Tuple[sv.Detections, List[str]]:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections, labels = model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        """
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        return detections, phrases

    def predict_with_classes(
        self,
        image: np.ndarray,
        classes: List[str],
        box_threshold: float,
        text_threshold: float
    ) -> sv.Detections:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections = model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )


        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        """
        caption = ", ".join(classes)
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        class_id = Model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        return detections

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: torch.Tensor,
            logits: torch.Tensor
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)

    @staticmethod
    def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            try:
                class_ids.append(classes.index(phrase))
            except ValueError:
                class_ids.append(None)
        return np.array(class_ids)
