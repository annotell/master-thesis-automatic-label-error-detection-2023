

# Steps

```
pip install -r requirements.txt
```

## Process the Raw Data (OD2_CL.py)

Raw data is the csv files directly from the model inference  (Directory: data_pred)

This step adds CL flag and ranking scores and generates the csv files suitable for visualization.

### Supervised Model
use `def process_bdd100k_fasterrcnn_combine_splits` for BDD100K
```
python OD2_CL.py
```

use `def process_shift_fasterrcnn` for SHIFT
```
python OD2_CL.py {noise_type}
```
noise_type: ass20, ass10, ass5, ass1, uni20, uni10, uni5, uni1

### SSL Model

use `def process_shift_gdino` for SHIFT
```
python OD2_CL.py {dataset} {noise_type} {subset}
python OD2_CL.py shift ass1 all
```

use `def process_coco_gdino` for COCO
```
python OD2_CL.py coco ass20 shiftall
```
shiftall is the subset of COCO (6 classes out of 91)

---

## Plot (OD2_CL.py)

### Supervised Model
SHIFT
```
OD3_SHIFTvis.ipynb
```

BDD100K
```
OD3_BDDvis.ipynb
```

### SSL Model
SHIFT
```
OD3_round_shift_swint.ipynb
```


COCO
```
OD3_round_coco_swint.ipynb
OD3_round_coco_swinb.ipynb
```


## Manunally check object detection label errors
```
OD_checker.py
```