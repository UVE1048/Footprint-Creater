---
license: mit
library_name: mmsegmentation
pipeline_tag: image-segmentation
tags:
  - building-footprint
  - building-extraction
  - remote-sensing
  - semantic-segmentation
  - hrnet
  - mmsegmentation
  - geospatial
  - orthophoto
  - aerial-imagery
  - 建筑物提取
  - 遥感影像
language:
  - en
  - zh
---

# HRNet-W48 Building Footprint Segmentation

Binary semantic segmentation of building footprints from high-resolution
orthophotos (aerial / satellite imagery). Trained with MMSegmentation using an
HRNet-W48 backbone and an FCN head.

This is the checkpoint used by
[Footprint-Creater](https://github.com/UVE1048/Footprint-Creater), the
post-processing pipeline that turns the raw mask into GIS-ready Shapefiles.

## Architecture

| Item | Value |
| :--- | :--- |
| Backbone | HRNet-W48 (high-resolution parallel branches) |
| Decode head | FCNHead, `in_channels=[48, 96, 192, 384]`, `channels=512` |
| Classes | 2 (`background`, `building`) |
| Input | 1024 × 1024 px, RGB, ImageNet mean/std, BGR→RGB |
| Test mode | `whole` (full-image inference) |
| Framework | MMSegmentation 1.x |
| Weights size | ~542 MB |

HRNet keeps a high-resolution representation throughout the network, which is
why it holds up on building edges better than encoder-decoder designs that
downsample aggressively before recovering detail.

## Intended use

- Pixel-level building footprint extraction from orthophotos at ~5–20 cm GSD.
- Pre-labelling for manual vectorisation.
- Change detection workflows where you need a building mask as an intermediate.

**Not intended for**: instance segmentation (this is semantic, adjacent buildings
merge), rooftop attribute classification, or imagery far outside the training
domain (see Limitations).

## Usage

Install the runtime:

```bash
pip install torch mmengine mmsegmentation mmcv-lite
```

Infer on a patch:

```python
from mmseg.apis import init_model, inference_model
import numpy as np

cfg = dict(
    type="EncoderDecoder",
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32,
    ),
    backbone=dict(
        type="HRNet",
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        init_cfg=None,   # weights come from the checkpoint
    ),
    decode_head=dict(
        type="FCNHead",
        in_channels=[48, 96, 192, 384],
        in_index=(0, 1, 2, 3),
        channels=512,
        input_transform="resize_concat",
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type="BN", requires_grad=False),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    test_cfg=dict(mode="whole"),
)

model = init_model(cfg, "hrnet_w48_iter_6000.pth", device="cuda:0")
result = inference_model(model, "patch.png")
mask = result.pred_sem_seg.data.cpu().numpy()[0]   # 0 = background, 1 = building
```

For full-size orthophotos do **not** feed the whole image. Use a sliding window
with ~50% overlap and stitch — a reference implementation is in
[`predict.py`](https://github.com/UVE1048/HRNet-Footprint-Trainer/blob/main/predict.py).

## Post-processing matters more than the mask

A raw semantic mask is not usable as a map product. The reference pipeline
applies, in order:

1. Drop connected components below a minimum area (default 100 m²).
2. Fill interior holes.
3. Smooth with a mitre-join buffer (expand then shrink) so right angles survive.
4. Douglas-Peucker simplification.
5. Replace with the minimum bounding rectangle when IoU with it exceeds a threshold.

Skipping step 3 or 4 is the usual reason extracted outlines look jagged in GIS.

## Limitations

- **Semantic, not instance.** Touching buildings merge into one polygon.
- **Black borders.** Pure black / white regions at orthophoto edges are
  systematically predicted as buildings. Mask them out **before** inference; area
  filtering afterwards does not fix it, because the false positives form one
  large blob.
- **Domain sensitivity.** Seasonal change, sensor differences, and unusual roof
  materials (e.g. metal sheet roofing, green roofs) degrade accuracy. Retraining
  on local data is the reliable fix — see
  [HRNet-Footprint-Trainer](https://github.com/UVE1048/HRNet-Footprint-Trainer).
- **No published benchmark numbers.** This checkpoint was trained on a private
  orthophoto/building-vector dataset, so no comparable IoU on public benchmarks
  is reported here. Validate on your own imagery before production use.

## Training summary

| Item | Value |
| :--- | :--- |
| Objective | CrossEntropy + Dice, equal weight |
| Optimiser | SGD with PolyLR decay |
| Iterations | 6000 |
| Patch size | 1024 × 1024, stride 512 |
| Dataset | private high-resolution orthophotos + building vector labels |

NoData handling: pure black `(0,0,0)` and pure white `(255,255,255)` pixels are
set to `ignore_index` in the labels, so image borders never contribute to the
loss.

## Licence

MIT. Provided as-is, without warranty. You are responsible for ensuring your
imagery and any derived data comply with local surveying and data regulations.
