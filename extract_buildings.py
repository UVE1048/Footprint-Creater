import os
import sys
import glob
import time
import shutil
import os
import sys
import time
import gc
import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
import warnings
from rasterio.windows import Window
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.validation import make_valid
import geopandas as gpd
from types import ModuleType
from unittest.mock import MagicMock
import torch

# Suppress mmcv-lite warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mmcv.cnn.bricks.transformer")
# Suppress transformers/torch pytree warning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic")

# ================= Patch for mmcv-lite (Mocking mmcv.ops) =================
try:
    import mmcv
    try:
        import mmcv.ops
    except ImportError:
        print("[Warning] mmcv.ops not found. Applying patch for mmcv-lite...")
        
        class FlexibleMock(MagicMock):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            def __reduce__(self):
                return (MagicMock, ())

        class MockOpsModule(ModuleType):
            def __getattr__(self, name):
                if name in ('__file__', '__path__', '__loader__', '__spec__'):
                    return super().__getattr__(name)
                return FlexibleMock()
        
        fake_ops = MockOpsModule('mmcv.ops')
        fake_ops.__file__ = "mock_mmcv_ops.py"
        fake_ops.__path__ = []
        sys.modules['mmcv.ops'] = fake_ops
        if 'mmcv' in sys.modules:
            sys.modules['mmcv'].ops = fake_ops
        print("[Success] mmcv.ops patched successfully.")
except Exception as e:
    print(f"[Error] Failed to patch mmcv.ops: {e}")

# Monkeypatch torch.load for PyTorch 2.6+ compatibility
_original_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_torch_load
# ==========================================================================

from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules
from mmengine.config import Config as MMConfig
from mmseg.datasets import BaseSegDataset
from mmengine.registry import DATASETS
import os.path as osp

# ==========================
# Embedded Config & Dataset
# ==========================
@DATASETS.register_module()
class FootprintDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'building'),
        palette=[[0, 0, 0], [255, 0, 0]]
    )
    def load_data_list(self):
        # Dummy implementation for inference
        return []

def get_mmseg_cfg():
    # Determine config based on model path
    is_w48 = 'w48' in Config.AI_MODEL_PATH.lower()
    
    if is_w48:
        # HRNet-W48 Config (Cascade + OCR)
        model_type = 'CascadeEncoderDecoder'
        num_stages = 2
        
        # W48 Channels: [48, 96, 192, 384]
        backbone_extra = dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384)
            )
        )
        
        decode_head = [
            dict(
                type='FCNHead',
                in_channels=[48, 96, 192, 384],
                in_index=(0, 1, 2, 3),
                input_transform='resize_concat',
                channels=256,
                kernel_size=1,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=2,
                norm_cfg=dict(type='BN', requires_grad=False),
                align_corners=False,
                loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
            ),
            dict(
                type='OCRHead',
                in_channels=[48, 96, 192, 384],
                in_index=(0, 1, 2, 3),
                input_transform='resize_concat',
                channels=512,
                ocr_channels=256,
                dropout_ratio=0.1,
                num_classes=2,
                norm_cfg=dict(type='BN', requires_grad=False),
                align_corners=False,
                sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
                loss_decode=[
                    dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                ]
            )
        ]
        
    else:
        # HRNet-W32 Config (Default)
        model_type = 'EncoderDecoder'
        num_stages = None
        
        # W32 Channels: [32, 64, 128, 256]
        backbone_extra = dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)
            )
        )
        
        decode_head = dict(
            type='FCNHead',
            in_channels=[32, 64, 128, 256],
            in_index=(0, 1, 2, 3),
            channels=512,
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=dict(type='BN', requires_grad=False),
            align_corners=False,
            loss_decode=[
                dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                dict(type='DiceLoss', loss_weight=1.0)
            ]
        )

    # Replicates the structure from V2/train.py but simplified for inference
    model_cfg = dict(
        type=model_type,
        data_preprocessor=dict(
            type='SegDataPreProcessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_val=0,
            seg_pad_val=255,
            size_divisor=32
        ),
        backbone=dict(
            type='HRNet',
            # init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/pretrain/third_party/hrnetv2_w32-dc9eeb4f.pth'),
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            extra=backbone_extra
        ),
        decode_head=decode_head,
        train_cfg=dict(),
        test_cfg=dict(mode='whole')
    )
    
    if num_stages:
        model_cfg['num_stages'] = num_stages

    cfg = MMConfig(dict(
        default_scope='mmseg',
        launcher='none',
        work_dir='./work_dirs/hrnet_w32_footprint',
        randomness=dict(seed=42),
        model=model_cfg,
        train_pipeline=[], # Not needed
        test_pipeline=[
            dict(type='LoadImageFromFile'),
            # dict(type='LoadAnnotations'), # Removed
            dict(type='PackSegInputs')
        ]
    ))
    return cfg

# ==========================
# Global Config Management
# ==========================
class Config:
    # 0. System & Model
    BLOCK_SIZE = 1024          # Crop Size
    BLOCK_OVERLAP = 512         # Stride = BLOCK_SIZE - BLOCK_OVERLAP (logic adaptation needed)
                                # Note: In predict.py, STRIDE is explicit. Here we adapt.
    AI_MODEL_PATH = "model/iter_40000.pth"
    AI_DEVICE = "auto"
    AI_IMG_SIZE = 1024         # Input size
    
    # 0.1 Input Robustness
    BLACK_MASK_THRESHOLD = 5   # Pixel value threshold to consider as valid image content (0-255)
    EDGE_MASK_EROSION = 5      # Pixels to erode from valid mask edges to remove model artifacts at Nodata boundaries

    # 3. Global Fill
    ENABLE_GLOBAL_FILL = True

    # 5. Area Filter
    ENABLE_FILTER_AREA = True
    MIN_AREA_M2 = 100.0

    # 7. Post-Processing
    ENABLE_SMOOTHING = True
    BUFFER_EXPAND_M = 0.6
    BUFFER_SHRINK_M = 0.3
    
    ENABLE_SIMPLIFY = True
    SIMPLIFY_TOL = 0.3
    RECT_IOU_THRESH = 0.90

    DEBUG_MODE = False
    DEBUG_DIR = "debug"

    @classmethod
    def update(cls, config_dict):
        for k, v in config_dict.items():
            if hasattr(cls, k):
                # Type conversion
                target_type = type(getattr(cls, k))
                if target_type == bool:
                    setattr(cls, k, str(v).lower() in ('true', '1', 'yes'))
                elif target_type == int:
                    setattr(cls, k, int(v))
                elif target_type == float:
                    setattr(cls, k, float(v))
                else:
                    setattr(cls, k, v)

class BuildingExtractor:
    def __init__(self, dom_path, output_dir=None, progress_callback=None):
        self.dom_path = dom_path
        self.progress_callback = progress_callback
        
        # Output directory setup
        if output_dir:
            self.output_dir = output_dir
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join("outputs", timestamp)
            
        debug_sub = getattr(Config, "DEBUG_DIR", "debug")
        self.debug_dir = os.path.join(self.output_dir, debug_sub)
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Read DOM metadata
        with rasterio.open(dom_path) as src:
            self.profile = src.profile.copy()
            self.width = src.width
            self.height = src.height
            self.transform = src.transform
            self.crs = src.crs
            self.res_x = self.transform[0]
            self.res_y = -self.transform[4]  # Usually negative
        
        # Model placeholder
        self._model = None

    def update_progress(self, current, total, message=""):
        if self.progress_callback:
            self.progress_callback(current, total, message)
        else:
            pct = (current / total) * 100 if total > 0 else 0
            print(f"[Progress {pct:.1f}%] {message}")

    def _get_model(self):
        if self._model is None:
            print(f"Loading MMSet model from: {Config.AI_MODEL_PATH}")
            register_all_modules()
            cfg = get_mmseg_cfg()
            
            # Ensure test_pipeline matches inference needs
            if hasattr(cfg, 'test_pipeline'):
                 cfg.test_pipeline = [p for p in cfg.test_pipeline if p['type'] != 'LoadAnnotations']
            else:
                 cfg.test_pipeline = [
                    dict(type='LoadImageFromFile'),
                    dict(type='PackSegInputs')
                ]

            device = Config.AI_DEVICE
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self._model = init_model(cfg, Config.AI_MODEL_PATH, device=device)
        return self._model

    def _to_uint8_rgb(self, image_data):
        # Handle different channel counts to RGB
        if image_data.ndim == 2:
            return cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
        c, h, w = image_data.shape
        if c == 1:
            img = image_data[0]
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif c == 3:
            return np.transpose(image_data, (1, 2, 0)) # C,H,W -> H,W,C
        elif c > 3:
            return np.transpose(image_data[:3], (1, 2, 0))
        return None

    def _save_tiling_debug(self, xs, ys, crop_size):
        print("Generating tiling preview...")
        max_dim = 2048
        scale = min(1.0, max_dim / max(self.height, self.width))
        out_w = int(self.width * scale)
        out_h = int(self.height * scale)
        
        with rasterio.open(self.dom_path) as src:
            # Read first 3 bands (or 1 if grayscale)
            count = min(3, src.count)
            indexes = list(range(1, count + 1))
            data = src.read(
                indexes=indexes,
                out_shape=(count, out_h, out_w),
                resampling=rasterio.enums.Resampling.bilinear
            )
            
        # Normalize to uint8 for display
        norm_data = []
        for i in range(data.shape[0]):
            band = data[i]
            # Simple min-max with percentile to avoid outliers
            lower = np.percentile(band, 2)
            upper = np.percentile(band, 98)
            if upper > lower:
                norm = (band - lower) / (upper - lower) * 255.0
                norm = np.clip(norm, 0, 255).astype(np.uint8)
            else:
                norm = np.zeros_like(band, dtype=np.uint8)
            norm_data.append(norm)
            
        if not norm_data:
            return

        # Stack and transpose to HWC
        if len(norm_data) == 1:
            img = cv2.cvtColor(norm_data[0], cv2.COLOR_GRAY2BGR)
        else:
            img = np.dstack(norm_data)
            # Rasterio is RGB, OpenCV needs BGR
            if img.shape[2] == 3:
                 img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Draw grid
        color = (0, 255, 0) # Green
        
        for y in ys:
            for x in xs:
                sx = int(x * scale)
                sy = int(y * scale)
                sw = int(crop_size * scale)
                sh = int(crop_size * scale)
                cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), color, 1)
                
        out_path = os.path.join(self.debug_dir, "00_Tiling_Preview.png")
        cv2.imwrite(out_path, img)

    def run_tiled_inference(self, progress_callback=None):
        """
        Executes sliding window inference similar to predict.py
        """
        print("Starting AI inference...")
        model = self._get_model()
        
        # Config mapping
        crop_size = Config.BLOCK_SIZE # Physical read size (e.g. 2048 for 100m context)
        img_size = Config.AI_IMG_SIZE # Model input size (e.g. 1024)
        
        # We use Config.BLOCK_OVERLAP as 'Stride' in the UI if adapted, or Stride = Size - Overlap
        stride = Config.BLOCK_OVERLAP 
        
        # If stride is larger than crop size, it's invalid. Default to 512 if weird.
        if stride > crop_size: 
            stride = crop_size // 2
        
        print(f"Inference Config: Read Size={crop_size}, Model Input={img_size}, Stride={stride}")
        
        ys = list(range(0, self.height - crop_size + 1, stride))
        xs = list(range(0, self.width - crop_size + 1, stride))
        
        if ys[-1] != self.height - crop_size: ys.append(self.height - crop_size)
        if xs[-1] != self.width - crop_size: xs.append(self.width - crop_size)
        
        try:
            self._save_tiling_debug(xs, ys, crop_size)
        except Exception as e:
            print(f"Warning: Failed to save tiling debug: {e}")
        
        total_patches = len(ys) * len(xs)
        result_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        processed_count = 0
        
        with rasterio.open(self.dom_path) as src:
            for y in ys:
                for x in xs:
                    window = Window(x, y, crop_size, crop_size)
                    img_data = src.read(window=window)
                    
                    # Preprocess for mmseg (BGR)
                    img_data = np.transpose(img_data, (1, 2, 0)) # H,W,C
                    
                    # ---------------------------------------------------------
                    # [Step 1] Robust Valid Area Mask Generation
                    # ---------------------------------------------------------
                    # Determine valid area BEFORE any resizing or channel dropping
                    # This handles:
                    # 1. Alpha channel (if exists) -> Best source of truth
                    # 2. Black/Nodata borders -> Use threshold
                    valid_mask = None
                    
                    if img_data.shape[2] == 4:
                        # Use Alpha channel if available
                        valid_mask = img_data[:, :, 3] > 0
                    elif img_data.shape[2] >= 3:
                        # Use RGB threshold
                        # Note: We use a threshold (default 5) instead of strict 0 to handle compression noise
                        # max(R,G,B) > threshold
                        valid_mask = np.max(img_data[:, :, :3], axis=2) > Config.BLACK_MASK_THRESHOLD
                    elif img_data.shape[2] == 1:
                        # Grayscale
                        valid_mask = img_data[:, :, 0] > Config.BLACK_MASK_THRESHOLD
                        
                    # Optimization: Skip empty blocks
                    if valid_mask is not None and not np.any(valid_mask):
                        processed_count += 1
                        if progress_callback:
                            progress_callback(processed_count / total_patches)
                        continue

                    # ---------------------------------------------------------
                    # [Step 2] Image Formatting for AI
                    # ---------------------------------------------------------
                    if img_data.shape[2] > 3:
                        img_data = img_data[:, :, :3]
                    elif img_data.shape[2] == 1:
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                    
                    # Align with predict.py: Always convert RGB (Rasterio) to BGR (OpenCV/MMCV)
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                    
                    # Resize if Read Size != Model Input Size
                    if crop_size != img_size:
                        img_input = cv2.resize(img_data, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                    else:
                        img_input = img_data

                    # ---------------------------------------------------------
                    # [Step 3] Inference
                    # ---------------------------------------------------------
                    result = inference_model(model, img_input)
                    pred_mask = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
                    
                    # Resize back to Read Size if needed
                    if crop_size != img_size:
                        pred_mask = cv2.resize(pred_mask, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

                    # ---------------------------------------------------------
                    # [Step 4] Apply Valid Area Mask (Fix for Edge Black/Noise)
                    # ---------------------------------------------------------
                    if valid_mask is not None:
                        # Ensure shapes match
                        if valid_mask.shape == pred_mask.shape:
                             # [Fix for White Lines at Edges]
                             # Erode the valid mask slightly to remove edge artifacts/white lines
                             # caused by model response to sharp black borders.
                             # We use borderValue=1 to ensure we ONLY erode internal Nodata boundaries,
                             # not the outer edges of the image block itself (which would cause grid lines).
                             kernel_size = Config.EDGE_MASK_EROSION
                             
                             # Adaptive kernel for W48 model which has larger receptive field and may produce wider edge artifacts
                             if 'w48' in Config.AI_MODEL_PATH.lower():
                                 kernel_size = max(kernel_size, 9)

                             if kernel_size > 0:
                                 kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                                 valid_mask_uint8 = valid_mask.astype(np.uint8)
                                 # borderValue=1 means "assume outside is valid", so we don't shrink from the block borders
                                 valid_mask_eroded = cv2.erode(valid_mask_uint8, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=1)
                                 valid_mask = valid_mask_eroded > 0

                             # Force invalid areas to 0 (Background)
                             pred_mask[~valid_mask] = 0
                        else:
                            print(f"Warning: Valid mask shape {valid_mask.shape} != Pred mask shape {pred_mask.shape}")

                    # Center Crop Write Strategy
                    y_start_write = y
                    y_end_write = y + crop_size
                    x_start_write = x
                    x_end_write = x + crop_size
                    
                    if y > 0 and y < self.height - crop_size:
                        y_start_write += (crop_size - stride) // 2
                    if y < self.height - crop_size and y > 0:
                        y_end_write -= (crop_size - stride) // 2
                        
                    if x > 0 and x < self.width - crop_size:
                        x_start_write += (crop_size - stride) // 2
                    if x < self.width - crop_size and x > 0:
                        x_end_write -= (crop_size - stride) // 2
                        
                    crop_y1 = y_start_write - y
                    crop_y2 = crop_y1 + (y_end_write - y_start_write)
                    crop_x1 = x_start_write - x
                    crop_x2 = crop_x1 + (x_end_write - x_start_write)
                    
                    result_mask[y_start_write:y_end_write, x_start_write:x_end_write] = \
                        pred_mask[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    processed_count += 1
                    if progress_callback:
                        progress_callback(processed_count / total_patches)
                    if processed_count % 10 == 0:
                        self.update_progress(processed_count, total_patches, "AI Inference...")
        
        return result_mask

    def _fill_polygon_holes(self, polygons, progress_callback=None):
        if not Config.ENABLE_GLOBAL_FILL:
            return polygons
        
        total = len(polygons)
        print(f"执行多边形孔洞填充 (Global Fill)... 待处理数量: {total}")
        filled = []
        for i, poly in enumerate(polygons):
            if poly.is_empty:
                continue
            
            if poly.geom_type == 'Polygon':
                # Only keep the exterior shell, effectively removing all holes
                filled.append(Polygon(poly.exterior))
            elif poly.geom_type == 'MultiPolygon':
                for p in poly.geoms:
                    filled.append(Polygon(p.exterior))
            elif poly.geom_type == 'GeometryCollection':
                 for p in poly.geoms:
                    if p.geom_type == 'Polygon':
                        filled.append(Polygon(p.exterior))
            
            if progress_callback and (i % 500 == 0 or i == total - 1):
                progress_callback((i + 1) / total)

        print(f"孔洞填充完成 (Hole Filling Done).")
        return filled

    def _collect_candidate_contours(self, mask):
        print("Collecting contours (Geo Coordinates)...")
        # Use rasterio.features.shapes to extract polygons directly in Geo Coordinates
        # This ensures the coordinate system is exactly consistent with the input TIF
        import rasterio.features
        
        candidates = []
        
        # mask is uint8, values 0 and 1
        # transform=self.transform ensures output coordinates are Projected/Geo
        # Note: rasterio.features.shapes returns a generator, so we can't easily track progress percentage
        # without consuming it first.
        count = 0
        for geom, val in rasterio.features.shapes(mask, transform=self.transform):
            if val == 1:
                poly = shape(geom)
                if not poly.is_valid:
                    poly = make_valid(poly)
                
                if poly.is_empty:
                    continue
                    
                # Decompose MultiPolygons if any (though shapes usually yields Polygons)
                if poly.geom_type == 'Polygon':
                    candidates.append(poly)
                elif poly.geom_type in ['MultiPolygon', 'GeometryCollection']:
                    for p in poly.geoms:
                        if p.geom_type == 'Polygon':
                            candidates.append(p)
                
                count += 1
                if count % 2000 == 0:
                    print(f"  ...Collected {count} candidates so far")
                            
        print(f"Collection finished. Total candidates: {len(candidates)}")
        return candidates

    def _verify_contours(self, candidates, progress_callback=None):
        total = len(candidates)
        print(f"Filtering contours (Area Only)... Total: {total}")
        final_polys = []
        debug_polys = [] 
        
        min_area = Config.MIN_AREA_M2
        
        for i, poly in enumerate(candidates):
            # Area Check
            # Note: Polygons are now in Geo Coordinates (Meters if Projected)
            # If CRS is WGS84, this area is in degrees^2, which requires special handling.
            # Assuming standard Projected CRS for "5cm Ortho".
            area_m2 = poly.area 
            
            if Config.ENABLE_FILTER_AREA and area_m2 < min_area:
                continue
                
            final_polys.append(poly)
            
            if progress_callback and (i % 500 == 0 or i == total - 1):
                progress_callback((i + 1) / total)
            
        print(f"Filter complete. Kept {len(final_polys)}/{total} polygons.")
        return final_polys, debug_polys

    def _simplify_polygons(self, polygons, progress_callback=None):
        if not Config.ENABLE_SIMPLIFY:
            return polygons
            
        total = len(polygons)
        print(f"Simplifying polygons (DP + Rect)... Total: {total}")
        simplified = []
        tol = getattr(Config, "SIMPLIFY_TOL", 0.5)
        rect_iou = getattr(Config, "RECT_IOU_THRESH", 0.85)
        
        for i, poly in enumerate(polygons):
            # Douglas-Peucker
            s_poly = poly.simplify(tol, preserve_topology=True)
            
            # Rectangularize if close to rectangle
            min_rect = s_poly.minimum_rotated_rectangle
            if min_rect.area > 0:
                iou = s_poly.area / min_rect.area
                if iou > rect_iou:
                    s_poly = min_rect
            
            simplified.append(s_poly)
            
            if progress_callback and (i % 500 == 0 or i == total - 1):
                progress_callback((i + 1) / total)
            
        print("Simplification done.")
        return simplified

    def _smooth_polygons(self, polygons, progress_callback=None):
        if not getattr(Config, "ENABLE_SMOOTHING", True):
            return polygons

        total = len(polygons)
        print(f"Smoothing polygons (Buffer)... Total: {total}")
        smoothed = []
        expand = getattr(Config, "BUFFER_EXPAND_M", 1.0)
        shrink = getattr(Config, "BUFFER_SHRINK_M", 0.5)
        
        for i, poly in enumerate(polygons):
            # Smoothing (Buffer)
            # Use join_style=2 (mitre) to preserve straight lines/sharp corners instead of rounding them
            # Also pre-simplify slightly to remove pixel noise before buffering
            # Increased tolerance to 0.3m to better straighten wobbly edges
            p_clean = poly.simplify(0.3, preserve_topology=True)
            s_poly = p_clean.buffer(expand, join_style=2).buffer(-shrink, join_style=2)
            smoothed.append(s_poly)
            
            if progress_callback and (i % 500 == 0 or i == total - 1):
                progress_callback((i + 1) / total)

        print("Smoothing done.")
        return smoothed

    def _final_simplify_if_enabled(self, polygons):
        # Legacy wrapper if needed, or just used by run()
        polys = self._simplify_polygons(polygons)
        polys = self._smooth_polygons(polys)
        return polys

    def _save_merged_ai_mask_debug(self, mask):
        # Save raw 0/1 mask for processing
        path = os.path.join(self.debug_dir, "01_Global_AI_Raw.tif")
        
        # Create a single-band profile for the mask
        debug_profile = self.profile.copy()
        debug_profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw',
            driver='GTiff',
            photometric='MINISBLACK'
        )
        
        with rasterio.open(path, 'w', **debug_profile) as dst:
            dst.write(mask, 1)
            
        # Save visible visualization (Red 255 for visibility) for user check
        vis_path = path.replace(".tif", "_vis.tif")
        vis_profile = self.profile.copy()
        vis_profile.update(dtype=rasterio.uint8, count=3, photometric='RGB', compress='lzw')
        
        vis_mask = np.zeros((3, self.height, self.width), dtype=np.uint8)
        mask_indices = (mask > 0)
        vis_mask[0][mask_indices] = 255 # R
        vis_mask[1][mask_indices] = 255 # G
        vis_mask[2][mask_indices] = 255 # B
        
        with rasterio.open(vis_path, 'w', **vis_profile) as dst:
            dst.write(vis_mask)
            
        self._save_png_limited(path.replace(".tif", ".png"), mask)

    def _save_png_limited(self, path, mask, max_dim=2048):
        # Downsample for PNG preview
        h, w = mask.shape
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            resized = mask
        
        vis = np.zeros((resized.shape[0], resized.shape[1], 3), dtype=np.uint8)
        # White color for preview (BGR for OpenCV)
        vis[resized > 0] = [255, 255, 255] 
        cv2.imwrite(path, vis)

    def save_debug_polygons(self, polygons, filename_prefix):
        # Create a raster for visualization
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Rasterize polygons
        # NOTE: Polygons are now in GEO coordinates (Meters).
        # We MUST pass self.transform here to map them back to pixels for the raster.
        import rasterio.features
        if polygons:
            shapes = [(p, 1) for p in polygons]
            rasterio.features.rasterize(
                shapes=shapes,
                out=mask,
                transform=self.transform # Required for Geo -> Pixel
            )
            
        path = os.path.join(self.debug_dir, f"{filename_prefix}.tif")
        
        # Create a single-band profile for the mask
        debug_profile = self.profile.copy()
        debug_profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw',
            driver='GTiff',
            photometric='MINISBLACK'
        )

        with rasterio.open(path, 'w', **debug_profile) as dst:
            dst.write(mask, 1)
            
        # Save visible visualization (White 255) for user check (Web Preview)
        vis_path = path.replace(".tif", "_vis.tif")
        vis_profile = self.profile.copy()
        vis_profile.update(dtype=rasterio.uint8, count=3, photometric='RGB', compress='lzw')
        
        vis_mask = np.zeros((3, self.height, self.width), dtype=np.uint8)
        mask_indices = (mask > 0)
        vis_mask[0][mask_indices] = 255 # R
        vis_mask[1][mask_indices] = 255 # G
        vis_mask[2][mask_indices] = 255 # B
        
        with rasterio.open(vis_path, 'w', **vis_profile) as dst:
            dst.write(vis_mask)

        self._save_png_limited(path.replace(".tif", ".png"), mask)

    def _write_shapefile(self, polygons):
        if not polygons:
            return None
            
        out_path = os.path.join(self.output_dir, "buildings.shp")
        
        # Polygons are already in Geo Coordinates (from rasterio.features.shapes)
        # Ensure CRS is properly passed
        original_crs = self.crs
        target_crs = original_crs
        
        print(f"Preparing to write Shapefile... {len(polygons)} polygons.")
        
        if original_crs:
            print(f"Writing Shapefile with CRS: {original_crs}")
            # Optimize CRS for compatibility (Prefer EPSG code)
            try:
                epsg_code = None
                # 1. Try native to_epsg()
                if hasattr(original_crs, 'to_epsg'):
                    epsg_code = original_crs.to_epsg()
                
                # 2. If failed, check for common Web Mercator signatures that might miss authority
                if not epsg_code and hasattr(original_crs, 'to_wkt'):
                    wkt = original_crs.to_wkt()
                    if "Pseudo-Mercator" in wkt or "Pseudo_Mercator" in wkt:
                         print("  Detected Pseudo-Mercator without explicit EPSG. Forcing EPSG:3857.")
                         epsg_code = 3857

                if epsg_code:
                    print(f"  Converting CRS to EPSG:{epsg_code} for better compatibility.")
                    target_crs = f"EPSG:{epsg_code}"
                else:
                    print("  Could not determine EPSG code, using original CRS WKT.")
            except Exception as e:
                print(f"  CRS conversion warning: {e}")

        try:
            gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=target_crs)
            
            # Force writing .prj file by ensuring driver handles it
            print(f"Saving to {out_path} ...")
            gdf.to_file(out_path, driver='ESRI Shapefile', encoding='utf-8')
            print("Shapefile saved successfully.")
        except Exception as e:
            print(f"Error writing shapefile: {e}")
            raise e
        
        # Double check if .prj was created
        prj_path = out_path.replace('.shp', '.prj')
        if not os.path.exists(prj_path) and original_crs:
            print("Manually creating .prj file...")
            try:
                with open(prj_path, 'w') as f:
                    # Use target_crs if it is a string (EPSG:xxxx), otherwise use original_crs WKT
                    if isinstance(target_crs, str) and target_crs.lower().startswith("epsg:"):
                         # Write standard WKT for EPSG
                         from rasterio.crs import CRS
                         # Re-create CRS object to get standard WKT
                         std_crs = CRS.from_string(target_crs)
                         f.write(std_crs.to_wkt())
                    else:
                         f.write(original_crs.to_wkt())
            except Exception as e:
                print(f"Failed to write .prj: {e}")
                
        return out_path

    def cleanup(self):
        if self._model is not None:
            del self._model
            self._model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()

    def save_source_preview(self):
        # Create a preview of the source DOM
        try:
            # Read a downsampled version for preview
            with rasterio.open(self.dom_path) as src:
                # Read first 3 bands if available, or just 1
                count = min(3, src.count)
                # Calculate scale for 2048px max dim
                max_dim = 2048
                h, w = src.height, src.width
                scale = min(1.0, max_dim / max(h, w))
                
                out_shape = (count, int(h * scale), int(w * scale))
                # Explicitly specify indexes to match out_shape count
                indexes = list(range(1, count + 1))
                
                # Pre-allocate buffer to avoid shape mismatch errors
                data = np.empty(out_shape, dtype=src.profile['dtype'])
                src.read(indexes=indexes, out=data, resampling=Resampling.bilinear)
                
                # Transpose to HWC for OpenCV
                if count >= 3:
                    img = data[:3].transpose(1, 2, 0)
                    # Convert RGB to BGR for OpenCV
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img = data[0]
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                path = os.path.join(self.debug_dir, "00_Input_Preview.png")
                cv2.imwrite(path, img)
        except Exception as e:
            print(f"Failed to save source preview: {e}")

    def run(self):
        try:
            global_ai_mask = self.run_tiled_inference()
            self._save_merged_ai_mask_debug(global_ai_mask)
            
            # Step 1: Collect Contours directly from AI Mask
            candidates = self._collect_candidate_contours(global_ai_mask)
            
            # Step 2: Area Filter
            filtered_polys, _ = self._verify_contours(candidates)
            self.save_debug_polygons(filtered_polys, "02_Area_Filtered")
            
            # Step 3: Global Hole Filling (After Area Filter)
            filled_polys = self._fill_polygon_holes(filtered_polys)
            self.save_debug_polygons(filled_polys, "03_Holes_Filled")
            
            # Step 4: Smoothing (Step 5 in UI)
            # Note: UI Step 4 is Hole Filling (Done above)
            # UI Step 5: Smoothing
            smoothed_polys = self._smooth_polygons(filled_polys)
            self.save_debug_polygons(smoothed_polys, "04_Smoothed")
            
            # Step 5: Simplification (Step 6 in UI)
            simplified_polys = self._simplify_polygons(smoothed_polys)
            self.save_debug_polygons(simplified_polys, "05_Simplified")
            
            self._write_shapefile(simplified_polys)
            
            print("Done.")
            
        finally:
            self.cleanup()

    def __del__(self):
        self.cleanup()

if __name__ == "__main__":
    # Test block
    dom_path = r"test/YP-DQ.tif" # Example
    if os.path.exists(dom_path):
        extractor = BuildingExtractor(dom_path)
        extractor.run()
