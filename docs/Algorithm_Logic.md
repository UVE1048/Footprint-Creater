# 建筑物提取算法逻辑文档 (Algorithm Logic)

本文档整理并对齐当前 `extract_buildings.py` 的实际实现逻辑。
整体架构为 **分块 AI 推理 (Block Inference) → 全局合并 (Global Merge) → 全局轮廓校验 (Post-Validation) → 最终规则化 (Post Regularization)**。

---

## 1. 总体流程概览

| 阶段 | 步骤 | 描述 | 对应调试输出 |
| :--- | :--- | :--- | :--- |
| **P1. 分块处理** | 1. 预处理 | 分块、Padding、Nodata/植被/阴影掩码提取 | `block_*_01_Input`, `block_*_02_VegMask`, `block_*_03_ShadowMask` |
|  | 2. AI 推理 | 对干扰区涂黑后推理，并强制从 AI Mask 移除干扰区 | `block_*_03a_InputToAI`, `block_*_04_AI_Raw` |
| **P2. 全局处理** | 3. 合并 | 将各分块 AI Mask 用 OR 合并为全局掩码 | `07_Global_AI_Merged_Raw.tif` |
|  | 4. 全局孔洞填充 | 对全局掩码仅做“轮廓填充”，用于消除内部空洞 | `07b_Global_AI_Filled.png` |
|  | 5. 轮廓校验与筛选 | 面积、宽度、形状(矩形度/实心度)、植被冲突、阴影支撑 | `02_Area_Filtered` |
|  | 6. 孔洞填充 | 对保留图斑进行内部孔洞填充 | `03_Holes_Filled` |
| **P3. 规则化** | 7. 平滑处理 | 对图斑进行膨胀+腐蚀平滑操作 | `04_Smoothed` |
|  | 8. 简化处理 | Douglas-Peucker 简化与矩形化 | `05_Simplified` |
|  | 9. 结果输出 | 生成 Shapefile 和压缩包 | `buildings.shp`, `output.zip` |

---

## 2. 详细逻辑步骤

### Phase 1: 分块 AI 推理 (Block Processing)

对每个影像分块（Block）独立执行以下操作。分块之间存在 `BLOCK_OVERLAP` 的重叠。

#### Step 1: 图像预处理与先验提取
1. **边缘填充 (Padding)**  
   若分块处于图像边缘、尺寸小于 `BLOCK_SIZE`，则在右侧/下侧填充黑色，保证送入 AI 的输入尺寸稳定。

2. **Nodata / 透明区识别 (Nodata Mask)**  
   同时使用两类信息识别无效区域：
   - 影像 RGB 全黑区域视为 Nodata；
   - 读取 `rasterio.read_masks(1)` 的 mask（0 表示无效）。
   **新特性**：在 AI 推理后，会强制移除位于纯黑背景区域的预测结果，消除黑边导致的误检。

#### Step 2: AI 推理 (AI Inference)
1. **AI 推理 (HRNet/SegFormer)**  
   使用语义分割模型推理，得到二值 Mask。

2. **后置清洗 (Post Clean)**  
   强制置零 Nodata 区域，避免边缘黑边干扰。

---

### Phase 2: 全局校验与过滤 (Global Validation)

所有分块处理完成后，合并得到全局掩码（`global_ai_mask`），再提取连通域轮廓并逐一校验。

#### Step 3: 轮廓提取与面积过滤 (Area Filter)
从全局 AI Mask 提取轮廓，并根据 `MIN_AREA_M2` 过滤小面积图斑。
- Debug: `02_Area_Filtered.tif`

#### Step 4: 全局孔洞填充 (Hole Filling)
对保留的轮廓进行内部孔洞填充（仅填充闭合轮廓内部的空洞）。
- Debug: `03_Holes_Filled.tif`

---

### Phase 3: 规则化与输出 (Regularization)

#### Step 5: 平滑处理 (Smoothing)
对填充后的轮廓进行形态学平滑。
1. **预简化**: 使用小公差 (0.3m) 去除光栅化产生的锯齿噪声。
2. **直角缓冲**: 使用 `join_style=mitre` 进行先膨胀后腐蚀，保持边缘直线特性和直角，避免圆角化。
- 参数：`BUFFER_EXPAND_M` (膨胀), `BUFFER_SHRINK_M` (腐蚀)
- Debug: `04_Smoothed.tif`

#### Step 6: 简化处理 (Simplification)
1. **Douglas-Peucker 简化**  
   对平滑后的 Polygon 执行 `poly.simplify(SIMPLIFY_TOL)`。
2. **矩形化 (Rectangularization)**  
   若简化后 Polygon 与其最小外接矩形的 IoU > `RECT_IOU_THRESH`，则替换为规则矩形。
- Debug: `05_Simplified.tif`

#### Step 7: 输出
生成最终的 `buildings.shp` 及配套文件，并打包为 ZIP。

---

## 3. AI 模型说明 (AI Model)

本项目采用深度学习语义分割模型进行建筑物像素级提取。

- **架构**: Encoder-Decoder 结构。
- **Backbone**: **HRNet-W32 (High-Resolution Net)**。HRNet 通过并行连接高分辨率和低分辨率子网，在整个过程中保持高分辨率表示，特别适合建筑物这种边缘精细的目标提取。
- **框架**: 基于 **MMSegmentation** (OpenMMLab) 构建。
- **训练数据**: 针对高分辨率遥感影像（如 Google Earth, Inria Dataset 等）微调的二分类模型（Building vs Background）。
- **输入**:
  - 尺寸: `AI_IMG_SIZE` (默认 1024x1024)
  - 通道: 3通道 (RGB)
- **输出**: 二值掩码 (0: 背景, 1: 建筑)

## 4. 关键参数说明 (Parameters)

| 参数名 | 默认值 | 说明 | 建议调整场景 |
| :--- | :--- | :--- | :--- |
| **BLOCK_SIZE** | 1024 | 分块推理的物理尺寸（像素）。 | 显存不足时调小；大建筑被切断严重时调大。 |
| **BLOCK_OVERLAP** | 512 | 分块间的重叠区域（像素）。 | 通常设为 BLOCK_SIZE 的 50% 以消除边缘效应。 |
| **AI_IMG_SIZE** | 1024 | 送入 AI 模型的缩放尺寸。 | 保持默认即可，除非模型更换。 |
| **MIN_AREA_M2** | 100.0 | 最小建筑面积（平方米）。 | 需要提取小棚房时调小；只需主楼时调大。 |
| **BUFFER_EXPAND_M** | 0.6 | 平滑处理时的膨胀距离（米）。 | 边缘断裂严重时增大；建筑粘连时减小。 |
| **BUFFER_SHRINK_M** | 0.3 | 平滑处理时的腐蚀距离（米）。 | 通常设为膨胀值的一半，用于回缩轮廓。 |
| **SIMPLIFY_TOL** | 0.3 | Douglas-Peucker 简化公差（米）。 | 需要更平直边缘时增大；保留细节时减小。 |
| **RECT_IOU_THRESH** | 0.90 | 矩形化阈值 (IoU)。 | 只有当形状与矩形的重合度超过此值时，才强制变为矩形。 |
