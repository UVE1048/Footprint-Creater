# Footprint-Creater

基于高分辨率正射影像（DOM）自动提取建筑物轮廓，输出可直接入 GIS 的 Shapefile。

> English: Automatic building footprint extraction from high-resolution orthophotos. HRNet semantic segmentation with block-wise inference, followed by geometry-aware post-processing (area filter, hole filling, morphological smoothing, Douglas-Peucker simplification, rectangularisation). Ships with a Flask control console that streams per-stage previews and logs.

链路：**分块 AI 推理 → 全局合并 → 全局轮廓校验 → 规则化输出**。前端是一个 Flask 控制台，每一步的中间结果都能实时看到，不用等到最后才知道哪里出了问题。

![控制台界面](assets/ui.jpg)

## 目录

- [为什么是这个流程](#为什么是这个流程)
- [功能特点](#功能特点)
- [快速开始](#快速开始)
- [使用流程](#使用流程)
- [模型权重](#模型权重)
- [关键参数](#关键参数)
- [算法流程](#算法流程)
- [输出结果](#输出结果)
- [项目结构](#项目结构)
- [性能参考](#性能参考)
- [许可](#许可)

## 为什么是这个流程

把语义分割直接套到大影像上会撞三堵墙，这个项目的设计基本都是围着它们展开的：

1. **大图装不进显存** → 滑窗分块推理，块间留 50% 重叠，最后 OR 合并全局掩码。
2. **拼缝产生假轮廓** → 先在全局掩码上做轮廓校验，再做平滑与简化；而不是逐块美化后再拼。
3. **影像黑边 / Nodata 被误判成建筑** → 推理前用 Nodata + 纯黑 + 植被 + 阴影掩码把干扰区涂黑，推理后强制从结果里剔除。

第三点是最容易踩的坑：一块斜切的正射影像四周是黑的，模型会稳定地把边界黑区识别成大片"建筑"，面积过滤又因为连成一片而拦不住。所以掩码是前置的，不是后置修补。

## 功能特点

- **Web 可视化控制台**：任务管理、参数配置、六阶段实时预览、日志流式输出、历史任务回溯。
- **全自动提取**：上传 `.tif` → 点开始 → 下载 ZIP（含 SHP + 预览图）。
- **稳健性**：自动识别 Nodata / 黑边，支持断点状态恢复，失败任务可单独重启。
- **双模型切换**：HRNet-W32（快）/ HRNet-W48（准），界面下拉即可换。
- **参数全暴露**：分块大小、重叠、面积阈值、平滑膨胀/腐蚀、简化公差、矩形化 IoU 都能在界面上调，也能存档复用。
- **无第三方依赖 UI**：前端 HTML/JS 全部内联在 `web_app.py`，不依赖 CDN 或外部框架。

## 快速开始

```bash
git clone https://github.com/UVE1048/Footprint-Creater.git
cd Footprint-Creater

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 下载预训练权重（约 773 MB，见 model/README.md）
python scripts/download_models.py

python web_app.py
```

浏览器打开 <http://localhost:5000>。

> **装不上 mmcv？** `mmcv` 全量版需要编译 CUDA 算子。纯推理用 `pip install mmcv-lite>=2.1.0` 即可 ——
> `extract_buildings.py` 里已经内置了 `mmcv.ops` 缺失时的 polyfill，不需要 GPU 算子。

## 使用流程

**1. 上传影像** — 支持 `.tif` / `.tiff`，单文件上限 16 GB（`web_app.py` 的 `MAX_CONTENT_LENGTH`）。影像必须是**带地理参考的正射影像**（含像素大小与坐标系），否则面积过滤与 SHP 输出都会失效。

**2. 配置参数**

- **分块大小 / 重叠**：按显存调。1024 / 512 是 8 GB 显存的稳妥值。
- **模型**：W32 默认；追求边界精度换 W48。
- **后处理开关**：全孔洞填充、面积过滤、平滑、简化可单独关闭，便于定位问题。

**3. 开始处理** — 依次执行 6 个阶段，顶部进度条实时更新，中间预览区显示当前阶段的掩码：

![处理结果](assets/result.jpg)

**4. 下载结果** — 完成后自动打包 `<任务名>.zip`（落在 `outputs/`），内含 `buildings.shp` 全家桶 + 预览图。

## 模型权重

权重约 773 MB，通过 GitHub Releases 分发（不进 git）：

```bash
python scripts/download_models.py          # 两个都下
python scripts/download_models.py --model w32   # 只下 W32
python scripts/download_models.py --check       # 校验完整性
```

| 文件 | 骨干 | 体积 | 说明 |
| :--- | :--- | :--- | :--- |
| `iter_40000.pth` | HRNet-W32 + FCNHead | ~231 MB | 默认，速度快 |
| `hrnet_w48_iter_6000.pth` | HRNet-W48 + FCNHead | ~542 MB | 精度更高，约 2× 耗时 |

权重放在 `model/` 下即可。也可以用自己的数据训练 —— 训练流程见 [HRNet-Footprint-Trainer](https://github.com/UVE1048/HRNet-Footprint-Trainer)。

## 关键参数

| 参数 | 默认值 | 作用 | 什么时候改 |
| :--- | :--- | :--- | :--- |
| `BLOCK_SIZE` | 1024 | 分块推理的像素尺寸 | 显存不足调小；大建筑常被切断时调大 |
| `BLOCK_OVERLAP` | 512 | 分块重叠像素 | 保持 `BLOCK_SIZE` 的 50%，低于 25% 拼缝会明显 |
| `AI_IMG_SIZE` | 1024 | 送入网络的缩放尺寸 | 一般不动，换模型时才需要 |
| `MIN_AREA_M2` | 100.0 | 最小建筑面积（m²） | 要小棚房调小；只要主楼调大 |
| `BUFFER_EXPAND_M` | 0.6 | 平滑膨胀距离（m） | 边缘断裂多时增大 |
| `BUFFER_SHRINK_M` | 0.3 | 平滑腐蚀距离（m） | 常规取膨胀值的一半 |
| `SIMPLIFY_TOL` | 0.3 | Douglas-Peucker 公差（m） | 要更平直的边调大，保留细节调小 |
| `RECT_IOU_THRESH` | 0.90 | 矩形化 IoU 阈值 | 阈值越低越多图斑被强制成矩形 |

## 算法流程

完整推导与每步的调试产物见 **[docs/Algorithm_Logic.md](docs/Algorithm_Logic.md)**。概要：

| 阶段 | 步骤 | 做什么 |
| :--- | :--- | :--- |
| P1 分块 | 1. 预处理 | 边缘 Padding、生成 Nodata / 植被 / 阴影掩码 |
| | 2. AI 推理 | 干扰区涂黑后推理，再把干扰区从结果强制置零 |
| P2 全局 | 3. 合并 | 各块掩码 OR 合并为全局掩码 |
| | 4. 全局填洞 | 仅做轮廓填充，消除内部空洞 |
| | 5. 轮廓校验 | 面积 / 宽度 / 矩形度 / 实心度 / 植被冲突 / 阴影支撑 |
| | 6. 孔洞填充 | 对保留图斑填补内部孔洞 |
| P3 规则化 | 7. 平滑 | `join_style=mitre` 先膨胀后腐蚀，保直角不圆角 |
| | 8. 简化 | Douglas-Peucker + 矩形化（IoU > 0.90 则替换为最小外接矩形） |
| | 9. 输出 | 生成 Shapefile 并打包 ZIP |

规则化这一步刻意用 `mitre` 而不是圆弧连接：建筑物的边是直的，圆角化之后在 GIS 里做拓扑检查会对不上。

## 输出结果

```text
outputs/<任务名>.zip
├── buildings.shp / .shx / .dbf / .prj   # 建筑物轮廓矢量
└── 预览图                                # 结果叠加预览
```

矢量会尽量转成整数 EPSG 码再写出，保证 `.prj` 在各家 GIS 里都能正确识别；拿不到 EPSG 时退回原始 CRS WKT，并手工补写 `.prj`。

`debug/` 目录（`DEBUG_MODE = True` 时生成）保留各阶段中间栅格，命名规则见 `docs/Algorithm_Logic.md`，用于逐段排查是哪一步出了问题。

## 项目结构

```text
.
├── web_app.py               # Flask 控制台：路由、任务调度、内联前端
├── extract_buildings.py     # 提取算法核心：推理 + 后处理 + 矢量导出
├── scripts/
│   └── download_models.py   # 从 Releases 拉取权重并校验
├── model/
│   └── README.md           # 权重说明（.pth 不入库）
├── docs/
│   └── Algorithm_Logic.md  # 算法设计文档
├── assets/                  # README 截图
├── requirements.txt
└── .gitignore              # 排除 uploads/ outputs/ debug/ *.tif
```

运行时自动生成的目录：`uploads/`（上传影像）、`outputs/`（任务结果）、`debug/`（中间产物）。

## 性能参考

单张 10 cm GSD、约 2 km × 2 km 的正射影像，在 8 GB 显存 + `BLOCK_SIZE=1024/OVERLAP=512` 下，W32 全流程约 30–40 秒量级（与地物密度强相关）。W48 大约翻倍。

主要开销在 AI 推理。后处理（平滑 + 简化）通常是秒级，除非图斑数量上万。

## 许可

[MIT](LICENSE)
