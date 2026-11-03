# AGENTS.md

面向 AI 编码代理的仓库说明。人类读者请先看 [README](README.md)。

## 这是什么

从高分辨率正射影像自动提取建筑物轮廓，输出 Shapefile。Flask 控制台 + HRNet 语义分割 + 几何后处理。

## 关键文件

- `extract_buildings.py` — 算法核心：模型加载、分块推理、全局轮廓校验、平滑、简化、矢量导出。改算法看这里。
- `web_app.py` — Flask 应用，3128 行，**前端 HTML/JS 全部内联在此文件的字符串里**，没有 templates/ 或 static/ 目录。改界面就在这个文件里搜 CSS 类名。
- `docs/Algorithm_Logic.md` — 算法设计文档，改算法请同步更新。
- `scripts/download_models.py` — 从 GitHub Releases 拉权重并做 SHA256 校验。

## 命令

```bash
pip install -r requirements.txt
python scripts/download_models.py      # 权重约 773 MB
python web_app.py                       # http://localhost:5000
```

## 约定

- 权重路径由 `Config.AI_MODEL_PATH` 控制，默认 `model/iter_40000.pth`；GUI 下拉里另可选 W48。
- 处理的七个阶段顺序不可随意调换：预处理 → 推理 → 合并 → 全局填洞 → 轮廓校验 → 孔洞填充 → 平滑 → 简化 → 输出。
- 平滑必须用 `join_style=mitre`，不要换成圆角 —— 建筑边是直的，圆角化会让 GIS 拓扑检查对不上。
- 前端资源内联是有意为之（离线可用、无 CDN 依赖），不要拆成外部文件。
- `mmcv.ops` 缺失时的 polyfill 在 `extract_buildings.py` 顶部，不要删，否则 mmcv-lite 环境无法运行。

## 已知坑

- 输入必须是带地理参考的影像，否则面积过滤与 SHP 写出都会失效。
- 黑边误检靠推理前的掩码涂黑解决，不是后置面积过滤能拦住的。
- `web_app.py` 里 `MAX_CONTENT_LENGTH` 设为 16 GB，改小会挡住大影像上传。
