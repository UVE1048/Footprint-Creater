# 模型权重目录 / Model weights

权重文件 **不入库**，通过 GitHub Releases 分发（总计约 773 MB，超过仓库合理体积）。

```bash
python scripts/download_models.py
```

脚本会下载并做 SHA256 校验，产物落在本目录：

| 文件 | 骨干 | 训练轮次 | 体积 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `iter_40000.pth` | HRNet-W32 + FCNHead | 40000 iters | ~231 MB | 默认。速度快，适合批量生产 |
| `hrnet_w48_iter_6000.pth` | HRNet-W48 + FCNHead | 6000 iters | ~542 MB | 精度更高，显存占用与耗时约翻倍 |

只下其中一个：

```bash
python scripts/download_models.py --model w32
```

校验现有文件：

```bash
python scripts/download_models.py --check
```

## SHA256

```text
14eb942156ff3e520362329da855daeb8d15bf887a242530b52f29de9e67ea0c  iter_40000.pth
73db3f97998fc62cbbcb83897925d1c6ffaef2e1dadd78b5b7f90267dab5cf18  hrnet_w48_iter_6000.pth
```

> **买不到现成权重？** 也可以用 [HRNet-Footprint-Trainer](https://github.com/UVE1048/HRNet-Footprint-Trainer)
> 在自己的数据上训练一份出来 —— W48 的那个权重就是这套流程产出的。
