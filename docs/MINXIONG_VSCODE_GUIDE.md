# 民雄固定地形 PINN：VS Code 使用方式

这个版本直接读取 `data/minxiong/裁剪后民雄.tif` 的有效性掩膜。NoData 格不会进入训练，高程是否小于或等于 0 不用于判断 NoData，因此未来换入含负高程洼地的 DEM 也会保留负值。

## 在 VS Code 预测

1. 用 VS Code 打开仓库根目录。
2. 选择仓库中的 `.venv` Python 解释器。
3. 到「运行和调试」，选择 **民雄 PINN：用户输入预测**，按 F5。
4. 在窗口输入平均雨势、降雨时长、模拟总时长和输出间隔。
5. 选择输出目录，点击「开始预测」。日常预测不需要编辑任何 JSON。

界面会从已批准模型包读取实际支持范围。目前正式范围为：

- 平均雨势：0–300 mm/hr。
- 降雨时长：1–24 小时。
- 模拟总时长：最长 36 小时。
- 30–50 mm/hr 会自动平滑混合 low 与 medium，画面不会在 40 mm/hr 突然切换。

输出时刻会自动包含0、固定间隔、停雨时刻（若在模拟范围内）及模拟结束时刻。模拟可以在降雨结束前停止；此时界面会提示结束时仍在降雨。最多500帧，超过120帧会先显示容量确认。

每次预测会建立新的输出目录，里面包含：

- `prediction_summary.json`：雨情、时间及各帧最大/平均水深与最大流速。
- `prediction_summary.csv`：每帧的有效模型标签、主专家、混合权重、雨情、时间、文件名及最大/平均水深与最大流速，可直接交给 Unreal 或表格工具。
- `frame_*.csv`：有效格的时间、行列、EPSG:3826 米制坐标、水深、x/y 速度与合速度；不需要从 JSON 转换。
- `frame_*.npz`：有效格的行列、坐标、水深与速度。
- `depth_*.tif`：保留 EPSG:3826 与原始掩膜的水深 GeoTIFF，可直接放进 QGIS。

超出训练雨势或持续时间范围时程序会拒绝外推，并显示原因。

## 0–300 mm/hr 两专家正式状态

正式方案由两个固定地形专家组成：low 训练 0–50；medium 训练 30–300 mm/hr。用户范围为连续 0–300 mm/hr、降雨 1–24 小时、模拟最长 36 小时。重叠的 30–50 mm/hr 采用线性凸混合；40 mm/hr 时两者各占 50%。历史 0–800 失败输出会保留，但不作为已部署范围。

这个范围已经完成以下验收并出现在 GUI：

- low、medium 各自通过 PDE、连续方程、封闭边界法向速度和全域水量门槛；
- 30–50 mm/hr 重叠区通过理想化发布门槛并启用平滑混合；
- 完整模型包已部署到 `models/minxiong/experts/`，且 manifest、检查点和审核证据互相匹配。

若 manifest、模型哈希、审核证据或固定 DEM 不一致，GUI 会明确记录原因并安全回退 v4。完整结果见 `docs/MINXIONG_0_300_TRAINING_STATUS.md`。

`configs/minxiong.json` 与 `run_minxiong_vscode.py` 继续保留给开发、批次复现和训练，不再是一般用户输入预测参数的入口。

## 重新训练

仓库已带有通过物理门槛的模型 `models/minxiong/model.pt`，日常预测不需要重新训练。若要重新训练，把 `action` 改成 `train`，同时把 `training_output_dir` 改成一个不存在的新目录。训练完成后只有在 `physics_training_passed` 为 `true`，并完成人工复核，才替换正式模型。

训练成果报告位于 `reports/minxiong_training_results.html`。报告中的“通过”是研究级物理约束门槛，不是实测淹水精度认证。

0–300 两专家原训练与发布审核流程：

```bash
.venv/bin/python train_minxiong_experts.py configs/minxiong_experts_0_300.json
.venv/bin/python audit_minxiong_experts.py configs/minxiong_experts_0_300_idealized_release.json
```

上面第二个命令是本次带云端 `outputs/` 的历史发布流程；干净 clone 请用 `.venv/bin/python verify_minxiong_release.py` 做只读检查。训练每 50 步写入可恢复的 `training_state.pt`；若地形、事件或关键配置改变则拒绝恢复。发布配置是 `release_only`，不能用于启动训练。当前批准的 manifest、`low/`、`medium/` 及审核证据已整体部署到 `models/minxiong/experts/`。

## Unreal Engine 画面接口

运行 `.venv/bin/python run_minxiong_unreal_api.py` 后，UE5 可通过 `GET /health`、`GET /schema` 和 `POST /predict` 使用同一套范围检查与预测流程。服务直接返回 CSV、GeoTIFF 和 NPZ 的输出路径；UMG/Blueprint 的请求字段、异步调用方式、CSV 列和坐标转换见 [Unreal Engine 对接指南](UNREAL_INTEGRATION.md)。接口与 GUI 使用同一批准专家包；校验失败时才会安全回退 v4。

## 目前假设与仍需资料

- 初始状态为全干。
- 所有 DEM 掩膜外缘暂时视为无流墙。
- Manning n 暂用全域常数 0.03。
- 2024 逐小时记录被整理成连续降雨事件，以事件平均雨势和持续时间作为两个条件变量。
- 尚未加入雨水下水道、入渗、抽水站、建筑物、河道水位或边界入流。

若要把研究模型升级为真实民雄淹水预测，需要提供边界水位/流量、粗糙度分布或土地利用、初始水位，以及至少一组实测淹水深度/范围或可信水动力模型结果。
