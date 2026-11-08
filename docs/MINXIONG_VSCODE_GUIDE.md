# 民雄固定地形 PINN：VS Code 使用方式

这个版本直接读取 `data/minxiong/裁剪后民雄.tif` 的有效性掩膜。NoData 格不会进入训练，高程是否小于或等于 0 不用于判断 NoData，因此未来换入含负高程洼地的 DEM 也会保留负值。

## 在 VS Code 预测

1. 用 VS Code 打开仓库根目录。
2. 选择仓库中的 `.venv` Python 解释器。
3. 打开 `configs/minxiong.json`，保持 `action` 为 `predict`。
4. 修改 `prediction` 里的两个输入：
   - `rainfall_mm_hr`：平均雨势，单位 mm/hr；当前模型可用范围为 0.5–23。
   - `duration_min`：降雨持续时间，单位分钟；当前模型可用范围为 60–2220（1–37 小时）。
5. `times_min` 是要输出的模拟时刻，最长为 2580 分钟。
6. 到「运行和调试」，选择 **Minxiong PINN: fixed DEM train/predict**，按 F5。

每次预测会建立新的输出目录，里面包含：

- `prediction_summary.json`：雨情、时间及各帧最大/平均水深与最大流速。
- `frame_*.npz`：有效格的行列、坐标、水深与速度。
- `depth_*.tif`：保留 EPSG:3826 与原始掩膜的水深 GeoTIFF，可直接放进 QGIS。

超出训练雨势或持续时间范围时程序会拒绝外推，并显示原因。

## 重新训练

仓库已带有通过物理门槛的模型 `models/minxiong/model.pt`，日常预测不需要重新训练。若要重新训练，把 `action` 改成 `train`，同时把 `training_output_dir` 改成一个不存在的新目录。训练完成后只有在 `physics_training_passed` 为 `true`，并完成人工复核，才替换正式模型。

训练成果报告位于 `reports/minxiong_training_results.html`。报告中的“通过”是研究级物理约束门槛，不是实测淹水精度认证。

## 目前假设与仍需资料

- 初始状态为全干。
- 所有 DEM 掩膜外缘暂时视为无流墙。
- Manning n 暂用全域常数 0.03。
- 2024 逐小时记录被整理成连续降雨事件，以事件平均雨势和持续时间作为两个条件变量。
- 尚未加入雨水下水道、入渗、抽水站、建筑物、河道水位或边界入流。

若要把研究模型升级为真实民雄淹水预测，需要提供边界水位/流量、粗糙度分布或土地利用、初始水位，以及至少一组实测淹水深度/范围或可信水动力模型结果。
