# Smart-flood-twin

固定地形、多降雨情境的二维浅水方程参数化 PINN。范围限于模型、训练/验证、独立 Python 推理及数据交接，不包含 QGIS 制图、HTTP 服务或 Unreal Engine 场景。

**当前是可运行的研究原型，不是已验证的民雄预测模型。** 合成小案例的读入→训练→保存→新进程推理→时序/流量/水量导出已运行；短训练的误差和水量门槛尚未通过，默认生产推理会拒绝无验证证据的模型。真实 DEM、粗糙度/初边界、可信参考、雨情范围和精度/性能预算仍待团队提供。

## 在 VS Code 中运行（Windows 也适用）

1. 用 VS Code 的“打开文件夹”打开整个仓库，安装推荐的 **Python** 和 **Python Debugger** 扩展。
2. 首次使用按 `Ctrl+Shift+P`，选 **Python: Create Environment → Venv → Python 3.12 → requirements.txt**。已有`.venv`则使用 **Python: Select Interpreter** 选择它。这一步由VS Code安装依赖，不需要手动在Windows终端输入运行命令。
3. 打开`configs/vscode_run.json`，手动编辑`rainfall_mm_hr`和`duration`（unit支持`s`/`min`/`h`）。当前固定地形由`configs/fixed_terrain.json`中的CSV路径指定，每次读取同一文件，不随机生成。
4. 按 **F5**，选择 **PINN：固定地形、手动雨情与训练报告**。日志在VS Code“调试控制台”显示，结束后打开HTML报告；也可在`outputs/vscode/<本次运行>/training_results_report.html`找到它。

```json
"rainfall_mm_hr": 60,
"duration": {"value": 0.05, "unit": "min"}
```

示例是60mm/hr、降雨3秒、总模拟6秒的短流程测试，不能当作实际暴雨设定。固定起伏CSV是人工示例，并非民雄DEM。随机的是训练采样点，不是地形。训练用的多雨情列表在`fixed_terrain.json`中明确列出；上面的手填雨情用于本次预测，不会静默改写训练情境或外推范围。

`action`可选`train_and_predict`、`train`、`predict`。首次用默认训练+预测；之后仅改雨势时改成`predict`，将`model_dir`填为上次`training/model`的路径，避免重复训练。超出模型训练/验证范围的雨势会报错，需要先定义新范围并重新训练。详细图形操作、文件设置及报告说明见[VS Code使用指南](docs/VSCODE_GUIDE.md)。

报告同时提供自包含HTML和Markdown，汇总固定地形/指纹、手动雨情、迭代/Loss、PDE/IC/BC、参考误差（若有）、水量、图表和异常。训练完成与模型验收分开列出，无参考时明确“尚未完成外部验证”。Windows兼容分支有单元测试，本实例的实际运行验证仍在Linux，未声称已在原生Windows上实测。

## 安装与检查

以下在仓库根目录执行；Python 3.12/Linux CPU 环境已验证，CUDA需另按官方平台要求安装并验证。

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
MPLCONFIGDIR=/tmp/flood-pinn-mpl XDG_CACHE_HOME=/tmp/flood-pinn-cache .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m flood_pinn validate-input configs/synthetic_parametric.json
```

`requirements-cpu.lock`固定本实例已使用的依赖版本；不是对所有Python/OS/GPU组合的支持声明。一般依赖范围见`pyproject.toml`；需要验证其他投影CRS可安装可选`crs`依赖，PINN不做重投影。

## 有限时长合成演示

每个命令使用新的输出目录，保留已有结果。示例参数/地形均为synthetic，非民雄数据；默认示例训练最多60秒优化时间，不启动原脚本的15000次长训练。

```bash
.venv/bin/python -m flood_pinn train configs/synthetic_parametric.json --output outputs/demo_training
.venv/bin/python -m flood_pinn predict configs/synthetic_parametric.json --model outputs/demo_training/model --rainfall-mm-hr 60 --duration-min 0.058333333333333334 --times 0 2 3.5 6 --allow-unvalidated --output outputs/demo_prediction
.venv/bin/python -m flood_pinn assess --metrics outputs/demo_training/validation_metrics.json --simulation outputs/demo_prediction/simulation_summary.json --standards configs/synthetic_standards.json --output outputs/demo_assessment.json
```

最后一个命令对未达标准返回退出码2，并保存失败项；这是正确的诊断结果，不能以删除门槛或改成宽松阈值冒充通过。`--allow-unvalidated`是显式研究覆盖开关，输出一直标为`research_unvalidated`，仍禁止训练范围外推。正常推理去掉该开关后，必须有通过物理、参考和完整留出雨情验证的有效域证据。

另有`configs/synthetic_single.json`（三维单情境基线）及`configs/synthetic_undulating.json`（起伏地形、无外部参考，仅流程演示）。训练命令不会改写输入、原代码或已有输出。较长训练需明确设置`training.allow_long_training=true`及预算，不会由预测函数触发。

## Python交接入口

```python
from flood_pinn.terrain import TerrainGrid
from flood_pinn.predict import Predictor

grid = TerrainGrid.from_files(
    "data/examples/flat/terrain.csv",
    "data/examples/flat/terrain_metadata.json",
    roughness_path="data/examples/flat/roughness.csv",
)
predictor = Predictor.from_checkpoint("outputs/demo_training/model", grid)
result = predictor.predict(
    rainfall_mm_hr=60, duration_min=3.5/60,
    output_times_s=[0, 2, 3.5, 6],
    allow_unvalidated=True,  # 此示例仅研究；不是生产准确度许可
)
for frame in result.frames_iterator:
    pass  # API成员消费结构化帧；方向未知时为None
summary = result.finalize()  # 消费前调用只返回pending
```

函数不训练。模型和网格可在进程内复用。`output_mode="complete"`补齐常规/事件时刻；`"requested"`只返回指定帧并标partial。按帧流式输出，单帧内分批前向；最终物理表按需导出。低层`evaluate`仅供已检查点的研究/求积，不是API生产入口。

## 输出与限制

- 训练：`training_history.csv`、`training_report.json`、`physics_diagnostics.csv`、`validation_metrics.json`、Loss/PDE/有参考时误差图。
- 模型：`model/weights.pt`和`model_metadata.json`，含网络/尺度、固定条件、地形身份、校验和、训练与验证状态。
- 预测：`simulation_results.csv`、`flow_exchange.csv`、`cell_water_balance.csv`、`water_balance.csv`、`simulation_summary.json`、`grid_metadata.json`、`manifest.json`和水深/速度/地形/水量图。CLI另写`benchmark_report.json`。
- 瞬时单位宽流量m²/s、面流量m³/s和区间体积m³严格区分；共享面相消不等于PINN储水严格守恒，余差照实保留。
- 当前实现完整规则矩形网格、明确干初始和四壁条件。真实开放边界、初始积水、乡界掩膜/内部障碍尚未实现，相关配置会报错；不能当成民雄已适用。
- 未包含建筑阻水、入渗、地下排水、源追踪；原FV仍仅用于Case A回归，不是民雄全域可信参考。
- 完整训练恢复、通用GPU性能验证、真实API/UE联调未完成；当前保存的推理checkpoint不等于可精确续训的optimizer/RNG快照。

详见[实现与验证状态](docs/IMPLEMENTATION_STATUS.md)、[团队接口](docs/TEAM_INTERFACE.md)、[FV验证证据](docs/reference_validation.md)、[批准的设计](docs/design_v0.2.md)。
