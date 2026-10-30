# 固定地形、手动雨情、VS Code启动与成果报告

## 地形不会随机变化

默认运行读取：

- `data/examples/undulating/terrain.csv`
- 同目录`terrain_metadata.json`、`roughness.csv`

这些是已保存的固定起伏地形文件。每次加载后计算高程/坡度，记录地形指纹；改变雨势不会修改CSV或地形。训练会随机选择时空采样点来计算Loss，不代表随机生成地形。上述数据明确标记为synthetic，不是民雄实际地形。

实际DEM到位后，在`configs/fixed_terrain.json`修改terrain文件路径（相对该配置文件）；按真实空间/时间范围配置尺度、初边界和雨情列表，重新训练验证。不要仅把合成数据名称改成民雄；当前只支持完整规则矩形、干初始和四壁条件，真实开放边界/掩膜仍需适配。

## VS Code首次准备

使用64位Python 3.12。VS Code中选择“文件→打开文件夹”，打开含`run_in_vscode.py`、`configs`和`.vscode`的仓库根目录。使用Python、Python Debugger两项推荐扩展。

打开命令面板`Ctrl+Shift+P`：

1. **Python: Create Environment**。
2. 选择**Venv**及Python 3.12。
3. 选择**requirements.txt**让扩展安装依赖。
4. 通过**Python: Select Interpreter**确认选中了项目`.venv`解释器。

无需为VS Code新入口手动运行pip或输入Windows终端命令。依赖安装仍需要网络；VS Code扩展会显示安装进度。已有虚拟环境时只需选解释器。缺少torch/numpy等会在调试控制台及失败报告显示，不能仅更换系统Python路径假定依赖已安装。

`.vscode/launch.json`使用跨平台debugpy、工作区根目录和`internalConsole`；不写死某台电脑的Python安装路径。Linux/macOS也可选择本项目解释器运行。

## 编辑雨情并启动

在`configs/vscode_run.json`编辑：

```json
{
  "action": "train_and_predict",
  "base_config": "fixed_terrain.json",
  "rainfall_mm_hr": 60,
  "duration": {"value": 0.05, "unit": "min"},
  "output_root": "../outputs/vscode",
  "model_dir": null,
  "allow_unvalidated": true,
  "batch_size": 4096,
  "open_report": true
}
```

雨强单位mm/hr；雨时必须带`s`、`min`或`h`。这里0.05min=3s，总模拟时间6s，属于极短示范。不要直接填30分钟期待6秒训练时域的旧模型有效。总时间、网络尺度、训练雨情和适用范围在base_config单独定义。

按F5，选择“PINN：固定地形、手动雨情与训练报告”。日志出现在VS Code的“调试控制台”；没有`input()`交互，也不要求进入Windows Terminal。第一次按F5如提示安装Python Debugger，按扩展提示安装即可。

日志依次说明固定CSV、地形指纹、手填雨情、是否训练及是否预测。运行完成会显示HTML成果报告路径；`open_report=true`时在有桌面环境的机器自动调用浏览器打开。无桌面的云环境保留文件路径，不强行启动浏览器。

## 以后改变雨势：只预测

首次训练结果类似：

```text
outputs/vscode/<UTC时间_随机运行ID>/
  effective_config.json
  workflow_context.json
  training/
    model/
      weights.pt
      model_metadata.json
    training_report.json
    training_history.csv
    validation_metrics.json
    physics_diagnostics.csv
    *.png
  prediction/
    simulation_results.csv
    flow_exchange.csv
    cell_water_balance.csv
    water_balance.csv
    simulation_summary.json
    grid_metadata.json
    manifest.json
    *.png
  training_results_report.html
  training_results_report.md
```

运行ID中的随机部分只为了避免覆盖旧结果，与地形无关。

将action改为`predict`，model_dir填为上次`training/model`路径。相对路径以`vscode_run.json`所在目录为基准，例如：

```json
"action": "predict",
"model_dir": "../outputs/vscode/<上次运行目录>/training/model",
"rainfall_mm_hr": 40,
"duration": {"value": 0.05, "unit": "min"}
```

保留文件中其他设置，然后F5。此模式不训练，读取同一固定地形并核对版本，只生成新预测及报告。报告明确区分“所加载模型的历史训练记录”和“本次仅推理”。不要把null写成字符串`"null"`，或把布尔true/false写成字符串。

action=`train`只训练并生成报告，不执行预测。`train_and_predict`每次都会按配置重新训练；雨势经常调整时请切为predict。

## 如何阅读成果报告

`training_results_report.html`包含CSS和内嵌图像，可双击在浏览器打开，不需要FastAPI或网页服务器；可在浏览器打印/另存PDF。可单独分享HTML阅读图表，模型/CSV原文件仍在结果目录。Markdown版便于版本管理或复制摘要。

报告包含：固定地形路径、grid_id/指纹/数据来源；手填降雨与模拟时间；训练是否正常结束/收敛/物理通过/外部验证；Adam与LBFGS分项Loss/学习率/时间/梯度；各雨情PDE/IC/BC和参考MAE/RMSE/L2、速度/范围指标；逐格与全域水量；可用图及原始文件链接。

没有参考数据时不能产生可信预测误差，报告标未评估；完成100次短训练不代表收敛，更不代表民雄预测准确。`allow_unvalidated=true`是显式研究模式，只为演示流程，不会把模型认证为已验证，也不允许训练范围外推。正常可信推理应保持false，并先建立通过验证的有效域。

训练异常、数据错误、范围不符或时间预算停止都会尽可能生成失败报告，保留原始诊断。若使用VS Code强制终止整个进程/关闭系统，程序无法保证完成最后的报告写入，可查看已有JSON/CSV；重新启动会创建新目录而不会覆盖旧结果。

## 平台验证边界

移除了CLI对Unix resource的强制依赖、固定/tmp和绘图getuid调用，使用跨平台临时路径；配置/元数据/成果明确使用UTF-8，支持带BOM的用户输入，避免中文Windows默认编码导致乱码或读取失败。Windows无法获取的内存测量写null，不伪造0。已测试模拟Windows分支与Linux实际训练/推理/报告，但没有在原生Windows VS Code上实机执行。结果导出需要支持硬链接的本地文件系统（通常NTFS）；不支持时应改用本地NTFS结果目录，不关闭无覆盖保护。

指南与代码一同同步仓库；每次训练的模型/CSV/报告保留在本机被git忽略的outputs目录，不自动上传。真实QGIS处理、API服务和UE场景仍由对应成员负责。
