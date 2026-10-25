# 团队数据交接：已实现部分

## QGIS → PINN

PINN只读规范文件，不依赖QGIS、不重投影、不填洞/填洼。输入读取器为`TerrainGrid.from_files`。

`terrain.csv`字段`cell_id,row,col,x,y,z`，UTF-8，米制投影，格中心；row南→北、col西→东，ID=row*nx+col。完整规则矩形，每方向至少2格；内部NoData、缺格、非有限或不规则域报错。高程可以小于零。

`terrain_metadata.json`实际字段：schema_version、grid_id、terrain_id、terrain_version、synthetic（布尔）、crs、crs_type（真实资料须projected）、axis_order=x_east_y_north、horizontal_unit=m、vertical_unit=m、vertical_datum、x_edge_origin、y_edge_origin、dx/dy、nx/ny、extent=[xmin,xmax,ymin,ymax]、nodata_policy。建议附surface_type、来源/处理履历，sha256和roughness_sha256有值时必须匹配文件。原点是西南外边缘。查看`data/examples/undulating/terrain_metadata.json`了解实例，**不要将合成CRS改名冒充民雄**。

真实CRS需证明是米制投影；可选pyproj仅用于识别验证，不做GIS转换。尚未安装pyproj时仅保守接受已知米制投影，其他CRS明确要求补验证能力；经纬度及单位不符不能直接进入物理方程。

粗糙度文件`cell_id,manning_n`，单位s/m^(1/3)，正值且完整覆盖；否则必须明确传常数manning_n，不能同时选两者。元数据和网格数组公开接口不可变，地形/粗糙度/元数据形成固定指纹。

当前IC/BC实现只有显式干域和四壁；真实水文条件、开放边界/乡界掩膜需要提供资料并扩展模型侧支持。拟定的水位/流量/初值CSV契约见design_v0.2，**尚不是已实现的读取/求解功能**。

## PINN → API

入口是`Predictor.from_checkpoint(model_dir, grid, conditions=...)`和`predict(rainfall_mm_hr,duration_min,output_times_s,output_mode="complete",batch_size=4096,output_dir=None,allow_unvalidated=False)`。

雨强mm/hr，duration_min只接受分钟，小时由调用方显式转换；内部R为m/s、雨时和输出时间为秒。模型固定T_end，不用最后一个请求时间冒充模拟结束。完整模式补事件时间；请求模式返回partial，摘要峰值仅代表采样范围。

迭代`frames_iterator`消费帧；每帧字段包含cell_id、x/y/z/t/h/water_surface/u/v/speed/direction/flooded和原始速度。空间ID不变，帧方向无定义为None。`finalize()`在消费结束前只返回pending；消费结束后完成可选导出并生成最终状态。耗时评估包含消费和finalize，不能只计创建迭代器的时间。

错误为`PredictionError`，可调用`as_dict()`获得code/message/details。包括INVALID_ARGUMENT、TIME_OUT_OF_RANGE、UNVALIDATED_SCENARIO、TERRAIN_VERSION_MISMATCH、CONDITION_VERSION_MISMATCH、MODEL_INCOMPATIBLE、NONFINITE_PREDICTION、OUTPUT_IO_ERROR。输入文件读取阶段可抛带原因的ValueError/OSError；CLI统一输出WORKFLOW_ERROR。API成员负责HTTP映射、任务状态和并发，给每个run独立输出目录。

预测不调用训练；单情境模型拒绝换雨情；参数化模型默认也拒绝无验证域证据的雨情。显式研究开关不豁免数值、版本、时域或训练参数范围检查。`evaluate`是未执行情境准入的低层研究函数，API不得用它绕过`predict`。

数据包正常文件含manifest（SHA-256/字节/CSV行数）。导出有独占锁、暂存和无覆盖发布；若进程被强制杀死留下锁/暂存，应先由API检查该任务已终止、归档残留，再使用新目录，不能自动把半成品标完成。导出结束后添加文件应调用`refresh_manifest`。

## PINN → UE

源CSV保持SI：x/y/z/h/water_surface为m，t为s，u/v/speed为m/s。流向0°东、90°北、逆时针；干格和静止格方向为空，不能用0°代替未知。

UE成员确认anchor和正交轴映射B后转换：点=100*B*(GIS点−anchor)，速度=100*B*(u,v,0)，高程用相同垂直基准；水面=z+h。`coordinates.py`仅提供数学换算及回归样例，不是UE插件。GIS大坐标应先double减原点；UE轴向/手性确认后由转换速度重算yaw。

四边交换：flow_exchange每个唯一面/区间一行，法向固定+x/+y，minus/plus为两侧格。Q为m³/s，正反向时间积分V为m³。cell_water_balance每格四边引用同一面，不重复计算。原始场计算物理交换，干格显示速度掩膜只影响可视化；不把显示场与原始场混做守恒结论。

PINN侧CSV/JSON回读、稳定ID、方向/单位及坐标数学测试不等于已通过真实API/UE端到端联调。场景、材质、动画和服务器均由对应成员实现。
