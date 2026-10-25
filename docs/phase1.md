# 第一批：Case A 基线与物理修复

历史记录：后续已扩展数据、参数化训练、推理和导出；当前状态见`IMPLEMENTATION_STATUS.md`。

原始文件保存在legacy目录，不通过导入执行。flood_pinn.case_a抽取原网络、采样和Loss，保留隐藏层和输出变换；修正守恒Manning摩擦；参考锚点改为显式传参。make_lbfgs_closure在创建时冻结点集。flood_pinn.reference修正HLL额外分母、精确快照和停雨事件截步。

测试：`.venv/bin/python -m unittest discover -s tests -v`。

本批没有执行完整训练，没有真实DEM，不代表民雄预测精度通过。原FV的截零、动量裁剪与复杂地形静水保持仍需后续验证；该求解器暂不作为可信全乡参考。下一批继续建立通量/守恒/摩擦基准及标准化地形接口。依赖范围为初始兼容范围，不是冻结锁文件。
