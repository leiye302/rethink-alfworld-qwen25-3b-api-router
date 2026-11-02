# 配方与实现审核记录

本页记录代码差异和验证边界，不发布私人历史实验日志，不把其他模型/其他硬件的成绩冒充本消融结果。

## 来源

- 母仓库公开版本：`rethink-alfworld-qwen25-1.5b`，commit `93e1341ffab835c21ed622ac80793fc1fb7f1487`。这里只把它作为可独立运行的代码模板；科学配方采用其附带的冻结3B reference，而非将1.5B训练结果迁为3B结果。
- 原始3B科学合同：configs/reference_3b.json。基模Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`。
- SDAR commit：`d511f043afe9199b8576fc99ea52e12d84841f32`。完整方法patch及本组patch的SHA见配置；静态skill、GRPO/SDAR数学实现、PPO Actor loss不因消融改写。
- SFT数据包字节、selection IDs、任务拆分、seen140清单与原公开3B数据一致。没有新采集、重标或重抽样。

## trigger_only 的唯一算法变化

1. Bad的真实训练序列止于RETHINK，纠正action/EOS不进入模型输入/target；Good不变。
2. Bad只保留Full后缀CE中的marker那一项，分母仍为原完整后缀token数；独立BCE系数0.1不变。
3. 编码器/损失/preflight/训练入口增加显式supervision_mode与错误配置保护，清楚记录没有纠正监督。
4. Stage2仍是Actor marker选位，hint原协议/原模型/同轨迹合并，Teacher-only；SDAR patch与完整方法相同。

## external_api_router 的必要变化

1. 无Route-SFT，直接固定base开始RL；模型无须已注册单token marker。只有外置组允许未注册marker的原样采样记录，不强制注入token。
2. 复用原rollout的完整 `rh_episode/rh_step` 记录；用实际response/attention_mask核对该步非空、单一think块及R对齐。每条唯一整轨迹只请求一次，审核者可看后续真实反馈，但仍只把实质 reasoning 错误判 Bad，不能将 action 错误或最终失败自动等价为 Bad。
3. 同一响应返回全部step的Good/Bad/uncertain，以及Bad的原R引文、错误说明和短hint。不替换Actor batch、marker metadata、response、reward或GRPO mask。报表分别记录API选择率与Actor marker率。
4. hint使用精确命令ID、只基于真实反馈重建的行动前状态，以及部分目标前提过滤。某条轨迹的hint被拒绝时集中补一次，补请求不允许重新选位或改标签；仍不合格保留Bad但不给额外hint。Good/uncertain使用普通静态skill Teacher。
   Teacher里的hint边界字符串沿用原模板以减少提示格式变化；这个字符串不作为Actor真实触发的证据，报表分别记录Actor marker与API选位。
5. 新API传输保留HTTPS、拒绝带凭据URL/重定向、缓存、重试与故障hold策略。使用真实User-Agent、持久私有会话及可选运行者组织头；不发布标识值，不自动切换付费提供商。整轨迹、route模型/prompt/源码/端点均进入身份绑定，缓存内容校验失败会停止。
6. 训练期Teacher评分接入hook；评测路径明确关闭两类API。Teacher仍是当前Actor、固定reference仍是本组RL起点。

## 工程适配，非新增科学变量

后续兼容修复：OPSD_PYTHON为裸命令时按PATH查找，相对路径按仓库根定位，
并在训练前验证，防止长训练结束后导出失败。结果比较只接受已逐项核对的
两份等价seen140 JSON序列化SHA；任务顺序、种子、game内容和split仍锁定。
这些修复不改训练数据、loss、采样设置或评测任务，附独立回归测试。

母模板曾配置4卡/组、两组并行；本独立消融恢复单卡入口。全局batch、采样、梯度缩放和SDAR公式保持3B reference，去掉自动启动额外baseline。API组保留未启用的原SFT模块只为原始配方/单元测试审核，主入口禁止误启动。

准备中修正了启动脚本与SFT训练器CPU preflight文件名不一致的问题，并强制校验监督模式，避免把Full preflight当trigger_only使用。修正本地预检报表中的旧每rank样本数；这些不改变优化目标。

## 检查与未完成项

CPU测试覆盖：固定数据与清单、mask/预测位置、全词表BCE、Bad无纠正/EOS输入及局部CE权重、Good不变、LR前2000步、断点恢复、整轨迹请求白名单、行动前状态不读本步或未来反馈、原R引文、精确动作ID、hint-only补请求、缓存与故障恢复、Actor不变、无API评测、独立打包和隐私白名单。

具体执行通过数由CI/交付检查报告记录。已进行小规模真实API协议/传输试测；**没有据此宣称全面语义质量校准通过，也未执行本消融的新GPU训练、CUDA/NCCL/vLLM完整链路和seen140性能评测。** CPU控制台成功不是这些项目的验收。本仓库不发布私人试测输入、账号或日志，没有填入成功率或声称优于完整方法。
