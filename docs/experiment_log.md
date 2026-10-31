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
2. 从真实batch的prompts/attention_mask提取Actor实际H，从responses提取已完成的非空R。单独调用API，不传当前action/feedback/future；输出Good/Bad/uncertain。
3. 只建立Teacher提示选择的浅拷贝metadata视图，不替换Actor batch、response、reward或GRPO mask。原marker标记另存，报表不混淆API选择率与Actor触发率。
4. Bad仍调用母方法整轨迹hint生成/验收；system中仅将“Actor已触发”改为“外置裁判选中”，其余指导/评分格式不变。Good/uncertain使用普通静态skill Teacher。
   Teacher里的hint边界字符串沿用原模板以减少提示格式变化；这个字符串不作为Actor真实触发的证据，报表分别记录Actor marker与API选位。
5. 新API传输复用原有HTTPS、拒绝带凭据URL/重定向、缓存、重试与故障hold策略。新的route模型/prompt/cache均进入身份绑定。
6. 训练期Teacher评分接入hook；评测路径明确关闭两类API。Teacher仍是当前Actor、固定reference仍是本组RL起点。

## 工程适配，非新增科学变量

母模板曾配置4卡/组、两组并行；本独立消融恢复单卡入口。全局batch、采样、梯度缩放和SDAR公式保持3B reference，去掉自动启动额外baseline。API组保留未启用的原SFT模块只为原始配方/单元测试审核，主入口禁止误启动。

准备中修正了启动脚本与SFT训练器CPU preflight文件名不一致的问题，并强制校验监督模式，避免把Full preflight当trigger_only使用。修正本地预检报表中的旧每rank样本数；这些不改变优化目标。

## 检查与未完成项

CPU测试覆盖：固定数据与清单、mask/预测位置、全词表BCE、Bad无纠正/EOS输入及局部CE权重、Good不变、LR前2000步、断点恢复、API请求不含action/future、选择metadata不污染Actor、同一hint协议校验、无API评测、独立打包和隐私白名单。

具体执行通过数及环境保存在交付检查报告。**未执行新GPU训练、CUDA/NCCL/vLLM完整链路、真实API语义质量/费用校准和seen140性能评测。** CPU控制台成功不是这些项目的验收。本仓库没有填入成功率或声称优于完整方法。
