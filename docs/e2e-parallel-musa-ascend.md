# e2e 并行执行:musa / ascend 串行限制根因排查

> 状态:排查完成(debug PR #150;E1 基线 + E2 同配置复跑均已完成)
> 排查对象:#148 引入的 DeviceScheduler 并行 e2e(每用例独占 `tp_size × pp_size` 连续卡组,子进程注入平台可见性环境变量钉卡)中,`musa` 与 `ascend` 被迫声明 `e2e_serial: true` 的原因。
> 环境变量语义:`MTHREADS_VISIBLE_DEVICES`(musa)/ `ASCEND_RT_VISIBLE_DEVICES`(ascend)按 CUDA_VISIBLE_DEVICES 惯例做每进程逻辑重映射。

## 结论摘要(TL;DR)

| 平台 | 根因 | 置信度 | 短期处置 | owner 行动项 |
|---|---|---|---|---|
| musa | **每进程 `MTHREADS_VISIBLE_DEVICES` 钉卡对 sglang-on-musa 无效**:被钉到尾卡的用例 rank 实际仍落在物理 0..tp-1,与先启动的用例撞显存(OOM),并非此前以为的 MCCL 并发死锁 | 高(日志实锤,见 §2) | 保持 `e2e_serial: true`;或 harness 改用 sglang `base_gpu_id` 偏移(见 §2.4) | mthreads/torch_musa 侧确认容器内每进程可见性变量语义(§2.3) |
| ascend | 唯一一次失败(`libccl_kernel.so` AICPU 内核错误)**同配置不复现**;同池当天多次出现 `drvErr=87` 驱动不可用 flake,指向 910c runner 池宿主/驱动态不稳 | 中(单次失败样本 vs 单次通过样本) | 观察性重开并行(flake 率与 drvErr 同源),或先修池子再开 | 910c 池稳定性(drvErr=87 高频,见 §3.3) |

## 1. 排查方法

debug PR #150(仅启用 musa + ascend,两家去掉 `e2e_serial` 强制并行),`tests/run.py` 新增隔离旋钮:

- `FL_E2E_PIN=nopin`:调度器仍独占分配卡位,但子进程继承环境可见性(隔离"钉卡掩码"变量)
- `FL_E2E_STAGGER=<秒>`:错峰启动(隔离初始化竞态)
- 常开诊断行:设备计数、环境可见性值、模式

## 2. musa:钉卡无效导致的显存撞车(非 MCCL 死锁)

### 2.1 背景

#148 将 musa 设为串行的依据是 vllm-plugin-FL #581 的结论("MCCL 同节点并发 TP 组初始化死锁")。本仓库此前从未第一手尝试过并行。

### 2.2 证据链(E1,run [39887129522](https://github.com/flagos-ai/sglang-plugin-FL/actions/runs/39887129522),S5000 × 8 卡,容器环境 `MTHREADS_VISIBLE_DEVICES=all`)

serving 用例调度(时间线,均为钉卡模式):

```
02:04:16 START e2e/serving/qwen3/4b_tp2        MTHREADS_VISIBLE_DEVICES=0,1     (tp2)
02:04:16 START e2e/serving/qwen3_6/35b_a3b_tp4 MTHREADS_VISIBLE_DEVICES=2,3,4,5 (tp4)
02:05:27 FAIL  35b_a3b_tp4   ← 71 秒即败
02:06:07 FAIL  27b_tp4(接续 2,3,4,5)  ← 40 秒即败
02:06:46 PASS  4b_tp2
```

关键报错:

- 35b_tp4(rank1):`torch.OutOfMemoryError: MUSA out of memory ... GPU 1 has a total capacity of 79.92 GiB of which 205.55 MiB is free. Of the allocated memory 15.05 GiB is allocated by PyTorch` —— 本进程只分配了 15 GiB,但卡已被别人占满:~64.7 GiB 的占用方正是 **4b_tp2 的 `mem_fraction_static=0.8 × 80 GiB` 预留**。
- 27b_tp4:`RuntimeError: The memory capacity is unbalanced. Some GPUs may be occupied by other processes. pre_model_load_memory=15.09, local_gpu_memory=78.03`
- inference 用例同构:4b_tp2(0,1) + 06b_tp1(2) + 35b_tp4(3,4,5,6) 并发,35b/27b 相继 OOM,4b/06b 通过。

### 2.3 根因判定

35b_tp4 被钉到 `2,3,4,5`、27b 被钉到 `2,3,4,5`,若钉卡生效,其 rank0/rank1 应落在物理卡 2/3 —— 该区间**没有其他用例**;但报错显示其卡上有 4b_tp2 的 64 GiB(物理卡 0/1 才有)。唯一自洽解释:**引擎 rank 实际落在物理 0..tp-1,每进程 `MTHREADS_VISIBLE_DEVICES` 未生效**(被容器运行时固定、被 torch_musa/mthreads 侧忽略、或 sglang scheduler 子进程重新枚举全量物理卡,三者之一,需 owner 侧确认)。

同时排除旧假设:两次失败均为快速 OOM(71s/40s),无任何 MCCL 集合通信挂死/超时签名;"死锁"叙事在本栈不成立。

### 2.4 出路

- **owner 侧(根治)**:mthreads/torch_musa 确认容器(DooD,宿主 docker socket)内 `MTHREADS_VISIBLE_DEVICES` 每进程语义是否支持逻辑重映射;若不支持,给出官方多引擎隔离方案。
- **harness 侧(绕过)**:sglang `ServerArgs` 原生支持 `base_gpu_id` + `gpu_id_step`(见 E1 日志 server_args dump)。调度器可改为对 musa 注入 `--base-gpu-id <first_slot>` 引擎参数而非可见性环境变量,不依赖 mthreads 语义。实现量小,可作为后续 PR。

## 3. ascend:失败一次性,同配置不复现;池子不稳

### 3.1 唯一失败样本(#148 验证 run [39734618368](https://github.com/flagos-ai/sglang-plugin-FL/actions/runs/39734618368),910c × 4 卡)

```
06:50:20 START e2e/serving/qwen3/4b_tp2  ASCEND_RT_VISIBLE_DEVICES=0,1  (tp2;tp4 用例排队,实际单跑)
06:54:11 FAIL  4b_tp2
  E39999: The error from device(chipId:0, dieId:1) ... AICPU execution ... msg:execute kernel param invalid
  Aicpu kernel execute failed, device_id=1, stream_id=182, task_id=1, soName=libccl_kernel.so
06:54:11 起全量掩码 0,1,2,3 的两个 tp4 serving 用例 → 均通过
```

### 3.2 反证(同头、同池、同配置)

- 同 run 的 inference 任务:`4b_tp2`(钉 `0,1`)与 `06b_tp1`(钉 `2`)**真并发**通过 —— 排除"钉卡子集掩码"与"并发"单独或组合作为确定性触发条件。
- debug E1(run 39887129522,次日):serving `4b_tp2` 钉 `0,1` **原样配置通过**(02:07:10–02:12:56),后续 tp4 全量掩码均通过。

### 3.3 池稳定性背景

flagcicd-910c 池连续两日多次出现 check.sh 阶段 `drvErr=87`(get platform info failed / driver error:internal error):#148 验证期间 3 次(重掷解决)、debug E1 1 次(inference 任务)、E2 1 次(functional 任务,阻断整条 e2e 链)——两日内 ≥5 次,且全部发生在任何测试代码执行之前。AICPU `libccl_kernel` 错误与 drvErr=87 同属宿主驱动态问题谱系;单发、不复现、与驱动失联高发同池同时段,倾向归因为**宿主瞬时态**而非钉卡/并发的确定性缺陷。

### 3.4 建议

- 观察性重开并行(ascend 去掉 `e2e_serial`),失败按 flake 处理重掷;若 libccl 错误率显著,再回退串行并升级 910c 池排查。
- 更优先:910c 池 drvErr=87 高发本身值得 owner 侧排查(已连续两天影响 CI,含非并行任务)。

## 4. 复现与工具

- debug PR #150:`FL_E2E_PIN` / `FL_E2E_STAGGER` 旋钮见 `tests/run.py` `_run_e2e_parallel`(诊断行 `E2E diag:` 打印设备数、环境可见性、模式)。
- E1:run 39887129522(musa 双 OOM、ascend serving 钉卡通过、ascend inference 撞 drvErr=87)。
- E2(run [39890037150](https://github.com/flagos-ai/sglang-plugin-FL/actions/runs/39890037150),同配置复跑):
  - **musa 确定性 2/2**:失败逐位复现 —— 相同掩码分配、相同失败顺序(4b/06b 过,两个 tp4 在 40–70 秒 OOM)、相同错误类(`OutOfMemoryError: MUSA out of memory ... GPU 0 ... ~50–130 MiB free` 与 `memory capacity is unbalanced`)。
  - ascend:E2 中其 functional 任务在 check.sh 阶段再撞 `drvErr=87`(两日内第 5 次,见 §3.3),e2e 链路被阻断;ascend 并行 serving 的通过样本仍以 E1 为准(1 次通过 vs 1 次 #148 失败,均为单样本)。

## 5. Owner 行动项汇总

1. **musa owner(mthreads/torch_musa)**:确认容器内每进程 `MTHREADS_VISIBLE_DEVICES` 逻辑重映射是否受支持;给出多引擎同节点隔离的推荐做法。harness 侧可先行以 `base_gpu_id` 绕过。
2. **ascend owner(910c 池)**:排查 `drvErr=87` 高发(两天内 ≥4 次,阻塞含串行在内的各类任务);确认 2026-10-22 06:53 UTC 时段宿主 AICPU/driver 状态是否异常。
3. **CI 侧(本仓库)**:musa 维持串行直至 1 有结论;ascend 可观察性重开并行(或与池子修复同步进行)。
