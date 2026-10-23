# Enflame dispatch unit tests 排除说明

## 结论

`tests/platforms/enflame.yaml` 当前排除下面四个文件，并不是因为这些文件中的
所有测试都不支持 GCU，也不是因为 dispatch 实现在 GCU 上发生了功能错误：

```yaml
exclude:
  - dispatch/test_call_op.py
  - dispatch/test_env_policy.py
  - dispatch/test_manager.py
  - dispatch/test_policy.py
```

根因是部分通用测试把 `reset_global_policy()` 当成了“恢复
`SelectionPolicy()` 的代码内建默认值 `flagos`”。实际上，该函数只清空缓存；
下一次 `get_policy()` 会重新按照环境加载有效策略。在 Enflame GCU 环境中，平台
自动检测会加载 `sglang_fl/dispatch/config/gcu.yaml`，其中明确配置了
`prefer: vendor`。因此 GCU 上的有效默认值和选择顺序分别是：

```text
prefer = "vendor"
order  = ["vendor", "flagos", "reference"]
```

而没有平台配置时的代码内建默认值才是：

```text
prefer = "flagos"
order  = ["flagos", "vendor", "reference"]
```

当前 exclusion 是避免这些错误假设阻塞 Enflame CI 的临时措施，而且粒度过大：
每个文件中不依赖隐式默认策略的测试也被一并跳过了。

## 策略加载过程

相关代码的调用关系如下：

1. `reset_global_policy()` 将 `PolicyManager._global_policy` 设为 `None`，并不创建
   `SelectionPolicy()`。
2. 随后的 `get_policy()` 发现缓存为空，调用 `_policy_from_env()`。
3. `_policy_from_env()` 按以下优先级解析策略：
   `SGLANG_FL_CONFIG`、有效的环境变量、自动检测的平台配置、代码内建默认值。
4. GCU 环境中 `torch.gcu.is_available()` 为真，`get_config_path()` 返回
   `sglang_fl/dispatch/config/gcu.yaml`。
5. 当 `SGLANG_FL_PREFER` 未设置、为空或值非法时，有效的平台配置仍然存在，
   因此 `prefer` 回退到平台默认 `vendor`，而不是代码内建默认 `flagos`。

这里需要区分两个概念：

- **代码内建默认值**：直接构造 `SelectionPolicy()` 时得到 `flagos`；
- **运行环境的有效默认值**：经过环境变量和平台配置解析后得到的值，GCU 上是
  `vendor`。

## 各文件的问题和报错

| 文件 | 确定受影响的测试 | 测试预期 | GCU 实际值 | 典型报错 |
| --- | --- | --- | --- | --- |
| `test_call_op.py` | `TestFullPipeline::test_full_pipeline_default_preference` | `call_op()` 选择 `default.flagos`，返回 `"flagos"` | vendor 优先，选择测试注册的 `vendor.cuda`，返回 `"vendor"` | `AssertionError: assert 'vendor' == 'flagos'` |
| `test_env_policy.py` | `TestEnvVarPrefer::test_prefer_invalid_falls_back_to_default`、`test_prefer_empty_string` | 非法值或空值回退到 `PREFER_DEFAULT`（`"flagos"`） | 环境变量没有形成有效覆盖，继续使用 GCU 平台默认 `"vendor"` | `AssertionError: assert 'vendor' == 'flagos'` |
| `test_manager.py` | `TestOpManagerResolve::test_resolve_default_prefers_flagos`、`TestOpManagerCache::test_cache_invalidated_by_policy_change` | manager 首次解析到 `flagos_silu` | manager 按 vendor-first 顺序解析到测试注册的 `cuda_silu` | `AssertionError: assert 'cuda_silu' == 'flagos_silu'` |
| `test_policy.py` | `TestPolicyManager::test_default_policy`、`TestGlobalPolicyFunctions::test_get_default`、`test_policy_context`、`test_with_preference` | reset 后以及 context 退出后恢复为 `"flagos"` | reset 后以及 context 退出后恢复为 GCU 有效默认值 `"vendor"` | `AssertionError: assert 'vendor' == 'flagos'` |

### `dispatch/test_call_op.py`

`test_full_pipeline_default_preference` 同时注册了 `default.flagos` 和
`vendor.cuda` 两个测试实现。测试在没有显式 policy context 的情况下调用
`call_op()`，并断言返回 `"flagos"`。在 GCU 上，reset 后加载的默认顺序以
vendor 开头，所以会先调用 `vendor.cuda`。这里的 `cuda` 只是该单元测试给模拟
vendor 实现使用的名字；失败不表示 GCU 实际执行了 CUDA 代码。

该文件的其他测试主要验证 builtin 注册、vendor discovery 和 callable 解析，
并不都依赖 GCU 的默认策略。因一个默认策略断言而排除整个文件会损失这些覆盖。

### `dispatch/test_env_policy.py`

两个失败测试把 “falls back to default” 定义成了无条件回退到常量
`PREFER_DEFAULT == "flagos"`。这与 `_policy_from_env()` 的真实优先级不符：非法或
空的 `SGLANG_FL_PREFER` 只是不能覆盖平台配置，GCU 平台配置仍应生效。

显式设置 `vendor`、`reference` 或 `flagos` 的测试，以及 strict、vendor filter、
per-op 和显式配置文件相关测试，不存在这个默认值冲突。

### `dispatch/test_manager.py`

`populated_manager` fixture 同时注册 `flagos_silu`、`ref_silu` 和
`cuda_silu`。两个失败测试在 `reset_global_policy()` 后假定首次解析一定选择
`flagos_silu`。GCU 的有效策略为 vendor-first，所以首次解析得到
`cuda_silu`，断言在真正验证 manager 解析或 cache invalidation 之前就失败了。

显式使用 `with_preference(...)` 或 `SelectionPolicy(...)` 的测试不依赖平台默认
值，不应随整个文件一起排除。

### `dispatch/test_policy.py`

此文件同时测试了两种不同层次的行为：

- `SelectionPolicy()` 的纯对象默认值；
- `PolicyManager` 结合环境变量和平台配置后返回的有效策略。

`TestSelectionPolicy::test_default_values` 断言前者为 `flagos`，这个断言仍然正确。
失败的四个测试则把后者也固定断言为 `flagos`。另外，policy context 退出时恢复的
是进入 context 前的有效全局策略；GCU 上应恢复 `vendor`，并非 `flagos`。

## 修复方案

修复目标是让测试明确自己验证的是“代码内建默认”还是“平台有效默认”，而不是
修改 GCU 的生产策略来迁就测试。

### 1. 为测试提供两类明确的 fixture

需要验证代码内建默认值的测试应屏蔽平台配置。例如可以提供类似下面的 fixture
（实现时应在完整测试环境中验证 patch 路径和缓存清理）：

```python
@pytest.fixture
def builtin_default_policy():
    with patch(
        "sglang_fl.dispatch.config.get_config_path",
        return_value=None,
    ):
        reset_global_policy()
        yield
        reset_global_policy()
```

另一类测试保留平台配置，专门验证有效默认策略。应增加 GCU 正向用例，明确验证：

```text
auto-detect gcu -> load gcu.yaml -> prefer == "vendor"
```

这样能够同时覆盖内建 fallback 和平台配置，且测试结果不依赖执行机器。

### 2. 修复 `test_call_op.py`

该测试的目标是验证完整的 register → resolve → call 流程，而不是平台探测。
调用 `call_op()` 时应显式使用 `policy_context(SelectionPolicy(prefer="flagos"))`，
随后再显式切换到 vendor，分别验证两条路径。这样保留原测试意图，并消除运行平台
对结果的影响。

### 3. 修复 `test_env_policy.py`

应把当前两个用例拆清楚：

- 验证“非法/空环境变量回退到代码内建默认”时，mock 掉平台配置；
- 验证生产加载语义时，断言非法/空环境变量回退到平台默认，并分别覆盖有平台
  配置和无平台配置的情况。

不应简单把预期值从 `flagos` 改成 `vendor`，否则测试只会从“仅在无平台配置环境
通过”变成“仅在 GCU 环境通过”。

### 4. 修复 `test_manager.py`

manager 和 cache 测试应在调用 `resolve()` 前显式进入 flagos policy context。
cache invalidation 用例再切换到显式 vendor context，验证 policy fingerprint 变化后
使用新的 cache key 重新解析。测试关注点由此仍是 manager cache，而不是当前机器的
平台配置。

### 5. 修复 `test_policy.py`

- `SelectionPolicy()` 的对象默认值测试继续断言 `flagos`；
- 测试内建 fallback 的 `PolicyManager` 用例使用无平台配置 fixture；
- context 恢复测试先显式设置一个已知全局策略，或记录进入 context 前的策略并
  断言退出后恢复该值；
- 另加平台配置加载测试，覆盖 GCU 的 `vendor` 默认。

## 不应采用的处理方式

- 不应把 `gcu.yaml` 改成 `prefer: flagos` 以使测试通过；这是 GCU 的生产 dispatch
  策略，不是测试 fixture。
- 不应在 Enflame CI 全局设置 `SGLANG_FL_PREFER=flagos`；这会掩盖真实的平台默认
  行为，并使测试环境偏离运行环境。
- 不应长期保留四个整文件 exclusion；其中大多数测试与该冲突无关。
- 不应通过 FlagGems blacklist 处理；失败发生在 dispatch policy 选择和测试断言
  阶段，不是某个 FlagGems 算子的正确性问题。

## 修复后的验收标准

1. 在无平台配置的环境中，内建默认相关测试稳定得到 `flagos`。
2. 在模拟或真实 GCU 环境中，平台默认相关测试稳定得到 `vendor`。
3. 显式 policy context 的 call、manager 和 cache 测试在所有平台得到相同结果。
4. 从 `tests/platforms/enflame.yaml` 移除四个整文件 exclusion 后，四个文件全量通过：

   ```bash
   pytest -v \
     tests/unit_tests/dispatch/test_call_op.py \
     tests/unit_tests/dispatch/test_env_policy.py \
     tests/unit_tests/dispatch/test_manager.py \
     tests/unit_tests/dispatch/test_policy.py
   ```

5. Enflame 的完整 unit scope 通过，且 GCU 平台配置仍保持 `prefer: vendor`。

## 报错证据说明

上表中的失败测试、实际值和断言错误可由当前测试代码、策略加载优先级以及
`gcu.yaml` 确定。引入 exclusion 之前的历史 Enflame unit job 日志目前已无法从
GitHub 下载，因此这里列出的是对应断言产生的典型 pytest 错误，而不是声称逐字
摘录历史 CI traceback。
