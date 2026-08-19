# D-5 构念效度修订与 Amazon Operational Transfer Acceptance v2

状态：在查看 Test2、Jacksonville 或 full corpus 之前冻结。本文只授权读取已经完成的
140-family non-release pilot；不授权重新生成 instance、full 7,500、archive/restore 或
profile promotion。

## 1. 修订结论

Stage-2 的证据边界冻结为：

```text
Amazon controls operational templates.
CLE controls target-city geography.
```

Amazon Last-Mile 2021 可以支撑 route/stop 数、depot-to-stop 径向时间、package、volume、
service time、time window、weekday/weekend 和订单属性联合结构的迁移。匿名 Amazon
station 不能为纽约、Los Angeles、San Antonio 等目标城市规定局部道路形态。

因此：

- M1 径向时间保留为 Amazon operational transfer hard gate；
- M4 route/region size 保留为 exact construction hard gate；
- package、volume、service time、time window、day type、source provenance 和 matching
  bias 进入 `amazon_operational_transfer_acceptance_v2`；
- M2、M3、M5 进入 `cross_city_spatial_diagnostic_v1`，只报告，不决定 PASS/FAIL；
- 旧 `station_block_q90_m2_m3_v1` 不删、不改阈值，也不改写为 PASS。

论文冻结表述：

> M2/M3 characterize local road-network morphology and spatial concentration.
> Because these quantities are city-specific and the Amazon locations are
> anonymized, they are reported as cross-domain diagnostics rather than enforced
> as transfer constraints.

## 2. 为什么这是构念修复，而不是事后放宽

历史 Q90 v1 在 24 个 `day_type × scale × component` 行上得到 0/24 PASS，
generated-to-real Q90 比例为约 1.46–6.89。该结果永久保留，并明确记录为触发
construct-validity review 的证据。

本修订没有：

- 把 Q90 阈值从 1 调到 6.89；
- 删除 M2/M3 pairing、distance、bootstrap 或 Q90 输出；
- 用 Test2、Jacksonville 或 full corpus 调参；
- 修改任何 family、matrix、manifest、seed 或 view；
- 重新运行生成器。

改变的是“Amazon 证据能够识别什么构念”，不是为了让某个数值通过而改变同一构念的
cutoff。

## 3. 三层 acceptance

### A. Hard correctness

- C0/C1/C2 connectivity、split 和 leakage；
- exact customer count、quota/rounding、matrix validity 和 feasibility；
- provenance、每个 `city × track` 的 5 weekday : 2 weekend；
- 140 planned/materialized/verified；
- no timeout/rejection/unresolved/orphan；
- verifier 和 reconciliation PASS；
- read-only audit 前后 family 文件的 path/size/mtime 清单完全相同。

### B. Amazon operational transfer

冻结配置为
`configs/amazon_operational_transfer_acceptance_v2.json`：

- M1 family 和 corpus normalized W1 均不超过 source P99 time envelope 的 5%；
- radial-decile row/column margins exact；
- source route count = retained region count，且无 route rounding drop；
- region sizes 之和精确等于 parent customer count；
- primary family 必须使用 `SINGLE_STRUCTURE_DAY + SINGLE_ORDER_DAY`；
- 每个 order template 必须属于冻结 cohort/source day，family 内不复用；
- 每个 view 的 package count、demand volume、service time、TW start/end 必须与观察到的
  Amazon stop template 经存储 dtype 转换后逐项相等；
- matched-vs-eligible pool 的 demand/package/service/TW-width 摘要相对差不超过 10%；
- TW presence 差不超过 0.02；
- weekday/weekend source labels、5:2 计数和分组运营摘要完整。

这里的 5%、10% 和 0.02 在运行本次 v2 evaluator、查看任何外部 evaluation track 前写入
版本化 config。运行后不得为了 PASS 修改。

### C. Cross-city spatial diagnostics

- M2：customer 到最近 customer 的 directed road time；
- M3：region 内 directed pairwise road-time P50/P90；
- M5：community concentration 与 same-count baseline；
- city/day-type 差异；
- proposed activation、radial-only/uniform baseline 对比。

C 层报告必须完整，但其中任何数值都不能进入 v2 的 `checks` 或改变 `passed`。

## 4. 版本化输出

在同一现有 pilot root 下只新增：

```text
reports/stage2_repair/q90_gate_v1_original_fail.json
reports/stage2_repair/cross_city_spatial_diagnostic_v1.json
reports/stage2_repair/amazon_operational_transfer_acceptance_v2.json
reports/stage2_repair/pilot_acceptance_report_v3_operational.json
```

原 `q90_gate.json` 和 `pilot_acceptance_report.json` 保留。前者是历史 0/24 FAIL；后者是
以旧 Q90 hard gate 得出的 v2 pilot acceptance FAIL。新报告通过重新读取已有证据生成，
不手工改旧 JSON。

所有证据 inventory 使用 path/size/mtime，不执行 SHA256 或文件 hash 校验。

## 5. 执行顺序和 STOP

```text
commit and push frozen v2 code/config
→ read existing 140 families
→ preserve Q90 v1 FAIL verbatim
→ build cross_city_spatial_diagnostic_v1
→ evaluate amazon_operational_transfer_acceptance_v2
→ build pilot_acceptance_report_v3_operational
→ STOP for reviewer
```

任一 A/B gate 失败时如实输出 RED 并停止；不得改阈值、换 seed、重跑 family 或开始
Test2/Jacksonville/full corpus。即使 v3 pilot report PASS，full 7,500-family generation、
profile promotion 和 archive/restore 仍未获批准。
