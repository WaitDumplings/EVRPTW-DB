# EVRPTW-DB C1b / R2-v2 连通性审核与 pilot_v6 执行手册

> 状态：待 reviewer 审核。本文取代
> `STAGE2_CONNECTIVITY_REPAIR_AND_PIPELINE_REVIEW_ZH.md` 中从 Phase C0 开始的旧
> C1-v2 / R2-v1 放行说明。CLE_v2 不重建，C0 split 不改写，C2、LA smoke、pilot 和
> full run 均未获准执行。

## 1. 最终冻结结论

执行阶段固定为：

```text
Stage-1 source/geometry/road-anchor eligibility
→ Stage-1 directional projection/SCC quarantine
→ C0 complete-community train/heldout split
→ family depot selection
→ Stage-2 node/canonical-turn preflight
→ family/depot connectivity mask
→ fixed post-mask capacity check
→ territory
→ spatial activation
→ customer/charger sampling
→ selected-terminal final closure
→ materialization
```

两类 customer 的数据语义不同：

| 类型 | C0 pool | assignment status | Stage-1 eligible | family eligible |
|---|---|---|---:|---:|
| Stage-1 directional/SCC quarantine | `null` | `excluded_pre_split_connectivity` | false | false |
| Stage-2-only node/turn quarantine | 保留冻结的 `train` 或 `heldout` | `assigned_before_stage2_turn_preflight` | true | false |

Stage-2-only customer 不得从 split registry 删除、重新分 pool 或跨 pool。它只在对应
depot/family 的 mask 中失效。mask 后固定 pool 少于 N 时，抛出 non-retryable
fixed-roster failure；不能换 seed 绕过。

family metadata 使用：

```text
cle_evrptw_terminal_connectivity_quarantine_v2
```

并显式记录 mask 在 territory capacity、activation、sampling、materialization 前执行。

## 2. v4 只作为历史证据

批准的 frozen C0 基线仍是：

```text
EVRPTW_Dataset/Instances_v2/us_10city_trainval_pilot_v4
```

v4 C1 的实际观测必须永久保留，不能通过修改阈值重写为 passed：

| city | customer union count | customer rate | charger union count | charger rate |
|---|---:|---:|---:|---:|
| chicago | 259 | 0.000671409 | 0 | 0 |
| dallas | 854 | 0.004155131 | 1 | 0.004524887 |
| fort-worth | 583 | 0.002857157 | 1 | 0.013513514 |
| houston | 1,280 | 0.002989746 | 3 | 0.008620690 |
| los-angeles | 33,030 | 0.055176354 | 71 | 0.034399225 |
| new-york | 612 | 0.001632174 | 0 | 0 |
| philadelphia | 123 | 0.001097411 | 0 | 0 |
| phoenix | 5,601 | 0.014285605 | 7 | 0.020588235 |
| san-antonio | 2,889 | 0.007907746 | 0 | 0 |
| san-diego | 565 | 0.002186253 | 10 | 0.012062726 |

customer 总计：

```text
Stage-1 directional = 45,486
Stage-2 node        = 0
Stage-2 turn        = 310
union               = 45,796
```

charger 总计：

```text
Stage-1 directional = 86
Stage-2 node        = 0
Stage-2 turn        = 7
union               = 93
```

因此 reviewer 概述中的 “45,796 个 Stage-1” 应读为 customer union；真正的 Stage-1
customer count 是 45,486。R2-v1 的固定 0.1% customer / 1% charger 规则输出：

```text
outcome = triggered_stop_and_review
active_acceptance_rule = false
superseded_by = r2_v2_replayable_connectivity_certificate_gate_v1
```

raw rate 仍为 mandatory report-only provenance。

## 3. 新 schema 与产物

C1：

```text
cle_evrptw_phase_c1_terminal_connectivity_audit_v3
rule_id = layered_stage1_pre_split_stage2_family_mask_v1
```

C1 输出：

- `connectivity_audit.json`：城市级 rate、split/ledger contract、PF-1、post-mask capacity；
- `connectivity_audit.ledger.parquet`：每个 city/kind/source 的 union ledger；
- `connectivity_audit.family_depot_ledger.parquet`：逐 depot Stage-2 ledger，保留
  family IDs、C0 pool 和四个 node/turn masks。

R2-v2：

```text
cle_evrptw_connectivity_audit_acceptance_v2
rule_id = r2_v2_replayable_connectivity_certificate_gate_v1
```

输出：

- `connectivity_audit_acceptance_v2.json`；
- `connectivity_audit_acceptance_v2.certificates.parquet`；
- `connectivity_audit_acceptance_v2.concentration.json`；
- `connectivity_audit_acceptance_v2.h64_samples.parquet`；
- `connectivity_audit_acceptance_v2.h64_review_template.json`；
- `connectivity_h64_maps/<city>.html` 和 `<city>.geojson`。

人工 sign-off 绑定 candidate commit，并逐项列出实际 reviewed sample IDs。后续放行不再计算或校验 sample、C1、certificate 等文件 SHA；另一个 commit 的 review 文件仍不能通过。

## 4. 版本纪律

所有 C0/C1/R2-v2 产物必须绑定同一个 clean candidate commit：

```bash
cd /data/Maojie/ICLR/EVRPTW-DB
git status --short
git branch --show-current
git rev-parse HEAD
```

要求：

```text
branch = stage2-repair-candidate
working tree = clean
```

先运行全量测试，再 commit/push，然后才建立 pilot_v6。任何代码变化都要求新 commit 和
新 root；不得用旧 C1 报告配新代码。pilot_v5 因 9 个 CBG/community 漂移（其中 3 个 pool 漂移）已 STOP；pilot_v6 直接复用批准的 v4 frozen split，不再重算 CBG。

## 5. Phase C0：建立 pilot_v6 plan，不 materialize

```bash
cd /data/Maojie/ICLR/EVRPTW-DB

PILOT_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_10city_trainval_pilot_v6

INSTANCE_MODE=non_release_pilot \
FROZEN_SPLIT_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_10city_trainval_pilot_v4 \
WORKERS=1 \
FAMILIES_PER_WORKER_TASK=1 \
PILOT_FAMILIES_PER_CITY=7 \
INSTANCE_OUTPUT_ROOT="$PILOT_ROOT" \
./generate_instances.sh \
  --stages preflight splits plan \
  --cities new-york los-angeles chicago houston phoenix philadelphia \
           san-antonio san-diego dallas fort-worth \
  --tracks train validation
```

这一步只创建 split 和 plan。不得加 `materialize` 或 `verify`。

### C0 精确比较

```bash
cd /data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset_Generator

PYTHONPATH=src /home/npg/miniconda3/envs/maojie/bin/python \
  scripts/compare_stage2_c0_plans.py \
  --baseline-root ../EVRPTW_Dataset/Instances_v2/us_10city_trainval_pilot_v4 \
  --candidate-root ../EVRPTW_Dataset/Instances_v2/us_10city_trainval_pilot_v6 \
  --output ../EVRPTW_Dataset/Instances_v2/us_10city_trainval_pilot_v6/reports/stage2_repair/c0_exact_comparison.json
```

必须同时满足：

```text
10-city split rows/fields exact
140 family rows/fields exact
2,590 view rows/fields exact
20 city×track cells; each 5 weekday + 2 weekend
```

比较器逐行检查 Parquet；只不比较顶层 `split_registry.json` 的新 commit provenance。

## 6. Phase C1：分层 ledger、PF-1、post-mask capacity

```bash
cd /data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset_Generator

PILOT_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_10city_trainval_pilot_v6

set +e
PYTHONPATH=src /home/npg/miniconda3/envs/maojie/bin/python \
  scripts/audit_stage2_terminal_connectivity.py \
  --cle-root ../EVRPTW_Dataset/CLE_v2/us_11city \
  --profile configs/us_reference_instance_profile_v2.json \
  --plan-root "$PILOT_ROOT/generation_plan" \
  --split-root "$PILOT_ROOT/customer_splits" \
  --block-group-preset configs/us_census_block_groups_v1.json \
  --block-group-source-dir data/sources/census_block_groups_2025 \
  --output "$PILOT_ROOT/reports/stage2_repair/connectivity_audit.json"
C1_RC=$?
set -e
test "$C1_RC" -eq 2
```

exit code 2 是预期的 R2-v1 historical stop，不代表结构契约失败。随后必须检查：

```bash
jq '{
  schema,
  passed,
  structural_contract_passed,
  r2_v1,
  r2_v2,
  cities: [.cities[] | {
    city_slug,
    split: .customer_split_contract.passed,
    ledger: .quarantine_ledger_contract.passed,
    capacity: .stage2_post_mask_capacity.passed,
    pf1: .pf1.passed
  }]
}' "$PILOT_ROOT/reports/stage2_repair/connectivity_audit.json"
```

期望：

```text
passed = false
structural_contract_passed = true
r2_v1.outcome = triggered_stop_and_review
r2_v1.active_acceptance_rule = false
r2_v2.status = requires_connectivity_audit_acceptance_v2
all city split/ledger/capacity/PF-1 = true
```

不得把顶层 `passed` 改成 true，也不得把 0.1%/1% 调高。

## 7. Phase C1b：R2-v2 自动证书与 H64 产物

首次不提供 `--manual-review`：

```bash
set +e
PYTHONPATH=src /home/npg/miniconda3/envs/maojie/bin/python \
  scripts/build_connectivity_audit_acceptance_v2.py \
  --cle-root ../EVRPTW_Dataset/CLE_v2/us_11city \
  --profile configs/us_reference_instance_profile_v2.json \
  --connectivity-audit "$PILOT_ROOT/reports/stage2_repair/connectivity_audit.json" \
  --cohort-split configs/amazon_cohort_split_v1.json \
  --materialized-root "$PILOT_ROOT" \
  --output "$PILOT_ROOT/reports/stage2_repair/connectivity_audit_acceptance_v2.json"
R2V2_RC=$?
set -e
test "$R2V2_RC" -eq 2
```

未签字时 exit 2 是正确行为。自动证书逐条完成：

1. 从 portable GraphML 独立重建 eligible physical-edge catalog；
2. 对全部 stored `directed_projection_offsets` 检查 `u/v/key`；
3. stored ref set 必须与 catalog ref set 完全相同；
4. Stage-1 inbound/outbound 重新计算两次并与 raw/ledger 对齐；
5. Stage-2 按 city/depot/kind 重新运行 node/canonical-turn preflight 两次；
6. turn-only customer 必须 node outbound/return 都 true，且至少一个 turn mask false；
7. reason 必须属于冻结 reason set，不能 unknown/missing；
8. certificate、ledger、summary count 必须完全一致。

真实数据探针已经验证：

```text
Fort Worth Stage-1 certificates: 582/582 passed
Fort Worth Stage-2 depot-terminal certificates: 28/28 passed
```

这些只是实现探针，不代替 pilot_v6 的 10 城报告。

## 8. 集中度报告

每个 city × terminal kind 至少有 `__all_union__` 总行和逐 reason 行。总行记录：

- audit input / Stage-1 / Stage-2 / union count 和 rate；
- unique physical edge / OSM way / CBG / community / SCC；
- 每条 physical edge 的 terminal count；
- top 1/5/10 edge share；
- projection fraction q0/q25/q50/q75/q100；
- highway/road type；
- directed ref count；
- quarantined 与 eligible 的对应分布。

若少量 edge 占据大量 quarantined buildings，不能仅凭证书自动解释为真实道路问题；必须
由地图抽检确认 projection semantics。

## 9. H64 人工抽检

冻结 namespace：

```text
EVRPTW-DB:C1b:R2-v2:H64:v1
```

抽样规则：

- 每城、每个非空 Stage-1 customer reason：至少 5 个 unique terminals；
- Stage-1∪Stage-2 quarantined chargers 总数不超过 500 时全部抽取；否则每城/reason 至少 10；
- 每个存在 Stage-2 turn-only customer 的城市：抽取 `min(5, available)`；不足 5 个时全部检查，不复制；
- 每城 Stage-2 top-5 physical edges 各增加一个 H64 representative；
- 不允许手选、复制 terminal 或用同一 terminal 冒充多个 unique 样本。

HTML/GeoJSON 显示 terminal、projection connector、physical edge、全部 directed refs、
`u→v/key` 方向箭头和失败方向。

v4 已知 Fort Worth 只有 2 个 unique Stage-2 turn-only customers。按更新后的可用量规则，这 2 个全部检查，coverage row 记录 `requested=5, required=2, selected=2, all_available_selected=true`；不得复制样本凑 5 个。

若自动 gate 全部通过，reviewer 复制
`connectivity_audit_acceptance_v2.h64_review_template.json` 为签字文件，逐图审核后：

```json
{
  "ignored_valid_access_option_count": 0,
  "incorrect_road_or_projection_semantics_count": 0,
  "certificate_replay_disagreement_count": 0
}
```

并填写非空 `reviewer_signoff_id`。code commit 不得修改；不再要求 sample SHA。然后用相同命令
增加：

```bash
--manual-review /absolute/path/to/signed_h64_review.json
```

任何一项非零或 commit 不匹配均失败。

## 10. 生成能力与 materialization exclusion

R2-v2 自动 gate 同时要求：

- 10 城 PF-1 全部通过；
- 140 个 planned family 的 frozen pool 在 depot mask 后均不少于 N；
- primary 100/500/1000 × weekday/weekend PF-2 support 全部非零；
- 当前 root 中所有已有 `terminal_index.parquet` 对 Stage-2 turn-only IDs 出现次数为 0；
- materialization 后仍需再次执行同一 exclusion check；
- selected-terminal final closure 仍为最终硬门。

pre-C2 root 没有 materialized views 时，appearance count 为 0，但报告明确记录
`runtime_recheck_required_after_each_materialization=true`，不能把这个空集合检查当成
最终 pilot 证明。

## 11. C2 放行条件

C2 CLI 新增强制参数：

```text
--connectivity-acceptance .../connectivity_audit_acceptance_v2.json
```

只有以下条件全部成立才允许 C2：

```text
acceptance.schema = cle_evrptw_connectivity_audit_acceptance_v2
acceptance.rule_id = r2_v2_replayable_connectivity_certificate_gate_v1
acceptance.passed = true
acceptance.c2_allowed = true
acceptance commit = current clean candidate commit
R2-v1 outcome = triggered_stop_and_review
```

命令：

```bash
PYTHONPATH=src /home/npg/miniconda3/envs/maojie/bin/python \
  scripts/run_stage2_release_preflight.py \
  --amazon-artifact-root ../EVRPTW_Dataset/Calibration_v2/amazon_stage2_v3 \
  --cohort-split configs/amazon_cohort_split_v1.json \
  --connectivity-audit "$PILOT_ROOT/reports/stage2_repair/connectivity_audit.json" \
  --connectivity-acceptance "$PILOT_ROOT/reports/stage2_repair/connectivity_audit_acceptance_v2.json" \
  --plan-root "$PILOT_ROOT/generation_plan" \
  --output "$PILOT_ROOT/reports/stage2_repair/release_preflight.json"
```

当前不执行该命令。R2-v2 未通过或人工 review 未签字时，C2 会在计算 H3/PF2 结果作为
放行结果前拒绝。

## 12. STOP 条件

任一条件发生立即停止，不进入 C2：

- C0 exact comparison 任一项不等；
- Stage-1 null-pool 或 Stage-2 retained-pool contract 失败；
- 任一 certificate 无法重放；
- stored refs 与 graph/catalog 不完整一致；
- reason unknown/missing；
- 两次 replay 不一致；
- ledger/mask/summary count 不一致；
- PF-1、post-mask capacity 或 primary PF-2 失败；
- 存在可用 H64 样本但未全部选择；
- major edge 未覆盖；
- 人工三类 finding 任一非零；
- Stage-2 quarantined customer 出现在 materialized view；
- commit 不匹配。

STOP 只输出失败证书、coverage row 和地图例子；不得重试 seed、提高 R2-v1 阈值、
重分 C0 pool、启动 LA smoke 或 materialize pilot。
