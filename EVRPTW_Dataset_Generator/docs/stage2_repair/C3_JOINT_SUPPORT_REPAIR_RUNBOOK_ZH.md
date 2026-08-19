# Stage-2 C3 联合空间支持修复与执行纪律

> 状态：实现冻结候选；必须先形成 clean executable commit，再重建 plan/C2/C3
> 证据。本文不批准 full 7,500-family generation、archive 或 restore artifact。

## 1. 修复目标

v8 的三个 San Antonio 失败不是随机 seed 失败，而是 plan 在不知道
depot × Amazon structure source 联合容量是否可行时就固定了 family。旧 runner
随后在 materialization retry 中同时改变 depot、structure source、road-state 与
customer activation seed，因而把 planning 缺陷伪装成随机 rejection。

C3 位于 C2 之后、materialization 之前。它不改变：

- C0 family ID、city/track slot、5:2 weekday/weekend ledger；
- frozen customer split；
- customer pool、parent scale、radial-decile quota；
- region-first activation、exact global uniqueness max-flow；
- Amazon primary single-day rule；
- C1/R2-v2 connectivity certificate。

## 2. Candidate tuple 与顺序

每个 tuple 固定 city、track、day_type、parent scale、customer pool、depot、
Amazon structure source、depot seed、customer-superset seed 和 road-state seed。

候选顺序：

1. depot 第一项与旧 _select_depot_group 逐行一致；
2. 其余 physical facility groups 使用 family depot seed 的确定性 rank；
3. 每个 group 仍只取 canonical access point，Tier-A 优先规则不变；
4. Amazon single structure days 使用原 stable seeded rank；
5. pair 按 depot rank、再按 structure-source rank 枚举；
6. 选择第一个同时通过 C3-A 与 C3-B 的 pair；
7. 全部 pair 耗尽时是 planning hard fail，不进入 materialization。

## 3. C3-A：aggregate radial support

先使用正式 materialization 的同一条数据路径：

    frozen split/source pool
    → selected depot directed node/turn connectivity mask
    → selected source q99 time envelope
    → direct depot-customer-depot energy mask
    → source-specific radial deciles

随后直接调用 radial_decile_support_contract。该函数也由正式
activate_spatial_customers 调用，因此 quota construction 没有第二份近似实现。
对 decile b=0..9 必须满足 available_b >= required_b。

失败代码为 SPATIAL_QUOTA_UNSUPPORTED，报告同时保存 required_decile_counts、
available_decile_counts 和精确 deficit。

## 4. C3-B：exact assignment feasibility

C3-A 通过后，C3 调用正式 activate_spatial_customers。region seed、
road-community growth、competition expansion 与 exact maximum-flow primitive
均与 materialization 共用；不允许独立的 greedy/approximate feasibility
实现。C3-B 只有完成全局唯一 customer assignment 后才算通过。

## 5. Plan 必填字段

每个通过 C3 的 family 写入：

    joint_support_contract_id
    candidate_depot_count
    candidate_structure_source_count
    joint_pair_count
    aggregate_gate_pass_count
    exact_gate_pass_count
    selected_depot_id
    selected_structure_source_id
    required_decile_counts
    available_decile_counts
    capacity_contract_fingerprint
    rejected_pair_reason_counts

capacity_contract_fingerprint 是对小型 deterministic planning contract 的
BLAKE2 标识，不读取或校验数据文件，不是 SHA256/file-integrity 流程。根据当前
执行约束，C3、pilot、archive/restore 都不得新增 SHA256 文件扫描。

## 6. Materialization replay 与 retry

含 C3 contract 的 family 在 retry 时固定 depot_seed、
customer_superset_seed、road_state_seed、selected_depot_id、
selected_structure_source_id 和 capacity_contract_fingerprint。

charger/order/vehicle 等下游 stochastic namespace 可以按既有 lifetime attempt
ledger 变化，但不得重新搜索联合空间 pair。

若冻结 depot/source 不再合法、C3-approved pair 再次发生 spatial activation
failure，或 required/available counts 与 fingerprint 不一致，则抛出
C3_ACTIVATION_CONSISTENCY_BUG，retryable=false，并写入非空
capacity_contract_fingerprint；整批按既有 process-group abort contract 停止。

## 7. Run-report 状态机

顶层状态严格为：

    planned → materializing → verifying → passed | failed

C0/plan-only 必须是 status=planned、passed=null。所有状态写入使用同目录
temporary file、fsync、os.replace。verifier 或其他未捕获异常必须保留：

    planned_family_ids
    materialized_family_ids
    verified_family_ids
    unresolved_family_ids
    exception.type
    exception.message
    last_completed_stage

然后保持 non-zero exit。

## 8. 冻结执行顺序

    new clean executable commit
    → full deterministic tests
    → fresh root C0 reconstruction
    → C0 equality/140-family/2,590-view/5:2 assertions
    → reuse approved C1/R2-v2 evidence as permitted
    → rerun C2 bound to the new clean commit
    → C3 targeted three-family plan gate
    → confirm each original rank-0 pair is rejected by C3
    → confirm replacement pair selected for the same three slots
    → materialize and verify exactly those three replacement families
    → verifier-exception fixture
    → STOP if any targeted condition fails
    → fresh-root full 140-family C3 plan
    → 12-worker 140-family non-release pilot
    → verifier/metrics/process-group cleanup
    → STOP for reviewer

140-family 参数仍为 INSTANCE_MODE=non_release_pilot、WORKERS=12、
FAMILIES_PER_WORKER_TASK=1、PILOT_FAMILIES_PER_CITY=7、
MAX_ATTEMPTS_PER_FAMILY=4、FAMILY_WALL_TIMEOUT_S=7200、
TERMINATION_GRACE_S=60、STOP_POLICY=abort_all_inflight_after_grace。

## 9. C3 CLI

在 C2 PASS 后运行：

    PYTHONPATH=src python scripts/apply_stage2_joint_support_gate.py \
      --cle-root <CLE_ROOT> \
      --plan-root <PILOT_ROOT>/generation_plan \
      --customer-split-root <PILOT_ROOT>/customer_splits \
      --amazon-artifact-root <AMAZON_ARTIFACT_ROOT> \
      --amazon-cohort-split <AMAZON_COHORT_SPLIT_JSON> \
      --profile configs/us_reference_instance_profile_v2.json \
      --c2-report <PILOT_ROOT>/reports/stage2_repair/c2_release_preflight.json \
      --output <PILOT_ROOT>/reports/stage2_repair/c3_joint_support.json \
      --mode non_release_pilot

三个 replacement target 增加 --family-ids ID1 ID2 ID3 --targeted-gate。
--report-only 只形成 evidence，不改 plan；正式 targeted/full C3 不使用该选项。

## 10. 放行边界

三个 targeted replacement 必须 planned=3、materialized=3、verified=3、
timed_out=0、unresolved=0、materialization rejection=0、remaining process
groups=0。通过后才能启动 fresh 140-family pilot。

即使 pilot 140/140 PASS，也必须停止并交回报告；full 7,500-family、
archive/restore/release artifact 仍未获批准。
