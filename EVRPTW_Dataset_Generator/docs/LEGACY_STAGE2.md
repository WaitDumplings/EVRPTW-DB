# Legacy Stage-2 compatibility

The following files implement the previous synthetic service-territory and
operating-day generator:

```text
evrptw_hierarchy/
configs/amazon_hierarchy.yaml
prepare_region_pool.py
prepare_ac_benchmark_suite.py
instance_generate.py
requirements-legacy-stage2.txt
```

They remain because current TERRAN utilities import `evrptw_hierarchy`
directly. Deleting them would break existing benchmark experiments unrelated to
the Stage-1 migration.

These modules are frozen compatibility code. They do not consume the new CLE
schema and are not evidence that the new real-road Stage-2 instance generator
has been completed. The future Stage-2 implementation should be introduced as
a separate package that reads only verified CLE artifacts, then retired through
an explicit migration after TERRAN no longer imports the legacy package.
