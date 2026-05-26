# EVRPTW_Dataset

Generated datasets are stored here. The dataset family is **EVRPTW-D**; the current release profile is **AC-v1** (Amazon-Calibrated v1).

Recommended AC-v1 layout:

```text
EVRPTW_Dataset/
  AC_v1/
    train/
      service_territory_pool.pkl
      manifest.json
    eval/
      service_territory_pool.pkl
      manifest.json
      AC_Tiny_5/
        instances.pkl
        metadata/
        analysis_outputs/
      AC_Small_15/
        instances.pkl
      AC_Medium_50/
        instances.pkl
      AC_Large_100/
        instances.pkl
      AC_XLarge_1000/
        instances.pkl
    generation_timing.csv
    dataset_manifest.json
```

`train/service_territory_pool.pkl` stores the reusable training service-territory pool as one pickle stream. Training instances are sampled online from this pool and are not stored by default. `eval/service_territory_pool.pkl` stores held-out service territories for fixed evaluation suites. Each `eval/AC_*` directory stores its fixed operating-day instances as one `instances.pkl` pickle stream. Generated `.pkl` files are ignored by git by default; release datasets separately or with Git LFS if needed.
