# EVRPTW_Dataset

Generated datasets are stored here and grouped by dataset profile, customer count, and charging-station count.

Recommended layout:

```text
EVRPTW_Dataset/
  Amazon_Calibrated_v1/
    Cus_50/
      CS_4/
        regions/
        instances/
        metadata/
        analysis_outputs/
    Cus_1800/
      CS_12/
```

Generated `.pkl` instances are ignored by git by default. Release datasets separately or with Git LFS if needed.
