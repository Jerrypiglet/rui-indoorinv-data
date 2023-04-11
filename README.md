# FIPT-data: data preparation for FIPT

### [Project Page](https://jerrypiglet.github.io/fipt-ucsd/) | [Paper]() | [Data]()

## Overview
This repo (branch: `fipt`) contains the code for generating customized data for [FIPT](), from scratch. Currently supported datasets include:

1. indoor_synthetic
   - See [**README_indoor_synthetic.md**](README_indoor_synthetic.md) for details.
   - Based on [Mitsuba XML files by Benedikt Bitterli](https://benedikt-bitterli.me/resources/)
   - Scripts support: sampling poses, rendering numerous modalities, visualization and export to FIPT/Monosdf/FVP/Li22.
2. real
   - See [**README_real.md**](README_real.md) for details.
   - Captured for FIPT
   - Scripts support: visualization and export to FIPT/Monosdf/FVP/Li22.

The repo is also useful for generating data for generic indoor inverse rendering pipelines. Exporting to the data format of other methods are possible, by adding customized format to `lib/class_exporter.py`.

## Related Works
- [FIPT]()
- [Monosdf]()
- [FVP]()
- [Li22]()

## TODO

## Citation

If you find our work is useful, please consider cite:

```
```
