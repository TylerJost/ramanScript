# ***Raman*** Tran***script***omics

A set of analysis/structuring files to read in and process Raman spectroscopy data. 

Raw data should be structured like so:
```
./data
└── experiment
    ├── images
    └── raw
        ├── bio condition 1
        │   ├── raman condition 1
        │   └── raman condition 2
        └── bio condition 2
            │
    ├── annotations.json
    └── experiment.npy
└── ramanAxis.txt
```
Each folder in `data` should be separated by experiment, then further sectioned by biological conditions and raman spectroscopy parameters (this is subject to change). 

This can be run by calling the main script, `ramanScript.py` with the appropriate parameters. All spectra are stored into an `experiment.npy` for quicker loading. Spectra with the appropriate number of points (i.e. the same number of points as contained in `ramanAxis.txt`) and that have a length that has an integer square root are then generated into images. These images can be manually annotated in COCO format and put into `annotations.json`. 
