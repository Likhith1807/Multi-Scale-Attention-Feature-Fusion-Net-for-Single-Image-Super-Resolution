# Paste datasets here
No image data is distributed with this project.

## Option A: ZIP archives (recommended)
Paste these completed files directly into
`C:\SISR_Computer-Vision\data\archives`:

| Exact filename | Official download |
|---|---|
| DIV2K_train_HR.zip | https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip |
| DIV2K_train_LR_bicubic_X4.zip | https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip |
| DIV2K_valid_HR.zip | https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip |
| DIV2K_valid_LR_bicubic_X4.zip | https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip |

Then run `03_PREPARE_DATA_OFFLINE.cmd` from the project folder.
Do not rename `.part` or `.crdownload` files as finished ZIPs.
Keep names exact; remove browser-added `(1)` only on a complete intended ZIP.

## Option B: already extracted PNGs
Paste the images into exactly these folders, without extra nested folders:

| Folder relative to project | Count | Example |
|---|---:|---|
| data/DIV2K_train_HR | 800 | 0001.png |
| data/DIV2K_train_LR_bicubic/X4 | 800 | 0001x4.png |
| data/DIV2K_valid_HR | 100 | 0801.png |
| data/DIV2K_valid_LR_bicubic/X4 | 100 | 0801x4.png |

Run `03_PREPARE_DATA_OFFLINE.cmd`; complete extracted folders are accepted.
Original training IDs are 0001–0800; validation IDs are 0801–0900.

## Optional online download
In an activated environment run `python download_div2k.py`.
It shows MiB/total and percentage if the server sends Content-Length.
Counters refer to the current archive and reset between archives.
Partial files resume on rerun only when the server supports byte ranges.

Official dataset and terms: https://data.vision.ee.ethz.ch/cvl/DIV2K/
