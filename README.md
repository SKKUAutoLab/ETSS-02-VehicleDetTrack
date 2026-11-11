# Vehicle Detection and Tracking

For traffic Flow analysis, we provide a vehicle detection and tracking system.



#### GitHub Stats
![](https://img.shields.io/github/downloads/SKKU-AutoLab-VSW/ETSS-02-VehicleDetTrack/total.svg)

## Getting Started

Linux, an NVIDIA GPU (CUDA 11.8) and conda:

```shell
conda create --name etss-vehicledettrack python=3.10 -y
conda activate etss-vehicledettrack
cd src/tfe
bash setup.sh
```

Put the videos in `src/tfe/data/Korea_cctv/video/` and the detector weights in
`src/tfe/model_zoo/` (each camera's config names its weights file).


## Runs

From `src/tfe`:

```shell
bash run_web_vehicle_detection_tracking.sh                                      # browser GUI, port 8000, open to the LAN
python main_web.py                                                              # browser GUI, this machine only
python main.py --dataset Korea_cctv --config 40_SUWON.yaml --write_video True   # one video, no GUI
```

Counts go to `src/tfe/data/Korea_cctv/temps/<camera>.txt` (and `.mp4`).

- Setup, data, options and results: [src/tfe/README.md](src/tfe/README.md)
- How the pipeline works: [docs/pipline_framework.readme.md](docs/pipline_framework.readme.md)


## Acknowledgement

This project is supported by the Institute for Information & Communications 
Technology Promotion (IITP) grant funded by the Korea government (MSIP) 
(2021-0-01364).
