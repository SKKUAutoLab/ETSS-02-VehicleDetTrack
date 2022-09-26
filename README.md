### Automation Lab, Sungkyunkwan University

#### GitHub Stats
![](https://img.shields.io/github/downloads/SKKU-AutoLab-VSW/RnT-TFE/total.svg?style=for-the-badge)



## A Region-and-Trajectory Traffic Flow Estimation 

---

#### I. Installation

1. Download & install Miniconda or Anaconda from https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html


2. NVidia CUDA environment requirement:
    * CUDA 11.3
    * cuDNN 8.3.2


3. Open new Terminal, create new conda environment named skku and activate it with following commands:
```shell
conda create --name skku_automationlab python=3.8

conda activate skku_automationlab
```

4. Go to "**automationlab_object_detection**" folder and execute following commands
```shell
pip install -r requirements.txt
```
---

5. (Optional) Docker will be updated soon

#### II. Data preparation

##### a. Data download

Go to the website of AI-City Challenge to get the dataset.

- https://www.aicitychallenge.org/2021-data-and-evaluation/

##### b. Video data import

Add video files to **/data/aicity2021_final/video**.
   
The program folder structure should be as following:

```
Region-and-Trajectory-TFE
├── data
│   ├── aicity2021_final
│   │   ├── video
│   │   │   ├── cam_1.mp4
│   │   │   └── cam_7.mp4
│   │   │   ...
│   ├── carla
│   │   ├── video
│   │   │   ├── Town10HD_location_1.mp4
│   │   │   ...
...
```

---

#### III. Reference

##### a. Weight 

Download weight from [Release](https://github.com/SKKU-AutoLab-VSW/Region-and-Trajectory-TFE/releases/tag/v1.0.0-alpha) then put it into:
```
tss/detector/yolov5/weights/yolov5s6
```

##### b. Change running file

Change the name of yaml file which belong to each video in "main.py"

```python
parser.add_argument(
	"--config",
	default="Town10HD_location_1.yaml",
	help="The config file for each camera. The final path to the config file is: TSS/data/[dataset]/configs/[config]/"
)
```

And the running script

```shell
bash bin/run_inference.sh
```

##### c. Get the result

```
Region-and-Trajectory-TFE
├── data
│   ├── aicity2021_final
│   │   ├── result
│   │   │   ├── cam_1.mp4
│   │   │   └── cam_7.mp4
...
```
