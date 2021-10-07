### Automation Lab, Sungkyunkwan University

#### A Region-and-Trajectory Traffic Flow Estimation 

---

#### I. Data preparation

##### a. Data download

Go to the website of AI-City Challenge to get the dataset.

* https://www.aicitychallenge.org/

##### b. Video data import

Add video files to **/data/aicity2021_final/video**.
   
The program folder structure should be as following:

```
Region-and-Trajectory-TFE
├── data
│   ├── aicity2021_final
│   │   ├── video
│   │   │   ├── cam_1.mp4
│   │   │   ...
│   │   │   └── cam_7.mp4
...
```

---

#### II. Data preparation

##### a. Change running file

Change the name of yaml file which belong to each video in "main.py"

```python
parser.add_argument(
	"--config",
	default="cam_1.yaml",
	help="The config file for each camera. The final path to the config file is: TSS/data/[dataset]/configs/[config]/"
)
```

And the running script

```shell
bash bin/run_inference.sh
```

##### b. Get the result

```
Region-and-Trajectory-TFE
├── data
│   ├── aicity2021_final
│   │   ├── result
│   │   │   └── cam_7.mp4
...
```
