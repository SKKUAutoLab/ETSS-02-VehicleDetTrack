import torch
import struct
import glob
import os
from utils.torch_utils import select_device

# Initialize
device = select_device('cpu')

for ckpt_pt in glob.glob("tss/detector/yolov5/weights/yolov5s6_trt/yolov5s6_aicity2021_*full.pt"):
    print(ckpt_pt)

    # Load model
    model = torch.load(ckpt_pt, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()

    dir_parrent = os.path.dirname(ckpt_pt)
    ckpt_wts    = os.path.join(dir_parrent, f"{os.path.basename(ckpt_pt).split('.')[0]}.wts")

    with open(ckpt_wts, 'w') as f:
        f.write('{}\n'.format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f',float(vv)).hex())
            f.write('\n')

    del model
