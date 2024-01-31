# Goodbye Human

Streams back a video feed with people deleted.


## Installation

`pip install requirements.txt`


## Run

`python stream_video.py`


## Comments

The code this project employs was mostly writted in 2020 and 2021. Look for a newer model!
- model: https://github.com/ayoolaolafenwa/PixelLib/releases/download/0.2.0/pointrend_resnet50.pkl
- https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/README.md
- https://towardsdatascience.com/real-time-image-segmentation-using-5-lines-of-code-7c480abdb835
- experiment with confidence: `ins.load_model("pointrend_resnet50.pkl", confidence = 0.3)`
- experiment with model speed: `fast`, `rapid`
- experiment with model: `resnet101` is slower and more accurate: `ins.load_model("pointrend_resnet101.pkl", network_backbone="resnet101")`
- Possible BETTER model: https://github.com/open-mmlab/mmdetection/tree/main/demo
