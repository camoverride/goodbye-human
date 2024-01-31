# Goodbye Human

Streams back a video feed with people deleted.


## Installation

- Download model: https://github.com/ayoolaolafenwa/PixelLib/releases/download/0.2.0/pointrend_resnet50.pkl
- `pip install requirements.txt`


## Run

- `python stream_video.py`


## Comments

The code this project employs was mostly writted in 2020 and 2021. Look for a newer model!

- https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/README.md
- https://towardsdatascience.com/real-time-image-segmentation-using-5-lines-of-code-7c480abdb835
- experiment with confidence: `ins.load_model("pointrend_resnet50.pkl", confidence = 0.3)`
- experiment with model speed: `fast`, `rapid`
- experiment with model: `resnet101` is slower and more accurate: `ins.load_model("pointrend_resnet101.pkl", network_backbone="resnet101")`
- Possible BETTER model: https://github.com/open-mmlab/mmdetection/tree/main/demo
- VERY fast model (paper): https://arxiv.org/abs/2303.07815
- another older model that is fast, and has code! https://github.com/irfanICMLL/ETC-Real-time-Per-frame-Semantic-video-segmentation
- pixel lib: https://pixellib.readthedocs.io/en/latest/video_instance.html + https://github.com/ayoolaolafenwa/PixelLib
- OpenCV background subtraction: https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
- Optical flow estimation? https://learnopencv.com/optical-flow-using-deep-learning-raft/
- Optical flow (torch): http://pytorch.org/vision/master/auto_examples/others/plot_optical_flow.html
- Optical flow (openCV): https://docs.opencv.org/4.x/db/d7f/tutorial_js_lucas_kanade.html +s https://learnopencv.com/optical-flow-in-opencv/
- semantic seg (papers with code): https://paperswithcode.com/task/real-time-semantic-segmentation
- grab cut? https://ujangriswanto08.medium.com/real-time-image-segmentation-using-contour-detection-and-grabcut-algorithm-cc5ebfa41caf
- YOLO? https://thepythoncode.com/article/real-time-object-tracking-with-yolov8-opencv
- does yolo have masks? https://docs.ultralytics.com/tasks/segment/ + https://stackoverflow.com/questions/76168470/how-to-create-a-binary-mask-from-a-yolo8-segmentation-result
- YOLO: https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993
-
