## Install
### Dependencies

You need dependencies below.
- python3
- tensorflow 1.4.1+
- opencv3, protobuf, python3-tk
- slidingwindow
  - https://github.com/adamrehn/slidingwindow
  - I copied from the above git repo to modify few things.
  
Personally, I used the next dependencies:

Dedicate use of GPU:
  - Cuda v.10
  - cuDNN v7.4
Other dependencies:
- python3
- tensorflow 1.4.1

Operative System:
- Windows 10

### Steps

Clone the repo and install 3rd-party libraries.

```bash
$ git clone https://github.com/paulnajera/skeleton_humanactivity.git
$ cd skeleton_humanactivity
$ pip3 install -r requirements.txt
```
If there is a problem installing the requiremets (related to MVS and pycocotools) I followed this steps and it worked:
https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/62381

Build c++ library for post processing. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess
(wig have to be installed in order to run this command)
```
$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

## Models & Performances

See [experiments.md](./etc/experiments.md)

### Download Tensorflow Graph File(pb file)

Before running, you should download graph files. You can deploy this graph on your mobile or other platforms.

- cmu (trained in 656x368)
- mobilenet_thin (trained in 432x368)
- mobilenet_v2_large (trained in 432x368)
- mobilenet_v2_small (trained in 432x368)

CMU's model graphs are too large for git, so I uploaded them on an external cloud. You should download them if you want to use cmu's original model. Download scripts are provided in the model folder.

```
$ cd models/graph/cmu
$ bash download.sh
```

## Run programs>

### Test Inference

In order to test the next steps, the model have to be downloaded.

You can test the inference feature with a single image.

```
$ python run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg
```

The image flag MUST be relative to the src folder with no "~", i.e:
```
--image ../../Desktop
```

Then you will see the screen as below with pafmap, heatmap, result and etc.

![inferent_result](./etcs/inference_result2.png)

### Realtime Webcam

```
$ python run_webcam.py
```
Including other options:
```
$ python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0
```

Apply TensoRT 

```
$ python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0 --tensorrt=True
```

### From video

```
$ python run_video.py --video=videos/test3.mp4
```

In both cases, the result is recorded and save in the main folder where the .py are)

### Realtime Webcam skeleton 3D

```
$ python webcam_3D.pz
```

### Video skeleton 3D

```
$ python video_3D.py
```

Video source shoul be specified in line 50:

```
camera = './skeleton_humanactivity/videos/test1-2.mp4'
```

## Python Usage

This pose estimator provides simple python classes that you can use in your applications.

See [run.py](run.py) or [run_webcam.py](run_webcam.py) as references.

```python
e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
humans = e.inference(image)
image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
```

## ROS Support

See : [etcs/ros.md](./etcs/ros.md)

## Training

See : [etcs/training.md](./etcs/training.md)

## References

See : [etcs/reference.md](./etcs/reference.md)
