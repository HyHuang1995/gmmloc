# GMMLoc

[![Build Status](https://travis-ci.org/HyHuang1995/gmmloc.svg?branch=master)](https://travis-ci.org/github/HyHuang1995/gmmloc)
[![LICENSE](https://img.shields.io/badge/license-GPL%20(%3E%3D%202)-informational)](https://github.com/HyHuang1995/gmmloc/blob/master/LICENSE)

Dense Map Based Visual Localization. [[project]](https://sites.google.com/view/gmmloc/)

## Paper and Video

Related publication:
```latex
@article{huang2020gmmloc,
  title={GMMLoc: Structure Consistent Visual Localization with Gaussian Mixture Models},
  author={Huang, Huaiyang and Ye, Haoyang and Sun, Yuxiang and Liu, Ming},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={4},
  pages={5043--5050},
  year={2020},
  publisher={IEEE}
}
```

Demo videos:

<a href="https://www.youtube.com/watch?v=Ul4-H33uwx4" target="_blank"><img src="https://www.ram-lab.com/image/gmmloc_v103.gif" alt="v103" height="240" border="10" style="margin-right:10em"/></a>
<a href="https://www.youtube.com/watch?v=Ul4-H33uwx4" target="_blank"><img src="https://www.ram-lab.com/image/hyhuang_iros2020_cover.png" 
alt="gmmloc" height="240" border="10" /></a>

## Prerequisites

We have tested this library in Ubuntu 18.04. Prerequisites for installation:

1. [ROS](http://wiki.ros.org/melodic/Installation) (melodic)

2. [OpenCV3](https://docs.opencv.org/3.4.11/d7/d9f/tutorial_linux_install.html)
```
apt-get install libopencv-dev
```
3. miscs:
```
apt-get install python-wstool python-catkin-tools 
```
4. [evo](https://github.com/MichaelGrupp/evo) (optional)
```
pip install evo --upgrade --no-binary evo
```

## Installation
Initialize a workspace:

```
mkdir -p /EXAMPLE/CATKIN/WORK_SPACE
cd /EXAMPLE/CATKIN/WORK_SPACE

mkdir src
catkin init
catkin config --extend /opt/ros/melodic
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin config --merge-devel
```

Clone the code:
```
cd src
git clone git@github.com:hyhuang1995/gmmloc.git
```

If using SSH keys for github, prepare the dependencies via:
```
wstool init . ./gmmloc/gmmloc_ssh.rosinstall
wstool update
```

or using https instead:
```
wstool init . ./gmmloc/gmmloc_https.rosinstall
wstool update
```

Compile with:
```
catkin build gmmloc_ros
```

## Running Examples
We provide examples on EuRoC Vicon Room sequences. For example, to run the demo on V1_03_difficult:

1. Download the [sequence](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) (ASL Format)

2. Replace the [**/PATH/TO/EUROC/DATASET/**](https://github.com/HyHuang1995/gmmloc/blob/770eadc99229eff17f2f613e969e4e9c10499496/gmmloc_ros/launch/v1.launch#L25) in [v1.launch](https://github.com/HyHuang1995/gmmloc/blob/master/gmmloc_ros/launch/v1.launch) with where the sequence is decompressed:
```
<param name="data_path" value="/PATH/TO/EUROC/DATASET/$(arg seq)/mav0/" />
```

3. Launch:
```
roslaunch v1.launch seq:=V1_03_difficult
```

## Evaluation
If evo is installed, we provide script for evaluating on Vicon Room sequences.
```
roscd gmmloc_ros
./scripts/evaluate_euroc.sh
```
and the results would be saved to **gmmloc_ros/expr**.
By default, we follow the evaluation protocol of [DSO](https://vision.in.tum.de/research/vslam/dso) to perform evaluation without multi-threading. If you would like to run the script in online mode, uncomment [this line](https://github.com/HyHuang1995/gmmloc/blob/770eadc99229eff17f2f613e969e4e9c10499496/gmmloc_ros/scripts/evaluate_euroc.sh#L60) in the script:
```
rosparam set /gmmloc/online True
```

## Credits

Our implementation is built on top of [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2), we thank Raul et al. for their great work.
