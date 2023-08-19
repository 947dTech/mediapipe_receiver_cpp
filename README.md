# mediapipe_receiver_cpp

- Author: Hiroaki Yaguchi, 947D Tech.
- License: Apache 2.0 License

This repository includes humanoid joint coordinates mapper
from mediapipe holistic tracking.

## Dependincies

- Eigen3

## How to build

This is include library, you can build it just including.
Please clone this repositry into `INCLUDE_PATH` and just include headers as:

```
#include "mediapipe_receiver_cpp/mediapipe_receiver.h"
```

## How to use

### Preparing the modefied version of mediapipe

To run the modified version of mediapipe,
an Android device with a certain level of performance
(the minimum system requirements are Google Pixel 5) is required.

The modified version of mediapipe can be obtained by either of the following methods

#### Using apk in HIROMEIRO

Please download and install from the following repository release.

https://github.com/947dTech/HIROMEIRO

#### Building from source

Please prepare an environment (Docker is recommended) in which you can build the Mediapipe Android app.

First, obtain the source code for the modified version of mediapipe from github.

https://github.com/947dTech/mediapipe

Target branch is `holistic_v0.9.2.1_release`.

We recommend using Docker to build the mediapipe environment.

https://google.github.io/mediapipe/getting_started/install.html#installing-using-docker

Please refer to the official documentation for building the Android build environment.

https://google.github.io/mediapipe/getting_started/android.html

Android app can be built with the following command.

```
$ bazelisk build -c opt --config=android_arm64 --linkopt="-s" mediapipe/examples/android/src/java/com/google/mediapipe/apps/holistictrackinggpu:holistictrackinggpu
```

Installation can be done in the following way after the actual device is recognized by adb.

```
$ adb install bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/holistictrackinggpu/holistictrackinggpu.apk
```

When activated, after initialization (which takes some time), the recognition results are displayed on the screen.
Data is sent via UDP to port number 947D (38013 in decimal) of the configured IP address.
Since the UDP receiver and JSON converter are platform-dependent, please implement it by yourself.
The specification of the data sent is basically a reference to the original specification, converted to json.

https://google.github.io/mediapipe/solutions/holistic.html

- The posture recognition result in meters is stored in `pose_world_landmarks`.
- The result of posture recognition is stored in `pose_landmarks`.
- The `face_landmarks` stores the facial expression recognition results.
- The `right_hand_landmarks` and `left_hand_landmarks` contain the recognition results of both hands, respectively.

A timestamp is added to each landmark.

Note that recognition results other than `pose_world_landmarks` must take into account the screen aspect ratio.

By default, the in-camera is used.
Note that it is therefore a mirror image.

We also send `gravity` at the same time to recognize the orientation of the phone.
Please use it to estimate the camera orientation.

- `gravity`: gravity vector
- `gravity_stamp`: timestamp

Additional camera parameters are sent.
This allows estimation of the 3D position of the person in the camera coordinate system by
comparing `pose_landmarks` and `pose_world_landmarks`.

- `camera_params`
    - `focal_length`: Focal length [px]
    - `frame_width`: Image width
    - `frame_height`: Image height
