Links to referenced repos: https://www.github.com/ildoonet/tf-openpose, https://github.com/gsethi2409/tf-pose-estimation, https://github.com/abdelrahman-gaber/tf2-object-detection-api-tutorial

Based on Human Activity Knowledge Engine (https://arxiv.org/abs/2004.00945).

The repository uses TF2 Object Detection API for person detection and a Tensorflow implementation of OpenPose for detecting body keypoints and combines information from both to create body part bounding boxes. Individual body parts are obtained from anatomical knowledge.

To-do:
1. Create classification models to recognize actions of individual body parts.
2. Combine object detection with part-state actions
3. Pass through language understanding models to understand human activity
