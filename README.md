# Pi_Pose_Prediction
Use pose estimation to detect joint points. Train a model for certain structure group of points as a single pose class for prediction and classification.


## Inference and using the prediction model
`cd ./inference_prediction`

### Instal dependencies
Run `sh setup.sh` to  install neccessary packages for the enviornment

### Running pose prediction
```
python pose_classification.py --classifier CLASSIFIER_NAME.tflite --label_file LABEL_FILE_PATH.txt --cameraId CAMERA_ID(Integer) 
```

The default value of `cameraId` is `0` if not specified. Use `find_camera.py` to find `cameraId` for any camera that appears under `/dev/video*`

### Prediction customization options
```
python pose_classification.py 
--model MODEL_NAME 
--tracker TRACKER_TYPE 
--classifier CLASSIFIER_NAME 
--label_file LABEL_FILE_PATH 
--cameraId CAMERA_ID 
--frameWidth FRAME_WIDTH 
--frameHeight FRAME_HEIGHT
```
*  Here is the full list of parameters supported by the sample:
```python3 pose_classification.py```
  *   `model`: Name of the TFLite pose estimation model to be used.
    *   One of these values: `posenet`, `movenet_lightning`, `movenet_thunder`, `movenet_multipose`
    *   Default value is `movenet_lightning`.
  *   `tracker`: Type of tracker to track poses across frames.
    *   One of these values: `bounding_box`, `keypoint`
    *   Only supported in multi-poses models.
    *   Default value is `bounding_box`.
  *   `classifier`: Name of the TFLite pose classification model to be used.
    *   Default value is empty.
    *   If no classification model specified, the sample will only run the pose
        estimation step.
  *   `camera_id`: Specify the camera for OpenCV to capture images from.
    *   Default value is `0`.
  *   `frameWidth`, `frameHeight`: Resolution of the image to be captured from
      the camera.
    *   Default value is `(640, 480)`.


## Training custom poses

Follow instructions in pose_classification.ipynb

Prepare an archive file (ZIP, TAR, or other) that includes a folder with your images dataset. The folder must include sorted images of your poses as follows.

If you've already split your dataset into train and test sets, then set `dataset_is_split` to **True**. That is, your images folder must include `train` and `test` directories like this:

```
yoga_poses/
|__ train/
    |__ downdog/
        |______ 00000128.jpg
        |______ ...
|__ test/
    |__ downdog/
        |______ 00000181.jpg
        |______ ...
```

Or, if your dataset is NOT split yet, then set
`dataset_is_split` to **False** and we'll split it up based
on a specified split fraction. That is, your uploaded images
folder should look like this:

```
yoga_poses/
|__ downdog/
    |______ 00000128.jpg
    |______ 00000181.jpg
    |______ ...
|__ goddess/
    |______ 00000243.jpg
    |______ 00000306.jpg
    |______ ...
```