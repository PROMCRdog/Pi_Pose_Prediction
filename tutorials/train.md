# %% [markdown]
## 准备工作

## 准备工作

在这一部分，你将导入必要的库并定义几个函数来预处理训练图像，将其转换为包含关键点坐标和真实标签的CSV文件。

此处不会出现任何可观察到的结果，但你可以展开隐藏的代码单元查看我们稍后将调用的一些函数的实现细节。

**如果你只想创建CSV文件而不关心所有细节，只需运行本部分并继续进行第1部分。**

```python
!pip install -q opencv-python

import csv
import cv2
import itertools
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import tqdm

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### 用MoveNet执行姿势估计的代码

```python
pose_sample_rpi_path = os.path.join(os.getcwd(), 'examples/lite/examples/pose_estimation/raspberry_pi')
sys.path.append(pose_sample_rpi_path)

# 加载MoveNet Thunder模型
import utils
from data import BodyPart
from ml import Movenet
movenet = Movenet('movenet_thunder')

# 定义函数以使用MoveNet Thunder进行姿势估计
def detect(input_tensor, inference_count=3):
  image_height, image_width, channel = input_tensor.shape
  movenet.detect(input_tensor.numpy(), reset_crop_region=True)
  for _ in range(inference_count - 1):
    person = movenet.detect(input_tensor.numpy(), reset_crop_region=False)
  return person
```

### 可视化姿势估计结果的函数

```python
def draw_prediction_on_image(image, person, crop_region=None, close_figure=True, keep_input_size=False):
  image_np = utils.visualize(image, [person])
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  im = ax.imshow(image_np)
  if close_figure:
    plt.close(fig)
  if not keep_input_size:
    image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))
  return image_np
```

### 加载图像，检测姿势关键点并保存到CSV文件中的代码

```python
class MoveNetPreprocessor(object):
  def __init__(self, images_in_folder, images_out_folder, csvs_out_path):
    self._images_in_folder = images_in_folder
    self._images_out_folder = images_out_folder
    self._csvs_out_path = csvs_out_path
    self._messages = []
    self._csvs_out_folder_per_class = tempfile.mkdtemp()
    self._pose_class_names = sorted([n for n in os.listdir(self._images_in_folder) if not n.startswith('.')])

  def process(self, per_pose_class_limit=None, detection_threshold=0.1):
    for pose_class_name in self._pose_class_names:
      print('Preprocessing', pose_class_name, file=sys.stderr)
      images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
      images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
      csv_out_path = os.path.join(self._csvs_out_folder_per_class, pose_class_name + '.csv')
      if not os.path.exists(images_out_folder):
        os.makedirs(images_out_folder)
      with open(csv_out_path, 'w') as csv_out_file:
        csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        image_names = sorted([n for n in os.listdir(images_in_folder) if not n.startswith('.')])
        if per_pose_class_limit is not None:
          image_names = image_names[:per_pose_class_limit]
        valid_image_count = 0
        for image_name in tqdm.tqdm(image_names):
          image_path = os.path.join(images_in_folder, image_name)
          try:
            image = tf.io.read_file(image_path)
            image = tf.io.decode_jpeg(image)
          except:
            self._messages.append('Skipped ' + image_path + '. Invalid image.')
            continue
          else:
            image = tf.io.read_file(image_path)
            image = tf.io.decode_jpeg(image)
            image_height, image_width, channel = image.shape
          if channel != 3:
            self._messages.append('Skipped ' + image_path + '. Image isn\'t in RGB format.')
            continue
          person = detect(image)
          min_landmark_score = min([keypoint.score for keypoint in person.keypoints])
          should_keep_image = min_landmark_score >= detection_threshold
          if not should_keep_image:
            self._messages.append('Skipped ' + image_path + '. No pose was confidently detected.')
            continue
          valid_image_count += 1
          output_overlay = draw_prediction_on_image(image.numpy().astype(np.uint8), person, close_figure=True, keep_input_size=True)
          output_frame = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)
          cv2.imwrite(os.path.join(images_out_folder, image_name), output_frame)
          pose_landmarks = np.array([[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score] for keypoint in person.keypoints], dtype=np.float32)
          coordinates = pose_landmarks.flatten().astype(str).tolist()
          csv_out_writer.writerow([image_name] + coordinates)
        if not valid_image_count:
          raise RuntimeError('No valid images found for the "{}" class.'.format(pose_class_name))
    print('\n'.join(self._messages))
    all_landmarks_df = self._all_landmarks_as_dataframe()
    all_landmarks_df.to_csv(self._csvs_out_path, index=False)

  def class_names(self):
    return self._pose_class_names

  def _all_landmarks_as_dataframe(self):
    total_df = None
    for class_index, class_name in enumerate(self._pose_class_names):
      csv_out_path = os.path.join(self._csvs_out_folder_per_class, class_name + '.csv')
      per_class_df = pd.read_csv(csv_out_path, header=None)
      per_class_df['class_no'] = [class_index]*len(per_class_df)
      per_class_df['class_name'] = [class_name]*len(per_class_df)
      per_class_df[per_class_df.columns[0]] = (os.path.join(class_name, '') + per_class_df[per_class_df.columns[0]].astype(str))
      if total_df is None:
        total_df = per_class_df
      else:
        total_df = pd.concat([total_df, per_class_df], axis=0)
    list_name = [[bodypart.name + '_x', bodypart.name + '_y', bodypart.name + '_score'] for bodypart in BodyPart]
    header_name = []
    for columns_name in list_name:
      header_name += columns_name
    header_name = ['file_name'] + header_name
    header_map = {total_df.columns[i]: header_name[i] for i in range(len(header_name))}
    total_df.rename(header_map, axis=1, inplace=True)
    return total_df
```

### (可选) 运行MoveNet姿势估计逻辑的代码片段

```python
test_image_url = "https://cdn.pixabay.com/photo/2017/03/03/17/30/yoga-2114512_960_720.jpg" #@param {type:"string"}
!wget -O /tmp/image.jpeg {test_image_url}

if len(test_image_url):
  image = tf.io.read_file('/tmp/image.jpeg')
  image = tf.io.decode_jpeg(image)
  person = detect(image)
  _ = draw_prediction_on_image(image.numpy(), person, crop_region=None, close_figure=False, keep_input_size=True)
```

## 第1部分：预处理输入图像

由于我们的姿势分类器的输入是MoveNet模型的输出关键点，我们需要通过MoveNet运行已标记的图像，并将所有关键点数据和真实标签捕获到CSV文件中来生成我们的训练数据集。

我们为本教程提供的数据集是一个CG生成的瑜伽姿势数据集。它包含多个CG生成的模型在做5种不同瑜伽姿势的图像。目录已经分为`train`数据集和`test`数据集。

因此，在本节中，我们将下载瑜伽数据集并通过MoveNet运行，以便我们可以将所有关键点捕获到CSV文件中... **然而，将我们的瑜伽数据集输入MoveNet并生成这个CSV文件大约需要15分钟**。所以作为替代方案，你可以通过设置下面的`is_skip_step_1`参数为**True**，下载一个现成的瑜伽数据集的CSV文件。这样你将跳过此步骤，而是下载将在此预处理步骤中创建的相同CSV文件。

另一方面，如果你想用自己的图像数据集训练姿势分类器，你需要上传你的图像并运行此预处理步骤（保留`is_skip_step_1`为**False**）—按照以下说明上传你自己的姿势数据集。

```python
is_skip_step_1 = True #@param ["False", "True"] {type:"raw"}
```

### (可选) 上传你自己的姿势数据集

```python
use_custom_dataset = False #@param ["False", "True"] {type:"raw"}
dataset_is_split = False #@param ["False", "True"] {type:"raw"}
```

如果你想用自己标记的姿势（可以是任何姿势，不仅仅是瑜伽姿势）训练姿势分类器，请按照以下步骤进行：

1. 将上面的`use_custom_dataset`选项设置为**True**。
2. 准备一个包含你的图像数据集文件夹的压缩文件（ZIP、TAR或其他）。文件夹必须按照以下结构排序图像：
  如果你已经将数据集分为训练集和测试集，则将`dataset_is_split`设置为**True**。也就是说，你的图像文件夹必须包括“train”和“test”目录，如下所示：
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
  或者，如果你的数据集尚未拆分，则将`dataset_is_split`设置为**False**，我们将根据指定的拆分比例进行拆分。也就是说，你上传的图像文件夹应如下所示：
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
3. 点击左侧的**文件**标签（文件夹图标），然后点击**上传到会话存储**（文件图标）。
4. 选择你的压缩文件并等待上传完成后再继续。
5. 编辑以下代码块以指定你的压缩文件和图像目录的名称（默认情况下，我们期望一个ZIP文件，因此如果你的压缩文件是其他格式，也需要修改该部分）。
6. 现在运行其余的笔记本。

```python
import os
import random
import shutil

def split_into_train_test(images_origin, images_dest, test_split):
  _, dirs, _ = next(os.walk(images_origin))
  TRAIN_DIR = os.path.join(images_dest, 'train')
  TEST_DIR = os.path.join(images_dest, 'test')
  os.makedirs(TRAIN_DIR, exist_ok=True)
  os.makedirs(TEST_DIR, exist_ok=True)
  for dir in dirs:
    filenames = os.listdir(os.path.join(images_origin, dir))
    filenames = [os.path.join(images_origin, dir, f) for f in filenames if (
        f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.bmp'))]
    filenames.sort()
    random.seed(42)
    random.shuffle(filenames)
    os.makedirs(os.path.join(TEST_DIR, dir), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_DIR, dir), exist_ok=True)
    test_count = int(len(filenames) * test_split)
    for i, file in enumerate(filenames):
      if i < test_count:
        destination = os.path.join(TEST_DIR, dir, os.path.split(file)[1])
      else:
        destination = os.path.join(TRAIN_DIR, dir, os.path.split(file)[1])
      shutil.copyfile(file, destination)
    print(f'Moved {test_count} of {len(filenames)} from class "{dir}" into test.')
  print(f'Your split dataset is in "{images_dest}"')
```

```python
if use_custom_dataset:
  !unzip -q YOUR_DATASET_ARCHIVE_NAME.zip
  dataset_in = 'YOUR_DATASET_DIR_NAME'
  if not os.path.isdir(dataset_in):
    raise Exception("dataset_in is not a valid directory")
  if dataset_is_split:
    IMAGES_ROOT = dataset_in
  else:
    dataset_out = 'split_' + dataset_in
    split_into_train_test(dataset_in, dataset_out, test_split=0.2)
    IMAGES_ROOT = dataset_out
```

**注意：** 如果你使用`split_into_train_test()`来拆分数据集，它期望所有图像都是PNG、JPEG或BMP格式——它会忽略其他文件类型。

### 下载瑜伽数据集

```python
if not is_skip_step_1 and not use_custom_dataset:
  !wget -O yoga_poses.zip http://download.tensorflow.org/data/pose_classification/yoga_poses.zip
  !unzip -q yoga_poses.zip -d yoga_cg
  IMAGES_ROOT = "yoga_cg"
```

### 预处理`TRAIN`数据集

```python
if not is_skip_step_1:
  images_in_train_folder = os.path.join(IMAGES_ROOT, 'train')
  images_out_train_folder = 'poses_images_out_train'
  csvs_out_train_path = 'train_data.csv'
  preprocessor = MoveNetPreprocessor(images_in_folder=images_in_train_folder, images_out_folder=images_out_train_folder, csvs_out_path=csvs_out_train_path)
  preprocessor.process(per_pose_class_limit=None)
```

### 预处理`TEST`数据集

```python
if not is_skip_step_1:
  images_in_test_folder = os.path.join(IMAGES_ROOT, 'test')
  images_out_test_folder = 'poses_images_out_test'
  csvs_out_test_path = 'test_data.csv'
  preprocessor = MoveNetPreprocessor(images_in_folder=images_in_test_folder, images_out_folder=images_out_test_folder, csvs_out_path=csvs_out_test_path)
  preprocessor.process(per_pose_class_limit=None)
```

## 第2部分：训练一个姿势分类模型，将关键点坐标作为输入，并输出预测的标签

你将构建一个TensorFlow模型，该模型接受检测到的姿势关键点，然后计算姿势嵌入并预测姿势类别。模型包括两个子模型：

* 子模型1：从检测到的关键点坐标计算姿势嵌入（即特征向量）。
* 子模型2：通过多个`Dense`层传递姿势嵌入，以预测姿势类别。

然后你将根据第1部分预处理的数据集训练模型。

### (可选) 如果你没有运行第1部分，请下载预处理的数据集

```python
if is_skip_step_1:
  !wget -O train_data.csv http://download.tensorflow.org/data/pose_classification/yoga_train_data.csv
  !wget -O test_data.csv http://download.tensorflow.org/data/pose_classification/yoga_test_data.csv
  csvs_out_train_path = 'train_data.csv'
  csvs_out_test_path = 'test_data.csv'
  is_skipped_step_1 = True
```

### 将预处理的CSV加载到`TRAIN`和`TEST`数据集中

```python
def load_pose_landmarks(csv_path):
  dataframe = pd.read_csv(csv_path)
  df_to_process = dataframe.copy()
  df_to_process.drop(columns=['file_name'], inplace=True)
  classes = df_to_process.pop('class_name').unique()
  y = df_to_process.pop('class_no')
  X = df_to_process.astype('float64')
  y = keras.utils.to_categorical(y)
  return X, y, classes, dataframe

X, y, class_names, _ = load_pose_landmarks(csvs_out_train_path)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
X_test, y_test, _, df_test = load_pose_landmarks(csvs_out_test_path)
```

### 定义将姿势关键点转换为姿势嵌入（即特征向量）的函数

```python
def get_center_point(landmarks, left_bodypart, right_bodypart):
  left = tf.gather(landmarks, left_bodypart.value, axis=1)
  right = tf.gather(landmarks, right_bodypart.value, axis=1)
  center = left * 0.5 + right * 0.5
  return center

def get_pose_size(landmarks, torso_size_multiplier=2.5):
  hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
  shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER)
  torso_size = tf.linalg.norm(shoulders_center - hips_center)
  pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
  pose_center_new = tf.expand_dims(pose_center_new, axis=1)
  pose_center_new = tf.broadcast_to(pose_center_new, [tf.size(landmarks) // (17*2), 17, 2])
  d = tf.gather(landmarks - pose_center_new, 0, axis=0, name="dist_to_pose_center")
  max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))
  pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)
  return pose_size

def normalize_pose_landmarks(landmarks):
  pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
  pose_center = tf.expand_dims(pose_center, axis=1)
  pose_center = tf.broadcast_to(pose_center, [tf.size(landmarks) // (17*2), 17, 2])
  landmarks = landmarks - pose_center
  pose_size = get_pose_size(landmarks)
  landmarks /= pose_size
  return landmarks

def landmarks_to_embedding(landmarks_and_scores):
  reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)
  landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])
  embedding = keras.layers.Flatten()(landmarks)
  return embedding
```

### 定义用于姿势分类的Keras模型

我们的Keras模型接受检测到的姿势关键点，然后计算姿势嵌入并预测姿势类别。

```python
inputs = tf.keras.Input(shape=(51))
embedding = landmarks_to_embedding(inputs)
layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
layer = keras.layers.Dropout(0.5)(layer)
layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.5)(layer)
outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)
model = keras.Model(inputs, outputs)
model.summary()
```

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_path = "weights.best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                              patience=20)

history = model.fit(X_train, y_train,
                    epochs=200,
                    batch_size=16,
                    validation_data=(X_val, y_val),
                    callbacks=[checkpoint, earlystopping])
```

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['TRAIN', 'VAL'], loc='lower right')
plt.show()
```

```python
loss, accuracy = model.evaluate(X_test, y_test)
```

### 绘制混淆矩阵以更好地理解模型性能

```python
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=55)
  plt.yticks(tick_marks, classes)
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
              horizontalalignment="center",
              color="white" if cm[i, j] > thresh else "black")
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()

y_pred = model.predict(X_test)
y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
y_true_label = [class_names[i] for i in np.argmax(y_test, axis=1)]
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
plot_confusion_matrix(cm, class_names, title='Confusion Matrix of Pose Classification Model')
print('\nClassification Report:\n', classification_report(y_true_label, y_pred_label))
```

### (可选) 调查错误预测

你可以查看`TEST`数据集中预测错误的姿势，看看是否可以提高模型的准确性。

注意：这仅在你运行了第1步时有效，因为你需要本地计算机上的姿势图像文件来显示它们。

```python
if is_skip_step_1:
  raise RuntimeError('You must have run step 1 to run this cell.')

IMAGE_PER_ROW = 3
MAX_NO_OF_IMAGE_TO_PLOT = 30
false_predict = [id_in_df for id_in_df in range(len(y_test)) if y_pred_label[id_in_df] != y_true_label[id_in_df]]
if len(false_predict) > MAX_NO_OF_IMAGE_TO_PLOT:
  false_predict = false_predict[:MAX_NO_OF_IMAGE_TO_PLOT]
row_count = len(false_predict) // IMAGE_PER_ROW + 1
fig = plt.figure(figsize=(10 * IMAGE_PER_ROW, 10 * row_count))
for i, id_in_df in enumerate(false_predict):
  ax = fig.add_subplot(row_count, IMAGE_PER_ROW, i + 1)
  image_path = os.path.join(images_out_test_folder, df_test.iloc[id_in_df]['file_name'])
  image = cv2.imread(image_path)
  plt.title("Predict: %s; Actual: %s" % (y_pred_label[id_in_df], y_true_label[id_in_df]))
  plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
```

## 第3部分：将姿势分类模型转换为TensorFlow Lite

你将把Keras姿势分类模型转换为TensorFlow Lite格式，以便将其部署到移动应用程序、Web浏览器和边缘设备。在转换模型时，你将应用[动态范围量化](https://www.tensorflow.org/lite/performance/post_training_quant)，以在精度损失不显著的情况下将姿势分类TensorFlow Lite模型大小减少约4倍。

注意：TensorFlow Lite支持多种量化方案。如果你有兴趣了解更多，请参阅[文档](https://www.tensorflow.org/lite/performance/model_optimization)。

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
print('Model size: %dKB' % (len(tflite_model) / 1024))
with open('pose_classifier.tflite', 'wb') as f:
  f.write(tflite_model)
```

然后你将编写标签文件，其中包含从类索引到人类可读类名称的映射。

```python
with open('pose_labels.txt', 'w') as f:
  f.write('\n'.join(class_names))
```

由于你已经应用了量化来减少模型大小，让我们评估量化后的TFLite模型，检查精度下降是否可以接受。

```python
def evaluate_model(interpreter, X, y_true):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]
  y_pred = []
  for i in range(len(y_true)):
    test_image = X[i: i + 1].astype('float32')
    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    output = interpreter.tensor(output_index)
    predicted_label = np.argmax(output()[0])
    y_pred.append(predicted_label)
  y_pred = keras.utils.to_categorical(y_pred)
  return accuracy_score(y_true, y_pred)

classifier_interpreter = tf.lite.Interpreter(model_content=tflite_model)
classifier_interpreter.allocate_tensors()
print('Accuracy of TFLite model: %s' % evaluate_model(classifier_interpreter, X_test, y_test))
```

现在你可以下载TFLite模型（`pose_classifier.tflite`）和标签文件（`pose_labels.txt`），以分类自定义姿势。请参阅[Android](https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/android)和[Python/Raspberry Pi](https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/raspberry_pi)示例应用程序，以获取如何使用TFLite姿势分类模型的端到端示例。

```python
!zip pose_classifier.zip pose_labels.txt pose_classifier.tflite
try:
  from google.colab import files
  files.download('pose_classifier.zip')
except:
  pass
```