"""Unit test of pose classification."""

import unittest

import cv2
from ml.classifier import Classifier
from ml.movenet import Movenet

_ESTIMATION_MODEL = 'movenet_lightning'
_CLASSIFIER_MODEL = 'classifier'
_TEST_IMAGE = 'test_data/image3.jpeg'
_LABELS = 'labels.txt'


class ClassifierTest(unittest.TestCase):

  def test_pose_classification(self):
    """Test if yoga pose classifier returns correct result on test image."""
    # Detect the pose from the input image
    pose_detector = Movenet(_ESTIMATION_MODEL)
    image = cv2.imread(_TEST_IMAGE)
    person = pose_detector.detect(image)

    # Initialize a pose classifier
    classifier = Classifier(_CLASSIFIER_MODEL, _LABELS)
    categories = classifier.classify_pose(person)
    class_name = categories[0].label
    self.assertEqual(class_name, 'tree',
                     'Predicted pose is different from ground truth.')


if __name__ == '__main__':
  unittest.main()
