import unittest
import os
import sys
sys.path.append(sys.path[0].replace('/age/tests','')) # baa directory
sys.path.append(sys.path[0].replace('/tests','')) # age directory
import numpy as np
from model import mean_absolute_deviation, mean_absolute_error, BoneAgeAssessment, BaaModel
from utils import extract_info

class TestBoneAgeAssessment(unittest.TestCase):
    def setUp(self):
        # Initialize BoneAgeAssessment instance for testing
        self.bone_age_assessment = BoneAgeAssessment()

    def test_initialization(self):
        # Test initialization
        self.assertEqual(self.bone_age_assessment.image_size, (399, 399))
        self.assertEqual(self.bone_age_assessment.batch_size, (32, 32, 1396))
        self.assertEqual(self.bone_age_assessment.lr, 1e-05)
        self.assertEqual(self.bone_age_assessment.epochs, 20)

    def test_batch_size_update(self):
        # Test batch size update
        self.bone_age_assessment.__update_batch_size__([16, 16, 100], 'all')
        self.assertEqual(self.bone_age_assessment.batch_size, [16, 16, 100])

    def test_learning_rate_update(self):
        # Test learning rate update
        self.bone_age_assessment.__update_lr__(1e-04)
        self.assertEqual(self.bone_age_assessment.lr, 1e-04)

    def test_epochs_update(self):
        # Test epochs update
        self.bone_age_assessment.__update_epochs__(30)
        self.assertEqual(self.bone_age_assessment.epochs, 30)

    def test_show_info(self):
        # Test __show_info__ method
        info = self.bone_age_assessment.__show_info__()
        self.assertIsInstance(info, dict)
        self.assertIn('image size', info)
        self.assertIn('batch size', info)
        self.assertIn('learning rate', info)
        self.assertIn('number of epochs', info)
        self.assertIn('weights loc', info)

    def test_get_dataframe(self):
        # Test __get_dataframe__ method
        train_df = self.bone_age_assessment.__get_dataframe__('train')
        self.assertIsNotNone(train_df)
        # self.assertEqual(len(train_df), 9824)

    def test_get_generator(self):
        # Test __get_generator__ method
        train_generator = self.bone_age_assessment.__get_generator__('train')
        self.assertIsNotNone(train_generator)

    def test_change_training(self):
        # Test __change_training__ method
        self.bone_age_assessment.__change_training__(balanced=False)
        train_df = self.bone_age_assessment.__get_dataframe__('train')
        self.assertIsNotNone(train_df)
        # self.assertEqual(len(train_df), 27170)

class TestStatistics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.ones(100, dtype=float)
        self.y_pred = np.ones(100, dtype=float)
    
    def test_mean_absolute_error(self):
        # Test mean_absolute_error function
        self.assertEqual(mean_absolute_error(self.y_true, self.y_pred), 0)
    
    def test_mean_absolute_deviation(self):
        # Test mean_absolute_deviation function
        self.assertEqual(mean_absolute_deviation(self.y_true, self.y_pred), 0)

class TestModel(unittest.TestCase):
    def setUp(self):
        self.input_shape = (399,399,3)
        self.l2_factor = 1e-03
        self.summ = False
        self.baa = BaaModel(input_size=self.input_shape, summ=self.summ)
    
    def test_model(self):
        # Test vgg16regression_l2 function
        model = self.baa.vgg16regression_l2(self.l2_factor)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
