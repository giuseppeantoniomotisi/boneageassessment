import unittest
import sys
sys.path.append(sys.path[0].replace('/preprocessing/tests','')) # baa directory
sys.path.append(sys.path[0].replace('/tests','')) # age directory
import numpy as np
import tools

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Initialize Preprocessing instance for testing
        self.preprocessing = tools.Preprocessing()

    def test_in_box(self):
        #Test in box with a point inside a box with size (90, 120)
        coords = self.preprocessing.in_box(45, 67, 90, 120)
        self.assertSequenceEqual(coords, (45, 67))
        #Test in box with a point outside a box with size (90, 120)
        out_coords = self.preprocessing.in_box(93, 140, 90, 120)
        self.assertSequenceEqual(out_coords, (89, 119))
    
    def test_rectangle(self):
        #Test rectangle function
        fake_image = np.ones((1080, 1080))
        coords = [[220, 780],[680, 180]]
        top_left, bottom_right = self.preprocessing.rectangle(fake_image, coords)
        self.assertSequenceEqual(top_left, (58, 842))
        self.assertSequenceEqual(bottom_right, (942, 18))


    def test_cut_peak(self):
        #Test cut_peak function
        h = np.zeros(50)
        for i in range(0, 50):
            h[i] = (1000/(i+1))
        new_index = self.preprocessing.cut_peak(h, 0, 10, 'testimage')
        self.assertEqual(new_index, 9)
    
    def test_square(self):
        #Test square function
        fake_image = np.ones((720, 1080, 3))
        squared_image = self.preprocessing.square(fake_image)
        self.assertSequenceEqual(squared_image.shape, (1080,1080,3))

    def test_resize(self):
        #Test resize function
        fake_image = np.ones((720, 720))
        resized_image = self.preprocessing.resize(fake_image, (399,399))
        self.assertSequenceEqual(resized_image.shape, (399,399))

if __name__ == '__main__':
    unittest.main()
