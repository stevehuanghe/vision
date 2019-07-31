
import numpy as np
import torch
from torch.autograd import gradcheck

from torchvision import ops

from itertools import product
import unittest


class RoIAlignTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)
        cls.dtype = torch.float32
        cls.x = torch.rand(1, 1, 10, 10, dtype=cls.dtype)
        # cls.x = torch.ones(1, 1, 10, 10, dtype=cls.dtype)

        cls.single_roi = torch.tensor([[0, 0, 0, 4, 4]],  # format is (xyxy)
                                      dtype=cls.dtype)
        cls.rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                                 [0, 0, 5, 4, 9],
                                 [0, 5, 5, 9, 9]],
                                dtype=cls.dtype)
        cls.gt_y_single = torch.tensor(
            [[[[0.41617328, 0.5040753, 0.25266218, 0.4296828, 0.29928464],
               [0.5210769, 0.57222337, 0.2524979, 0.32063985, 0.32635176],
               [0.73108256, 0.6114335, 0.62033176, 0.8188273, 0.5562218],
               [0.83115816, 0.70803946, 0.7084047, 0.74928707, 0.7769296],
               [0.54266506, 0.45964524, 0.5780159, 0.80522037, 0.7321807]]]], dtype=cls.dtype)

        cls.gt_y_multiple = torch.tensor(
            [[[[0.49311584, 0.35972416, 0.40843594, 0.3638034, 0.49751836],
               [0.70881474, 0.75481665, 0.5826779, 0.34767765, 0.46865487],
               [0.4740328, 0.69306874, 0.3617804, 0.47145438, 0.66130304],
               [0.6861706, 0.17634538, 0.47194335, 0.42473823, 0.37930614],
               [0.62666404, 0.49973848, 0.37911576, 0.5842756, 0.7176864]]],
             [[[0.67499936, 0.6607055, 0.42656037, 0.46134934, 0.42144877],
               [0.7471722, 0.7235433, 0.14512213, 0.13031253, 0.289369],
               [0.8443615, 0.6659734, 0.23614208, 0.14719573, 0.4268827],
               [0.69429564, 0.5621515, 0.5019923, 0.40678093, 0.34556213],
               [0.51315194, 0.7177093, 0.6494485, 0.6775592, 0.43865064]]],
             [[[0.24465509, 0.36108392, 0.64635646, 0.4051828, 0.33956185],
               [0.49006107, 0.42982674, 0.34184104, 0.15493104, 0.49633422],
               [0.54400194, 0.5265246, 0.22381854, 0.3929715, 0.6757667],
               [0.32961223, 0.38482672, 0.68877804, 0.71822757, 0.711909],
               [0.561259, 0.71047884, 0.84651315, 0.8541089, 0.644432]]]], dtype=cls.dtype)

        cls.x_grad = torch.tensor(
            [[[[0.075625, 0.15125, 0.15124999, 0.15125002, 0.15812504,
                0.15812503, 0.15124999, 0.15124999, 0.15125006, 0.0756249],
               [0.15125, 0.30250007, 0.3025, 0.30250007, 0.31625012,
                0.31625003, 0.3025, 0.3025, 0.30250013, 0.1512498],
               [0.15124999, 0.3025, 0.30249995, 0.3025, 0.31625006,
                0.31625, 0.30249995, 0.30249995, 0.30250007, 0.15124978],
               [0.15125002, 0.30250007, 0.3025, 0.30250007, 0.31625012,
                0.3162501, 0.3025, 0.3025, 0.30250013, 0.15124981],
               [0.15812504, 0.31625012, 0.31625006, 0.31625012, 0.33062524,
                0.3306251, 0.31625006, 0.31625006, 0.3162502, 0.15812483],
               [0.5181251, 1.0962502, 1.0362502, 1.0962503, 0.69062525, 0.6906252,
                1.0962502, 1.0362502, 1.0962503, 0.5181248],
               [0.93125, 1.9925, 1.8624997, 1.9925, 1.0962502, 1.0962502,
                1.9925, 1.8624998, 1.9925, 0.9312496],
               [0.8712501, 1.8625, 1.7425002, 1.8625001, 1.0362502, 1.0362502,
                1.8625, 1.7425001, 1.8625002, 0.8712497],
               [0.93125004, 1.9925, 1.8625002, 1.9925, 1.0962503, 1.0962503,
                1.9925001, 1.8625001, 1.9925001, 0.93124974],
               [0.43562484, 0.9312497, 0.8712497, 0.9312497, 0.5181249, 0.5181248,
                0.9312496, 0.8712497, 0.93124974, 0.43562466]]]], dtype=cls.dtype)
    '''
    def test_roi_align_basic_cpu(self):
        device = torch.device('cpu')
        x = self.x.to(device)
        single_roi = self.single_roi.to(device)
        gt_y_single = self.gt_y_single.to(device)

        pool_h, pool_w = (5, 5)
        roi_align = ops.RoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2).to(device=device)
        y = roi_align(x, single_roi)

        assert torch.allclose(gt_y_single, y), 'RoIAlign layer incorrect for single ROI on CPU'

        y = roi_align(x.transpose(2, 3).contiguous().transpose(2, 3), single_roi)
        assert torch.allclose(gt_y_single, y), 'RoIAlign layer incorrect for single ROI on CPU'

    def test_roi_align_cpu(self):
        device = torch.device('cpu')
        x = self.x.to(device)
        rois = self.rois.to(device)
        gt_y_multiple = self.gt_y_multiple.to(device)

        pool_h, pool_w = (5, 5)
        roi_align = ops.RoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2).to(device=device)
        y = roi_align(x, rois)

        assert torch.allclose(gt_y_multiple, y), 'RoIAlign layer incorrect for multiple ROIs on CPU'

        y = roi_align(x.transpose(2, 3).contiguous().transpose(2, 3), rois)
        assert torch.allclose(gt_y_multiple, y), 'RoIAlign layer incorrect for multiple ROIs on CPU'

    def test_roi_align_gradient_cpu(self):
        """
        Compute gradients for RoIAlign with multiple bounding boxes on CPU
        """
        device = torch.device('cpu')
        pool_h, pool_w = (5, 5)
        roi_align = ops.RoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2).to(device=device)

        x = self.x.to(device).clone()
        rois = self.rois.to(device)
        gt_grad = self.x_grad.to(device)

        x.requires_grad = True
        y = roi_align(x, rois)
        s = y.sum()
        s.backward()

        assert torch.allclose(x.grad, gt_grad), 'gradient incorrect for RoIAlign CPU'

    def test_roi_align_gradient_single_cpu(self):
        """
        Compute gradients for RoIAlign with multiple bounding boxes on CPU
        """
        device = torch.device('cpu')
        pool_h, pool_w = (5, 5)
        roi_align = ops.RoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2).to(device=device)

        x = self.x.to(device).clone()
        rois = self.single_roi.to(device)
        gt_grad = self.x_grad.to(device)

        x.requires_grad = True
        y = roi_align(x, rois)
        y.retain_grad()
        s = y.sum()
        s.backward()

        # assert torch.allclose(x.grad, gt_grad), 'gradient incorrect for RoIAlign CPU'

    def test_roi_align_gradcheck_cpu(self):
        dtype = torch.float64
        device = torch.device('cpu')
        m = ops.RoIAlign((5, 5), 0.5, 1).to(dtype=dtype, device=device)
        x = torch.rand(1, 1, 10, 10, dtype=dtype, device=device, requires_grad=True)
        rois = self.rois.to(device=device, dtype=dtype)

        def func(input):
            return m(input, rois)

        assert gradcheck(func, (x,)), 'gradcheck failed for RoIAlign CPU'
        assert gradcheck(func, (x.transpose(2, 3),)), 'gradcheck failed for RoIAlign CPU'
    '''

    def test_roi_align_gradcheck_cpu(self):
        dtype = torch.float64
        device = torch.device('cpu')
        m = ops.RoIAlign((5, 5), 1, 1).to(dtype=dtype, device=device)
        x = torch.rand(1, 1, 10, 10, dtype=dtype, device=device, requires_grad=True)
        # x = torch.arange(0, 100, device=device, dtype=dtype, requires_grad=True).view(1,1,10,10)
        rois = torch.tensor([[0, 0, 3, 3]], dtype=dtype, requires_grad=True)
        # rois = self.single_roi.clone()
        rois = rois.to(device=device, dtype=dtype)
        rois.requires_grad = True

        def func(img, frois):
            n_rois = frois.size(0)
            idxs = torch.zeros(n_rois, 1, dtype=dtype).to(device)
            frois = torch.cat((idxs, frois), dim=1)
            return m(img, frois)

        # print("input")
        # print(x)
        # y = m(x, rois)
        # y.retain_grad()
        # print("roi_align output:")
        # print(y)
        # print("###################")
        # s = y.sum()
        # s.backward()
        # print("roi grad:")
        # print(rois.grad)

        assert gradcheck(func, (x, rois)), 'gradcheck failed for RoIAlign CPU'
        # assert gradcheck(func, (x.transpose(2, 3), rois)), 'gradcheck failed for RoIAlign CPU'


if __name__ == '__main__':
    unittest.main()
