import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .body import Body
from .hand import Hand
from annotator.util import annotator_ckpts_path


body_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth"
hand_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth"


class OpenposeDetector:
    def __init__(self):
        body_modelpath = os.path.join(annotator_ckpts_path, "body_pose_model.pth")
        hand_modelpath = os.path.join(annotator_ckpts_path, "hand_pose_model.pth")

        if not os.path.exists(hand_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(body_model_path, model_dir=annotator_ckpts_path)
            load_file_from_url(hand_model_path, model_dir=annotator_ckpts_path)

        self.body_estimation = Body(body_modelpath)
        self.hand_estimation = Hand(hand_modelpath)

    def __call__(self, oriImg, hand=False):
        oriImg = oriImg[:, :, ::-1].copy()
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            # flownet的heatmap，点坐标的十八层图像
            key_points = self.adjust_candidate(candidate.copy(), subset.copy())
            x = key_points[:, 0]  # x坐标
            y = key_points[:, 1]
            # x[8:-1] = x[9:]  # 对deepfashion是需要的，但生成的pose基本上是18点可分的
            # y[8:-1] = y[9:]
            x[x == 0] = -1
            y[y == 0] = -1
            # array1 = np.concatenate([y[:, None], x[:, None]], -1)
            # array1 = self.cords_to_map(array1, (256, 256), (512, 512))  # 转成gfla接受的256维度
            array1 = np.concatenate([x[:, None], y[:, None]], -1)
            array1 = self.kp_to_map((256, 256), array1)  # 这里也是h w n
            array1 = np.transpose(array1, (2, 0, 1))  # 变成18 h w
            # 结束
            canvas = np.zeros_like(oriImg)
            canvas = util.draw_bodypose(canvas, candidate, subset)
            if hand:
                hands_list = util.handDetect(candidate, subset, oriImg)
                all_hand_peaks = []
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :])
                    peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                    peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                    all_hand_peaks.append(peaks)
                canvas = util.draw_handpose(canvas, all_hand_peaks)
            return canvas, dict(candidate=candidate.tolist(), subset=subset.tolist()), array1

    def adjust_candidate(self, candi, sub):
        num_keypoints = 18
        # 创建一个固定形状的新 candidate 数组，初始值全为 0
        adjusted_candidate = np.zeros((num_keypoints, 4))
        # indices = np.where(subset[0] != -1)[0]
        for i in range(num_keypoints):
            index = sub[0, i]
            index = int(index)
            if index != -1:
                adjusted_candidate[i] = candi[index]
        return adjusted_candidate

    def cords_to_map(self, cords, img_size, old_size=None, affine_matrix=None, sigma=8):
        MISSING_VALUE = -1
        old_size = img_size if old_size is None else old_size
        cords = cords.astype(float)
        result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
        for i, point in enumerate(cords):
            if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
                continue
            point[0] = point[0] / old_size[0] * img_size[0]
            point[1] = point[1] / old_size[1] * img_size[1]
            if affine_matrix is not None:
                point_ = np.dot(affine_matrix, np.matrix([point[1], point[0], 1]).reshape(3, 1))
                point_0 = int(point_[1])
                point_1 = int(point_[0])
            else:
                point_0 = int(point[0])
                point_1 = int(point[1])
            xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            # gaussian
            result[..., i] = np.exp(-((yy - point_0) ** 2 + (xx - point_1) ** 2) / (2 * sigma ** 2))
            # result[..., i] = np.exp(-((yy - point_0) ** 2 + (xx - point_1) ** 2) / (sigma ** 2))   # 有的版本这里不除以2
            # binary
            # result[..., i] = np.exp((yy - point_0) ** 2 + (xx - point_1) ** 2 <= sigma ** 2)
        return result

    def kp_to_map(self, img_sz, kps, mode='binary', radius=8):
        '''
        因为用的deepfashion的权重，原本intrinsic_flow设置radius 就是8
        Keypoint cordinates to heatmap map.
        Input:
            img_size (w,h): size of heatmap
            kps (N,2): (x,y) cordinates of N keypoints
            mode: 'gaussian' or 'binary'
            radius: radius of each keypoints in heatmap
        Output:
            m (h,w,N): encoded heatmap
        '''
        w, h = img_sz
        x_grid, y_grid = np.meshgrid(range(w), range(h), indexing='xy')
        m = []
        for x, y in kps:
            if x == -1 or y == -1:
                m.append(np.zeros((h, w)).astype(np.float32))
            else:
                x = x / 2      # 这里除以2是从512维度缩放到256
                y = y / 2
                if mode == 'gaussian':
                    m.append(np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (radius ** 2)).astype(np.float32))
                elif mode == 'binary':
                    m.append(((x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius ** 2).astype(np.float32))
                else:
                    raise NotImplementedError()
        m = np.stack(m, axis=2)
        return m
