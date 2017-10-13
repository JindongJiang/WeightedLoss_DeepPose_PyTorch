import numpy as np
import cv2
from torchvision import transforms

label_colours = [(148, 65, 137), (255, 116, 69), (86, 156, 137), (255, 116, 69), (148, 65, 137),
                 (), (202, 179, 158), (155, 99, 235), (), (155, 99, 235), (202, 179, 158),
                 (), (161, 107, 108), (76, 152, 126)]


def keypoint_painter(images, landmarks, img_h, img_w, numpy_array=False):
    images = images.clone().cpu().data.numpy().transpose([0, 2, 3, 1]) * \
             np.array([0.229, 0.224, 0.225]).reshape((1, 1, 1, 3)) + \
             np.array([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3))
    images = np.clip(images, a_min=0, a_max=1) * 255
    landmarks = landmarks.clone().cpu().data.numpy()
    imgs_tensor = []
    for img, lm in zip(images, landmarks):
        all_x = lm[:len(lm) // 2]
        all_y = lm[len(lm) // 2:]
        all_x = (all_x * (img_w // 2) + img_w // 2).astype(np.int)
        all_y = (all_y * (img_h // 2) + img_h // 2).astype(np.int)
        img = cv2.resize(img, (img_w, img_h))
        for x, y in zip(all_x, all_y):
            img = cv2.circle(img.copy(), (x, y), radius=1, thickness=2, color=(255, 0, 0))
        if numpy_array:
            imgs_tensor.append(img.astype(np.uint8))
        else:
            imgs_tensor.append(transforms.ToTensor()(img))
    return imgs_tensor


def joint_painter(images, landmarks, img_h, img_w, numpy_array=False):
    images = images.clone().cpu().data.numpy().transpose([0, 2, 3, 1]) * \
             np.array([0.229, 0.224, 0.225]).reshape((1, 1, 1, 3)) + \
             np.array([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3))
    images = np.clip(images, a_min=0, a_max=1) * 255
    landmarks = landmarks.clone().cpu().data.numpy()
    imgs_tensor = []
    for img, lm in zip(images, landmarks):
        all_x = lm[:len(lm) // 2]
        all_y = lm[len(lm) // 2:]
        all_x = (all_x * (img_w // 2) + img_w // 2).astype(np.int)
        all_y = (all_y * (img_h // 2) + img_h // 2).astype(np.int)
        img = cv2.resize(img, (img_w, img_h))
        for i in range(len(lm) // 2):
            if i in [5, 8, 11, 13] or not 0 < all_x[i] <= img_w or not 0 < all_x[i + 1] <= img_w \
                    or not 0 < all_y[i] <= img_h or not 0 < all_y[i + 1] <= img_h:
                continue
            img = cv2.line(img.copy(), (all_x[i], all_y[i]), (all_x[i + 1], all_y[i + 1]),
                           color=label_colours[i], thickness=2, lineType=8)
        if 0 < all_x[12] <= img_w or 0 < all_y[12] <= img_h:
            if 0 < all_x[8] <= img_w or 0 < all_y[8] <= img_h:
                img = cv2.line(img.copy(), (all_x[8], all_y[8]), (all_x[12], all_y[12]),
                               color=label_colours[-1], thickness=2, lineType=8)
            if 0 < all_x[9] <= img_w or 0 < all_y[9] <= img_h:
                img = cv2.line(img.copy(), (all_x[9], all_y[9]), (all_x[12], all_y[12]),
                               color=label_colours[-1], thickness=2, lineType=8)
        if numpy_array:
            imgs_tensor.append(img.astype(np.uint8))
        else:
            imgs_tensor.append(transforms.ToTensor()(img))
    return imgs_tensor
