import random

import numpy as np


def resize_with_random_interpolation(img, size, return_param=False):
    """Resize an image with a randomly selected interpolation method.

    This function is similar to :func:`chainercv.transforms.resize`, but
    this chooses the interpolation method randomly.

    This data augmentation is used in training of
    Single Shot Multibox Detector [#]_.

    Note that this function requires :mod:`cv2`.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        img (~numpy.ndarray): An array to be transformed.
            This is in CHW format and the type should be :obj:`numpy.float32`.
        size (tuple): This is a tuple of length 2. Its elements are
            ordered as (height, width).
        return_param (bool): Returns information of interpolation.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`img` that is the result of rotation.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **interpolatation**: The chosen interpolation method.

    """

    import cv2

    cv_img = img.transpose((1, 2, 0))

    inters = (
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_NEAREST,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    )
    inter = random.choice(inters)
    H, W = size
    # This is for avoiding dead lock in Multiprocess.
    # See this issue @link{https://github.com/chainer/chainercv/issues/386}
    cv2.setNumThreads(0)
    cv_img = cv2.resize(cv_img, (W, H), interpolation=inter)

    # If input is a grayscale image, cv2 returns a two-dimentional array.
    if len(cv_img.shape) == 2:
        cv_img = cv_img[:, :, np.newaxis]

    img = cv_img.astype(np.float32).transpose((2, 0, 1))

    if return_param:
        return img, {'interpolation': inter}
    else:
        return img