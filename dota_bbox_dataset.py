import warnings

import chainer
import numpy as np
from chainercv.utils import read_image

from dota_utils import extract_annotations, extract_ids


class DotaBboxDataset(chainer.dataset.DatasetMixin):
    """Bounding box dataset

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`chainercv.datasets.voc_bbox_label_names`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir='auto', split='train',
                 use_difficult=False, return_difficult=False):
        if data_dir == 'auto':
            data_dir = '.'

        if split not in ['train', 'trainval', 'val']:
            if not (split == 'test'):
                warnings.warn(
                    'please pick split from \'train\', \'trainval\', \'val\''
                )

        fpath = f'{data_dir}/{split}/images'

        self.ids = extract_ids(fpath)
        self.data_dir = data_dir
        self.split = split
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """

        id_ = self.ids[i]
        print(id_)

        fname = f'{self.data_dir}/{self.split}/annotations/{id_}.txt'
        bbox, labels, difficult = extract_annotations(fname, self.use_difficult)

        bbox = np.stack(bbox).astype(np.float32)
        labels = np.stack(labels).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool)

        # Load a image
        img_file = f'{self.data_dir}/{self.split}/images/{id_}.png'
        img = read_image(img_file, color=True)
        if self.return_difficult:
            return img, bbox, labels, difficult
        return img, bbox, labels
