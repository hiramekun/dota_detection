import os

place_labels = ('plane',
                'baseball-diamond',
                'bridge',
                'ground-track-field',
                'small-vehicle',
                'large-vehicle',
                'ship',
                'tennis-court',
                'basketball-court',
                'storage-tank',
                'soccer-ball-field',
                'roundabout',
                'harbor',
                'swimming-pool',
                'helicopter')


def extract_annotations(fname, use_difficult):
    with open(fname) as f:
        lines = f.readlines()

    if not lines[1].startswith('gsd:'):
        Exception('no gsd')

    bbox = []
    labels = []
    difficult = []

    # for each object in the image.
    for line in lines[2:]:
        split_text = line.split(' ')

        difficult_ = int(split_text[-1].strip())
        if not use_difficult and difficult_ == 1:
            continue
        difficult.append(difficult_)

        xs = []
        ys = []
        points = ('x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4')
        for point, text in zip(points, split_text):
            if point.startswith('x'):
                xs.append(int(text))
            elif point.startswith('y'):
                ys.append(int(text))
        # sort xs and ys to get max/min
        sorted_x = sorted(xs)
        sorted_y = sorted(ys)
        min_x = sorted_x[0]
        min_y = sorted_y[0]
        max_x = sorted_x[3]
        max_y = sorted_y[3]

        # subtract 1 to make pixel indexes 0-based
        bbox.append([min_y - 1, min_x - 1, max_y - 1, max_x - 1])

        name = split_text[-2].lower().strip()
        labels.append(place_labels.index(name))

    return bbox, labels, difficult


def extract_ids(dir_path):
    return [os.path.splitext(id_)[0] for id_ in os.listdir(dir_path)]
