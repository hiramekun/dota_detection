import argparse
import os
import shutil

from tqdm import tqdm

from dota_utils import extract_annotations, extract_ids


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./train')
    args = parser.parse_args()
    return args


def main(args):
    out = f'{args.data_dir}_clean'
    if os.path.exists(out):
        shutil.rmtree(out)
    os.mkdir(out)
    os.mkdir(f'{out}/images')
    os.mkdir(f'{out}/annotations')

    fpath = f'{args.data_dir}/images'
    ids = extract_ids(fpath)

    for id_ in tqdm(ids):
        anno_file = f'{args.data_dir}/annotations/{id_}.txt'
        image_file = f'{fpath}/{id_}.png'
        bbox, labels, difficult = extract_annotations(anno_file, False)

        # exclude too large images.
        if os.path.getsize(image_file) < 1e+7:
            # exclude no bounding box images.
            if len(bbox) != 0:
                shutil.copyfile(image_file,
                                f'{out}/images/{id_}.png')
                shutil.copyfile(anno_file,
                                f'{out}/annotations/{id_}.txt')


if __name__ == '__main__':
    main(arg_parser())
