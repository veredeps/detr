from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import h5py
import json
import os
import scipy.misc
import sys
import numpy as np
import cv2
from PIL import Image
import random
import itertools
import tqdm

random.seed(123456)


# sample_random = random.Random()
# sample_random.seed(123456)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('--out_dir', default='./', type=str,
                        help="output dir for json files")
    parser.add_argument('--train_json_name', default='train.json', type=str,
                        help="name of train json file")
    parser.add_argument('--valid_json_name', default='valid.json', type=str,
                        help="name of valid json file")
    parser.add_argument('--data_dir', default='./', type=str,
                        help="data dir for annotations to be converted")
    parser.add_argument('--image_dir', default='JPEGImages', type=str,
                        help="image dir under data dir")
    parser.add_argument('--mask_dir', default='Annotations', type=str,
                        help="mask image dir under data dir")
    parser.add_argument('--image_type', default='jpg', type=str,
                        help="image dir under data dir")
    parser.add_argument('--mask_type', default='png', type=str,
                        help="mask image dir under data dir")
    parser.add_argument('--annotation_dir', default='Annotations', type=str,
                        help="image dir under data dir")
    parser.add_argument('--train_set_size', default=50, type=int,
                        help="size of train set to generate")
    parser.add_argument('--valid_set_size', default=5, type=int,
                        help="size of valid set to generate")
    parser.add_argument('--valid_set_video_percent', default=0.2, type=float,
                        help="precentage of scenarios selected for valid set")
    parser.add_argument('--max_videos_to_parse', default=-1, type=int,
                        help="maximal number of videos to parse (-1 no linit)")
    return parser.parse_args()


def collect_images_from_dir(args):
    images_base_dir = os.path.join(args.data_dir, args.image_dir)
    image_unique_id = 0
    images = []
    videos = {}
    for vid_ind, video in enumerate(sorted(os.listdir(images_base_dir))):
        print('collecting inamges of video %s' % video)
        video_path = os.path.join(images_base_dir, video)
        videos[video] = []
        for img_ind, image in enumerate(sorted(os.listdir(video_path))):
            im = Image.open(
                os.path.join(video_path, image))  # this is not so coasty because the image is not laoded in to memory
            width, height = im.size
            image_dic = {
                "file_name": image,
                "video_name": video,
                "height": height,
                "width": width,
                "id": image_unique_id
            }
            images.append(image_dic)
            videos[video].append(image_dic)
            image_unique_id += 1

    return images, videos


def poly_to_box(poly):
    """Convert a polygon into an array of tight bounding box."""
    box = np.zeros(4, dtype=np.float32)
    box[0] = min(poly[:, 0])
    box[2] = max(poly[:, 0])
    box[1] = min(poly[:, 1])
    box[3] = max(poly[:, 1])
    return box


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[0:2], xyxy[2:4] - xyxy[0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


class Categories(object):
    def __init__(self):
        self.data = {}
        self.unique_id = 0
        self.data2id = {}

    def add(self, new_data):
        if new_data not in self.data2id:
            self.data[self.unique_id] = {
                "supercategory": new_data,
                "id": self.unique_id,
                "name": new_data
            }
            self.data2id[new_data] = self.unique_id
            self.unique_id += 1


class Video(object):
    def __init__(self, name, image_path, image_type, mask_path, mask_type):
        self.frames = {}
        self.objects = {}
        self.frames2id = {}
        self.name = name
        self.path = image_path
        self.file_type = image_type
        self.mask_path = mask_path
        self.mask_type = mask_type

    def add_frame(self, new_frame, frame_id):
        file_name = new_frame + '.' + self.file_type
        im = Image.open(
            os.path.join(self.path, file_name))  # this is not so coasty because the image is not laoded in to memory
        width, height = im.size
        self.frames[frame_id] = {
            "file_name": file_name,
            "video_name": self.name,
            "height": height,
            "width": width,
            "id": frame_id,
        }
        self.frames2id[new_frame] = frame_id

    def add_object(self, obj_index, obj_data):
        self.objects[obj_index] = obj_data

    def get_frames(self):
        return list(self.frames.values())

    def get_image_file_name(self, obj_ind, frame_ind):
        frame = self.objects[obj_ind]['frames'][frame_ind]
        file_name =  frame + '.' + self.file_type
        full_file_name = os.path.join(self.path, file_name)
        return frame, file_name, full_file_name

    def get_mask_file_name(self, obj_ind, frame_ind):
        frame = self.objects[obj_ind]['frames'][frame_ind]
        file_name = frame + '.' + self.mask_type
        full_file_name = os.path.join(self.mask_path, file_name)
        return frame, file_name, full_file_name

    def __len__(self):
        return len(self.frames)


def get_object_bbox_from_image(image_path, object_index):
    pil_image = Image.open(image_path)
    palette = np.array(pil_image.getpalette()).reshape(-1, 3)
    palette_ids = []
    for i in pil_image.getcolors():
        palette_ids.append(i[1])
    if palette_ids[0] != 0:
        palette_ids = [0] + palette_ids
    mask = (np.asarray(pil_image) == object_index).astype(np.uint8)
    if mask.max() != 1:
        raise NameError('requested obejct is not found in mask image')
    mask = mask * 255
    contour = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygon = contour[0][0].squeeze()
    xyxy_bbox = poly_to_box(polygon)
    xywh_bbox = xyxy_to_xywh(xyxy_bbox)
    return xywh_bbox


def collect_images_from_json(args):
    annotation_base_dir = os.path.join(args.data_dir, args.annotation_dir)
    annotation_json = json.load(open(os.path.join(args.data_dir, 'meta.json')))
    images_base_dir = os.path.join(args.data_dir, args.image_dir)
    masks_base_dir = os.path.join(args.data_dir, args.mask_dir)
    image_unique_id = 0
    categories = Categories()
    images = []
    videos = {}
    vid_ind = 0
    for video_name, video_data in annotation_json['videos'].items():

        if args.max_videos_to_parse != -1 and vid_ind >= args.max_videos_to_parse:
            break
        vid_ind += 1
        print(str(vid_ind) + ': collecting inamges of video %s' % video_name)
        video = Video(video_name,
                      os.path.join(images_base_dir, video_name), args.image_type,
                      os.path.join(masks_base_dir, video_name), args.mask_type)
        frames = []
        for obj in video_data['objects']:
            o = video_data['objects'][obj]
            categories.add(o['category'])
            video.add_object(int(obj), video_data['objects'][obj])
            frames.extend(o['frames'])
        frames = sorted(set(frames))
        for frame_ind, frame in enumerate(frames):
            video.add_frame(frame, image_unique_id)
            image_unique_id += 1
        videos[video_name] = video
        images.extend(video.get_frames())

    return videos, images, categories


def get_positive_pair(video, categories):
    obj_ind, obj = random.choice(list(video.objects.items()))  # choose object

    template_index = random.randint(0, len(obj['frames']) - 2)  # choose template frame index
    search_index = random.randint(template_index + 1, len(obj['frames']) - 1)  # choose search frame index
    template_frame, template_frame_file_name, template_mask_full_file_name = video.get_mask_file_name(obj_ind, template_index)
    search_frame, search_frame_file_name, search_mask_full_file_name = video.get_mask_file_name(obj_ind, search_index)
    _, template_frame_file_name, _ = video.get_image_file_name(obj_ind, template_index)
    _, search_frame_file_name, _ = video.get_image_file_name(obj_ind, search_index)
    template_bbox = get_object_bbox_from_image(template_mask_full_file_name, obj_ind)
    search_bbox = get_object_bbox_from_image(search_mask_full_file_name, obj_ind)
    annotation = {
        "segmentation": [],  # NOTE: this value can be retrieved from mask polygon
        "area": 0,
        "iscrowd": 0,
        "image_id": video.frames2id[search_frame],  # image id of search image
        "image_name": search_frame_file_name,  # image name of search image
        "bbox": [float(v) for v in list(search_bbox)],  # bbox of search image
        "template_image_id": video.frames2id[template_frame],  # image id of template image
        "template_image_name": template_frame_file_name,  # image id of template image
        "template_bbox": [float(v) for v in list(template_bbox)],  # bbox of search image
        "category_id": categories.data2id[obj['category']],  # object category id
        "video_name": video.name
    }
    return annotation


def generate_annotations(videos, categories, annotation_size):
    annotations = []
    images_with_annotations = []
    video_iter = itertools.cycle(videos.values())
    annotation_count = 0
    print('generating '+str(annotation_size)+' annotations')
    pbar = tqdm.tqdm(total=annotation_size)
    for video in video_iter:  # go over videos in a cyclic manner
        try:  # following code my throw exception due to faulty annotation
            # such as missing object in annotation image
            annotation = get_positive_pair(video, categories)

            # don't add annotations with images which already have annotation
            if annotation['image_id'] in images_with_annotations:
                continue

            images_with_annotations.append(annotation['image_id'])
            annotation['id'] = annotation_count  # set annotation unique id
            annotations.append(annotation)

            annotation_count += 1
            pbar.update(1)
            if annotation_count >= annotation_size:
                break
        except:
            continue
    pbar.close()
    return annotations


def split_videos_into_train_valid_sets(args, videos, images):
    valid_set_video_num = round(args.valid_set_video_percent * len(videos))
    videos_names = list(videos.keys())
    random.shuffle(videos_names)
    valid_videos_names = videos_names[0:valid_set_video_num]
    train_videos_names = videos_names[valid_set_video_num:]

    train_videos = {name: video for (name, video) in videos.items() if name in train_videos_names}
    valid_videos = {name: video for (name, video) in videos.items() if name in valid_videos_names}

    train_images = [image for image in images if image['video_name'] in train_videos_names]
    valid_images = [image for image in images if image['video_name'] in valid_videos_names]

    return train_videos, train_images, valid_videos, valid_images


def remove_images_with_no_annotarions(images, annotations):
    anno_images_ids = {anno['image_id'] for anno in annotations}
    reduced_images = [image for image in images if image['id'] in anno_images_ids]
    return reduced_images


if __name__ == '__main__':
    args = parse_args()
    print('collect videos and images')
    videos, images, categories = collect_images_from_json(args)
    train_videos, train_images, valid_videos, valid_images = split_videos_into_train_valid_sets(args, videos, images)
    print('collect train annotation')
    train_annotations = generate_annotations(train_videos, categories, args.train_set_size)
    print('collect valid annotation')
    valid_annotations = generate_annotations(valid_videos, categories, args.valid_set_size)

    # the following is needed because COCO tools does not support images with no annotations
    train_images = remove_images_with_no_annotarions(train_images, train_annotations)
    valid_images = remove_images_with_no_annotarions(valid_images, valid_annotations)

    train_coco_format = {
        "info": {},
        "licenses": [],
        "images": train_images,
        "annotations": train_annotations,
        "categories": list(categories.data.values())
    }

    valid_coco_format = {
        "info": {},
        "licenses": [],
        "images": valid_images,
        "annotations": valid_annotations,
        "categories": list(categories.data.values())
    }

    with open(os.path.join(args.out_dir, args.train_json_name), 'w') as outfile:
        json.dump(train_coco_format, outfile)

    with open(os.path.join(args.out_dir, args.valid_json_name), 'w') as outfile:
        json.dump(valid_coco_format, outfile)

    print('bye bye')
