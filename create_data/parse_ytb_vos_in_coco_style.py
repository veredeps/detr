# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
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
from os.path import join
import random
import matplotlib.pyplot as plt
import time

random.seed(123456)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('--outdir', default='./', type=str,
                        help="output dir for json files")
    parser.add_argument('--datadir', default='./', type=str,
                        help="data dir for annotations to be converted")

    return parser.parse_args()


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
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]
    return boxes_from_polys


class Instance(object):
    instID     = 0
    pixelCount = 0

    def __init__(self, imgNp, instID):
        if (instID ==0 ):
            return
        self.instID     = int(instID)
        self.pixelCount = int(self.getInstancePixels(imgNp, instID))

    def getInstancePixels(self, imgNp, instLabel):
        return (imgNp == instLabel).sum()

    def toDict(self):
        buildDict = {}
        buildDict["instID"]     = self.instID
        buildDict["pixelCount"] = self.pixelCount
        return buildDict

    def __str__(self):
        return "("+str(self.instID)+")"


def  update_res_train_cat(res_train, categories):
    for ind, cat in enumerate(categories):
        cat_dict = {"supercategory": cat, "id": ind, "name": cat}
        res_train['categories'].append(cat_dict)


def extract_search_data(res):

    vid1 = res['images'][00000]['video_name']
    ann_dir = res['images'][00000]['anno_path'].split(vid1)[0]
    for ann_ind , ann in enumerate(res['annotations']):
        template_fr = ann['template_image_name'].split('.')[0]
        if template_fr in ann['fr_in_vid']:
            ind = ann['fr_in_vid'].index(template_fr)
        else:
            print('check why wrong fr_in_vid', ann['video_name'], 'template_fr:', template_fr, 'obId:',ann['objId'],'id:',ann['id'] )
            #log_file.write('check why wrong fr_in_vid' + ann['video_name']+ 'template_fr:'+ template_fr+ 'obId:'+ann['objId']+'id:'+ ann['id'] )

        if ((ind+1) < len(ann['fr_in_vid']) ):
            search_ind = random.randint(ind+1,len(ann['fr_in_vid']) - 1)
        else:
            search_ind = ind
            print('template ind is last ind in fr_in_vid check why ! set search to template ind ', ann['video_name'], 'template_fr:', template_fr, 'obId:',ann['objId'],'id:',ann['id'])
            #log_file.write('template ind is last ind in fr_in_vid  check why !' + ann['video_name']+ 'template_fr:'+ template_fr+ 'obId:'+ann['objId']+'id:'+ ann['id'] )

        search_fr = ann['fr_in_vid'][search_ind]
        if search_ind > (len(ann['frames_unique_id'])-1):
            print('frames_unique_id disagree with fr_in_vid', ann['video_name'], 'template_fr:', template_fr, 'obId:',ann['objId'],'id:',ann['id'])
            #log_file.write('frames_unique_id disagree with fr_in_vid'+ ann['video_name']+ 'template_fr:'+ template_fr+ 'obId:'+ann['objId']+'id:'+ ann['id'] )


        file_name = join(ann['video_name'], search_fr)
        fullname = os.path.join(ann_dir, file_name + '.png')
        img = cv2.imread(fullname, 0)
        mask = (img == ann['objId']).astype(np.uint8)


        if np.sum(mask) == 0:  #no annotation
            search_fr_found = ''
            for i, search_fr in enumerate(ann['fr_in_vid'][ind+1:]):
                file_name = join(ann['video_name'], search_fr)
                fullname = os.path.join(ann_dir, file_name + '.png')
                img = cv2.imread(fullname, 0)
                mask = (img == ann['objId']).astype(np.uint8)
                if np.sum(mask) != 0:
                    search_ind = ann['fr_in_vid'].index(search_fr)
                    search_fr_found = search_fr
                    break
            search_fr = search_fr_found

        if len(search_fr) == 0:
            print('no anno  for this objId :', ann['objId'], ' after  frame  :',template_fr, 'in video',  ann['video_name'])
            print('search  fr is set to template fr')
            #log_file.write('no anno  for this  objId search_fr is set to  template_fr' + ann['video_name'] + 'template_fr:' + template_fr + 'obId:' + ann['objId'].__str__() + 'id:' + ann['id'].__str__())
            #del(res['annotations'][ann_ind])
            ann['image_id'] = ann['template_image_id']
            ann['image_name'] = ann['template_image_name']
            ann['bbox'] = ann['template_bbox']

        else:
            contour, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            polygons = [c.reshape(-1).tolist() for c in contour]
            obj_contours = [p for p in polygons if len(p) > 4]

            if len(obj_contours) == 0 or len(obj_contours[0]) == 0:
                print('contour too small, search  fr is set to template fr   ')
                #log_file.write('contor too small search_fr is set to  template_fr' + ann['video_name'] + 'template_fr:' + template_fr + 'obId:' + ann['objId'].__str__() + 'id:' + ann['id'].__str__())
                ann['image_id'] = ann['template_image_id']
                ann['image_name'] = ann['template_image_name']
                ann['bbox'] = ann['template_bbox']


            else:
                search_image_unique_id = ann['frames_unique_id'][search_ind]
                ann['image_id'] = search_image_unique_id
                ann['image_name'] = search_fr + '.jpg'
                ann['bbox'] = xyxy_to_xywh(polys_to_boxes([obj_contours])).tolist()[0]   #search bbox


        #just for debug
        #dispay bbox on image
                """
                from PIL import Image
                from PIL import ImageDraw
        
                fig, ax  = plt.subplots(nrows=1, ncols = 2)
        
                pil_im = Image.open(fullname).convert("RGBA")
                draw = ImageDraw.Draw(pil_im)
                #xmin, ymin, xmax, ymax = polys_to_boxes([obj_contours]).tolist()[0]
                xmin, ymin, xmax, ymax = ann['bbox']
                draw.rectangle(((xmin, ymin), (xmax+xmin, ymax+ymin)),  outline = 'yellow', width = 4)
                ax[0].imshow(pil_im)
                #pil_im.show()
                
                full_name_template=os.path.join(ann_dir, ann['video_name']+'/'+ ann['template_image_name'].split('.')[0]+'.png')
                pil_im = Image.open(full_name_template).convert("RGBA")
                draw = ImageDraw.Draw(pil_im)
                #xmin, ymin, xmax, ymax = polys_to_boxes([obj_contours]).tolist()[0]
                xmin, ymin, xmax, ymax = ann['template_bbox']
                draw.rectangle(((xmin, ymin), (xmax+xmin, ymax+ymin)),  outline = 'yellow', width = 4)
                ax[1].imshow(pil_im)
        
                plt.close(fig)
                
                """
    return res


def clear_frames_with_no_anno(frames, ann_dir, json_ann, video):
    frames_with_anno = frames.copy()
    for frame in frames:
        file_name = join(video, frame)
        fullname = os.path.join(ann_dir, file_name + '.png')
        img = cv2.imread(fullname, 0)
        if np.sum(img) == 0:  # skip and remove images with no annotation
            frames_with_anno.remove(frame)
            print(video, ' ', frame, 'frame with no anno ')
            log_file.write(video + ' ' + frame + 'frame with no anno ')
            for i, ob in enumerate(json_ann['videos'][video]['objects']):
                if frame in json_ann['videos'][video]['objects'][ob]['frames']:
                    json_ann['videos'][video]['objects'][ob]['frames'].remove(frame)  # remove frames with no annotations from list to be cohetent with frames_unique_id size
            continue
    return frames_with_anno, json_ann


def convert_ytb_vos(data_dir, out_dir, log_file):

    res_train = {'info': {},  'licences': [], 'images': [], 'annotations':[], 'categories':[]}
    res_val = {'info': {}, 'licences': [], 'images': [], 'annotations': [], 'categories':[]}

    #res_train['categories'] = [{"supercategory": "track", "id":0, "name":"track"}]
    #res_val['categories'] = [{"supercategory": "track", "id": 0, "name": "track"}]

    im_unique_id = -1
    obj_unique_id = 0

    sets = ['train']
    #ann_dirs = ['train/Annotations/']
    ann_dirs = ['/home/n6ve/you_tube_vos/ytb_vos/train/Annotations']
    im_dir = ['/home/n6ve/you_tube_vos/ytb_vos/train/JPEGImages']
    #json_name = 'instances_%s.json'
    json_name = '%s.json'
    num_obj = 0
    num_ann = 0
    categories = []
    t1 = time.time()

    for data_set, ann_dir in zip(sets, ann_dirs):
        print('Starting %s' % data_set)
        ann_dict = {}
        ann_dir = os.path.join(data_dir, ann_dir)
        json_ann = json.load(open(os.path.join(ann_dir, '../meta.json')))

        #json_ann = json.load(open(os.path.join(ann_dir, 'meta.json')))
        for vid, video in enumerate(json_ann['videos']):
            is_train = random.uniform(0, 1) > 0.14
            #if video  == '01baa5a4e1':    #'011ac0a06f'
            #   print('wait')
            #if vid > 5:  #debug only
            #    continue
            v = json_ann['videos'][video]
            frames = []
            for obj in v['objects']:
                o = v['objects'][obj]
                frames.extend(o['frames'])
            frames = sorted(set(frames))
            #fr_delta = int(frames[1]) - int(frames[0])
            annotations = []
            instanceIds = []
            json_ann['videos'][video].update({'frames_unique_id': []})

            frames_with_anno, json_ann = clear_frames_with_no_anno(frames,  ann_dir ,json_ann, video)

            for frame in frames_with_anno:
                file_name = join(video, frame)
                fullname = os.path.join(ann_dir, file_name + '.png')
                img = cv2.imread(fullname, 0)

                h, w = img.shape[:2]

                image_with_idx = plt.imread(fullname)  # 4 entry reflects the index that from it e can conclue the category  the fourth entry is not read wiht cv2 !!

                #debug only
                #from PIL import Image
                #pil_im = Image.open(fullname).convert("RGBA")
                #pil_im.show()
                # end debug only

                image_path = os.path.join(im_dir[0], file_name + '.jpg')
                anno_path = os.path.join(ann_dirs[0], file_name + '.png')

                video_name, name  = file_name.split('/')[0:2]

                im_unique_id = im_unique_id + 1
                image = {"file_name":name,
                         "video_name":video_name,
                         "width": w,
                         "height":h,
                         "id":im_unique_id,
                         "images_path":image_path,
                         "anno_path": anno_path}

                json_ann['videos'][video]['frames_unique_id'].append(im_unique_id)
                res_train['images'].append(image) if is_train else res_val['images'].append(image)
                objects = dict()
                for instanceId in np.unique(img):
                    if instanceId == 0:
                        continue

                    instanceObj = Instance(img, instanceId)
                    instanceObj_dict = instanceObj.toDict()
                    mask = (img == instanceId).astype(np.uint8)
                    ind_non_zero = np.argwhere(np.asarray(mask) > 0)

                    ind1 = ind_non_zero[int(ind_non_zero.shape[0]/2)][0]
                    ind2 = ind_non_zero[int(ind_non_zero.shape[0]/2)][1]

                    idx_of_obj = image_with_idx[ind1, ind2,3]
                    idx_oob  = int(idx_of_obj.item()).__str__()

                    if idx_oob in json_ann['videos'][video]['objects'].keys():
                        instanceObj_dict['category'] = json_ann['videos'][video]['objects'][idx_oob]['category']
                        instanceObj_dict['fr_in_vid'] = json_ann['videos'][video]['objects'][idx_oob]['frames']
                    else:
                        continue

                    categories.append(instanceObj_dict['category'])


                    #_, contour, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, #cv2.CHAIN_APPROX_NONE)
                    contour, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    polygons = [c.reshape(-1).tolist() for c in contour]
                    instanceObj_dict['contours'] = [p for p in polygons if len(p) > 4]
                    if len(instanceObj_dict['contours']) and instanceObj_dict['pixelCount'] > 1000:
                        objects[instanceId] = instanceObj_dict
                    # else:
                    #     cv2.imshow("disappear?", mask)
                    #     cv2.waitKey(0)

                #objId = 0 #init value
                #if bool(objects):  #check if there  are objects
                #    objId = random.choice(list(objects.keys()))    #take only one object in a frame

                for objId in objects:
                    if len(objects[objId]) == 0:
                        continue
                    template_valid_frame = frame in objects[objId]['fr_in_vid'] and frame != objects[objId]['fr_in_vid'][-1]
                    if objId > 0  and template_valid_frame:
                    # if objId > 0 and frame != frames_with_anno[-1] and frame != frames_with_anno[-2] and frame in objects[objId]['fr_in_vid']:
                        obj = objects[objId]
                        len_p = [len(p) for p in obj['contours']]
                        if min(len_p) <= 4:
                            print('Warning: invalid contours.')
                            continue  # skip non-instance categories

                        ann = dict()
                        #ann['h'] = h
                        #ann['w'] = w
                        ann['segmentation'] = []  # mask
                        ann['area'] = []
                        ann['iscrowd'] = 0
                        ann['image_id'] = ''
                        ann['image_name'] = ''
                        ann['bbox'] = ''
                        ann['template_image_id'] = im_unique_id
                        ann['template_image_name'] = image_path.split('/')[-1]
                        ann['template_bbox'] = xyxy_to_xywh(polys_to_boxes([obj['contours']])).tolist()[0]
                        ann['category_id'] = obj['category']
                        ann['video_name'] = video
                        ann['id'] = obj_unique_id


                        ann['fr_in_vid'] = obj['fr_in_vid']  # frames where obj exists
                        ann['frames_unique_id'] =[]

                        ann['objId'] = int(objId)

                        obj_unique_id = obj_unique_id + 1

                        #ann['template_image_path'] = image_path  # video name and frame name
                        # ann['template_anno_path'] = anno_path  # video name and frame name
                        # ann['template_area'] = obj['pixelCount']
                        #ann["fr_delta"] = fr_delta


                        res_train['annotations'].append(ann) if is_train else res_val['annotations'].append(ann)

                        annotations.append(ann)
                        instanceIds.append(objId)
                        num_ann += 1

            instanceIds = sorted(set(instanceIds))
            num_obj += len(instanceIds)
            video_ann = {str(iId): [] for iId in instanceIds}
            for ann in annotations:
                video_ann[str(ann['objId'])].append(ann)

            ann_dict[video] = video_ann
            if vid % 50 == 0 and vid != 0:
                print("process: %d video" % (vid+1))

        for ind, _ in enumerate(res_train['annotations']):
            video_name = res_train['annotations'][ind]['video_name']
            #res_train['annotations'][ind].update({'fr_unique_id':[]})
            res_train['annotations'][ind]['frames_unique_id'] = json_ann['videos'][video_name]['frames_unique_id']

        for ind, _ in enumerate(res_val['annotations']):
            video_name = res_val['annotations'][ind]['video_name']
            res_val['annotations'][ind].update({'fr_unique_id':[]})
            res_val['annotations'][ind]['frames_unique_id'] = json_ann['videos'][video_name]['frames_unique_id']

        print("Num Videos: %d" % len(ann_dict))
        print("Num Objects: %d" % num_obj)
        print("Num Annotations: %d" % num_ann)

        update_res_train_cat(res_train, set(categories))
        update_res_train_cat(res_val, set(categories))

        if res_val['categories'] !=  res_train['categories']:
            print('problem: cat in val are likecat in train !')

        t11 = time.time()
        print('elapsed time =  ' + str(t11 - t1))

        extract_search_data( res_train)
        extract_search_data( res_val)

        with open(os.path.join(out_dir, json_name % 'train'), 'w') as outfile:
            json.dump(res_train, outfile)

        with open(os.path.join(out_dir, json_name % 'val'), 'w') as outfile:
            json.dump(res_val, outfile)

        t2 = time.time()
        print('elapsed time =  ' + str(t2-t1))

        #items = list(ann_dict.items())
        #train_dict = dict(items[:3000])
        #val_dict = dict(items[3000:])
        #with open(os.path.join(out_dir, json_name % 'train'), 'w') as outfile:
        #    json.dump(train_dict, outfile)
        #with open(os.path.join(out_dir, json_name % 'val'), 'w') as outfile:
        #    json.dump(val_dict, outfile)





if __name__ == '__main__':
    log_file = open("log.txt", "w")
    args = parse_args()
    convert_ytb_vos(args.datadir, args.outdir, log_file)
    log_file.close()