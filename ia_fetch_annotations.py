import random

import requests
from getpass import getpass
import json
import tempfile
import zipfile
import io
import os
import shutil
import sys
import pathlib
from glob import glob
import argparse
import pandas as pd
import numpy as np
from imageio import imsave, imread
import ia_image_annotation as annote

"""
Fetch finished annotations from CVAT via api and create color mask files based on original images, 
saving the result organized by cvat task id.  

Expected inputs, conditions: 
  /home/$USER/cvat_area/ with subdir $TASK_ID1/*.png, $TASK_ID2/*.png ... 
  cvat server running on localhost:8080
  There is a finished, unprocessed task in CVAT
  
Expected outputs, conditions:
  /home/$USER/cvat_labeled_images/$TASK_ID1/classes.txt     color to label map
  /home/$USER/cvat_labeled_images/$TASK_ID1/ground.csv      page type for training post-model
  /home/$USER/cvat_labeled_images/$TASK_ID1/images/*.png    original images copied from cvat_area
  /home/$USER/cvat_labeled_images/$TASK_ID1/labels/*.png    color masks

The intention is that the destination dir is ready to be organized into training and eval; we don't attempt
to do the train/eval split here because training experiments need flexibility to try various mixtures. Instead,
the output of this script is raw material (images labeled with color masks) for composing a training and eval
dataset.  The cvat_labeled_images contents can be considered golden, whereas there can be multiple training/eval sets 
composed from these golden files. 

If the destination dir $TASK_ID already exists, the processing will be skipped, this is how we can 
tell if that step was already done. 

To re-do this processing, rename or remove the $TASK_ID/ under cvat_labeled_images, taking care to 
copy the cvat_labeled_images/$TASK_ID1/images/* back to the /home/$USER/cvat_area//$TASK_ID1 ; the annotations 
should still be stored in the postgres db used by cvat.

During processing of annotations the corresponding images under cvat_area/$TASK_ID are moved from 
 cvat_labeled_images/; unannotated images will get blank/black masks. The cvat_area/$TASK_ID/*.png or .jpg will be 
 moved to destination to save space.   
  
"""

user = os.environ.get("CVAT_USER")
pw = os.environ.get("CVAT_PW")  # ToDo: get this from a file
if (user is None) or (pw is None):
    print("These env vars must be defined:  CVAT_USER, CVAT_PW")

FLAGS = None
# init the parser
parser = argparse.ArgumentParser()

#  images base dir, subdirs are journal or issue ids
parser.add_argument(
    '--image_dir', '-i',
    type=str,
    default='/home/peb/cvat_area',
    help='base path to input images in subdirs names as cvat task ids'
)
# processed images and labels, organized by subdirs aas cvat task ids
parser.add_argument(
    '--dest_dir', '-d',
    type=str,
    default='/home/peb/cvat_post',
    help='base path to save under subdirs as cvat task ids'
)
# path annotation_names.csv holding official labels so we do not need to hardwire label IDs
parser.add_argument(
    '--master_labels', '-m',
    type=str,
    default='labeling/annotation_names.csv',
    help='path annotation_names.csv holding official labels'
)
FLAGS, unparsed = parser.parse_known_args()
if len(unparsed) > 0:
    print(f"  Unknown args: {unparsed}")
    parser.print_usage()
    sys.exit(1)
image_dir = FLAGS.image_dir
dest_dir = FLAGS.dest_dir
master_label_file = FLAGS.master_labels

# validate
if not os.path.exists(image_dir):
    print(f"does not exist: {image_dir}")
    sys.exit(2)
if not os.path.exists(dest_dir):
    print(f"does not exist: {dest_dir}")
    sys.exit(2)

#
#         label annotation, int id and color correspondence
#
# we keep a master list of annotation types/labels in annotation_names.csv so that if we append a new one, the
# colors of existing color masks are still correct. We assign colors from label_colors in order of appearance.
# dhSegment model has a maximum of 7 labels.
try:
    df = pd.read_csv(master_label_file, header=None)
except FileNotFoundError:
    print("Could not find required file {master_label_file}")
    sys.exit(5)
df.columns = ['annotation_type']
annotation_type_series = df['annotation_type']
annotation_type_list = annotation_type_series.tolist()
len_label_list = len(annotation_type_list)

# Prepare dicts of labeling/annotation_names.csv; label id starts at 1
# expected_labels_i2name = {1: 'title_article', 2: 'authors_article', 3: 'refs_article', 4: 'toc'}
# expected_labels_name2i = {'title_article': 1, 'authors_article':2, 'refs_article': 3, 'toc': 4}
#  These are used to validate labels from CVAT to avoid inconsistencies.
expected_labels_i2name = {}
expected_labels_name2i = {}
for i in range(1, len_label_list+1):
    expected_labels_i2name[i] = annotation_type_list[i-1]
    expected_labels_name2i[annotation_type_list[i-1]] = i
print(f" expected_labels_i2name={expected_labels_i2name}") # DEBUG
print(f" expected_labels_name2i={expected_labels_name2i}") # DEBUG

def create_blank_mask(input_image_filename, output_image_filename):
    # load image to get dimensions
    img = imread(input_image_filename)
    h = img.shape[0]
    w = img.shape[1]
    # create blank image
    out_img = np.zeros((h, w, 3))
    # save
    imsave(output_image_filename, out_img.astype(np.uint8))

def generate_classes_file(dest: str):
    """
    Generate and save a file that tells dhSegment which color corresponds to which label.
    :param dest: where to save classes.txt
    :return:
    """
    #  We want to map the annotation types to colors
    out_file = open(f"{dest}/classes.txt", "w")
    # write out first line which represents no annotation at all for black
    out_file.write("0 0 0\n")  # black
    i = 0
    while i < len_label_list:
        annotation_type = annotation_type_list[i]
        rgb = label_val_to_color(i+1)
        out_file.write(f"{rgb[0]} {rgb[1]} {rgb[2]}\n")
        i = i + 1
    out_file.close()


def label_val_to_color(labelv):
    """
    Map a label number to a color.
    :param labelv: label value, 0-Nclasses inclusive where 0 is background.
    :return: [r,g,b]
    """
    # label colors
    label_colors = [[0, 0, 0],  # background is black (not a label)
                    [0, 255, 0],
                    [0, 0, 255],
                    [255, 0, 0],
                    [1, 255, 254],
                    [255, 166, 254],
                    [255, 219, 102],
                    [0, 100, 1]]
    return label_colors[labelv]

def process_annotations(task_name: str, image_list: list, annotation_list: list):
    """
    :param task_name: cvat task name, same as issue ID
    :param image_list: example item {"id": 1, "width": 1080, "height": 1640, "file_name": "somthing_0005.png", ...}}
      not showing here entries that are unused.
    :param annotation_list: item example {"id": 1, "image_id": 2, "category_id": 1, "bbox": [539.8, 384.9, 377.0, 48.9]
        not showing here entries that are unused.
    :return: None
    """
    dest_dir_images = dest_dir + "/" + task_name + "/" + "images"
    # check if already processed
    if os.path.exists(dest_dir_images):
        # already processed
        print(f" task {task_name} already processed, skipping")
        return None
    apath = pathlib.Path(dest_dir_images)
    apath.mkdir(parents=True, exist_ok=True)

    dest_dir_labels = dest_dir + "/" + task_name + "/" + "labels"
    apath = pathlib.Path(dest_dir_labels)
    apath.mkdir(parents=True, exist_ok=True)

    # create a dict image_id to [ ImageAnnotation(file_name, id, width, height) ]
    #  to which we will later add annotation
    image_id2details = {}
    for image_dict in image_list:  #
        id = image_dict["id"]
        image_id2details[id] = annote.ImageAnnotation(image_dict["file_name"], id, image_dict["width"],
                                                      image_dict["height"])
    # visit each annotation (a dict), collecting under the ImageAnnotation in image_id2details.
    #  coco annotation file also lists un-annotated files.
    for annot_dict in annotation_list:
        image_id = annot_dict["image_id"]
        category_id = annot_dict["category_id"]
        bbox = annot_dict["bbox"]
        image_annotation = image_id2details[image_id]
        image_annotation.add_annotation(category_id, bbox)
    #
    # process each annotated image.
    for image_annotation in image_id2details.values():
        w = image_annotation.width
        h = image_annotation.height
        src_filename = image_annotation.image_name
        if src_filename.endswith(".jpg"):
            # mask must be png
            dest_filename = src_filename.replace(".jpg", ".png")
        else:
            dest_filename = src_filename
        image_annotation_list = image_annotation.annotations
        # make the mask, color the annotation rectangles
        mask_img = np.zeros((w, h, 3), np.uint8)
        print(f"  create mask h={h}  w={w}  for {src_filename} id={image_annotation.image_id}")  # DEBUG
        for one_image_annotation in image_annotation_list:
            #   [category_id, [x, y, w, h]]
            label_id = one_image_annotation[0]
            bbox = one_image_annotation[1]
            print(f" bbox from coco {bbox}")  # DEBUG
            x_bbox = round(bbox[0])
            y_bbox = round(bbox[1])
            w_bbox = round(bbox[2])
            h_bbox = round(bbox[3])
            print(f" annotation abs location is [{x_bbox},{y_bbox},{x_bbox+w_bbox},{y_bbox+h_bbox}] ")  # DEBUG
            for x in range(x_bbox, x_bbox+w_bbox):
                for y in range(y_bbox, y_bbox+h_bbox):
                    c = label_val_to_color(label_id)
                    mask_img[x, y, 0] = c[0]
                    mask_img[x, y, 1] = c[1]
                    mask_img[x, y, 2] = c[2]
        #     write the color mask file
        #  but first re-orient for imsave()
        msg_ing_out = np.fliplr(np.rot90(mask_img, 3))
        imsave(dest_dir_labels + "/" + dest_filename, msg_ing_out)
        # mv image to dest/images to mark it as done
        shutil.move(image_dir + "/" + task_name + "/" + src_filename, dest_dir_images + "/" + src_filename)
    #      process unannotated images
    # Images that have no annotations need a blank-black mask. We handle this by moving images with annotations
    #   and any images that remain in the task directory need this treatment.
    #unprocessed_image_filename = os.listdir(image_dir)
    #for image_file in unprocessed_image_filename:
    #    if not (image_file.endswith(".png") or image_file.endswith(".jpg")):
    #        continue
    #    # make a blank-black image mask in label dir
    #    create_blank_mask(image_dir + "/" + image_file, dest_dir_labels + "/" + image_file)
    #    # mv to mark as done
    #    shutil.move(image_dir + "/" + task_name + "/" + image_file, dest_dir_images + "/" + image_file)




# CVAT api
r = requests.get('http://localhost:8080/api/v1/projects', auth=(user, pw))
if r.status_code != 200:
    print("problem with GET, HTTP code={r.status_code}")
    sys.exit(1)
r.encoding = 'utf-8'
project_dict = r.json()
project_list = project_dict["results"]
annotation_json = ""
for project in project_list:
    project_name = project["name"]
    task_list = project["tasks"]
    for task in task_list:
        task_url = task["url"]
        image_count = task["size"]
        task_name = task["name"]
        task_status = task["status"]  #  "completed" is what we want
        if (task_status != "completed"):
            print(f"   task={task_name}  count={image_count}  status={task_status}, not finished annotating")
            continue
        print(f"   task={task_name}  count={image_count}  status={task_status}, ready to be processed")
        annotation_url = task_url + "/annotations?format=COCO%201.0"
        r_annotation = requests.get(annotation_url, auth=(user, pw))
        if r_annotation.status_code != 202:
            print("error: r_annotation status is not 202")
            continue
        r_annotation2 = requests.get(annotation_url, auth=(user, pw))
        if r_annotation2.status_code != 201:
            print("error: r_annotation2 status is not 201")
            continue
        r_annotation3 = requests.get(annotation_url + "&action=download", auth=(user, pw))
        if r_annotation3.status_code != 200:
            print("error: r_annotation3 status is not 200")
            continue
        with tempfile.TemporaryDirectory() as tmp_dir:
            zipfile.ZipFile(io.BytesIO(r_annotation3.content)).extractall(tmp_dir)
            jsons = glob(os.path.join(tmp_dir, '**', '*.json'), recursive=True)
            for json_file in jsons:
                with open(json_file, 'r') as f:
                    annotation_json = json.load(f)
                    label_list = annotation_json["categories"]
                    #
                    # validate labels (categories in the coco format)
                    found_labels_i2name = {}
                    found_labels_name2i = {}
                    for item in label_list:
                        found_labels_i2name[item["id"]] = item["name"]
                        found_labels_name2i[item["name"]] = item["id"]
                    if expected_labels_i2name != found_labels_i2name:
                        print(f" task {task_name} has unexpected category/labels, skipping")
                        continue
                    if expected_labels_name2i != found_labels_name2i:
                        print(f" task {task_name} has unexpected category/labels, skipping")
                        continue
                    image_list = annotation_json["images"]
                    annotation_list = annotation_json["annotations"]
                    #   Process
                    process_annotations(task_name, image_list, annotation_list)




