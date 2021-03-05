import random

import requests
from getpass import getpass
import json
import tempfile
import zipfile
import io
import os
from glob import glob

user = "peb"
pw = '******'

expected_labels_i2name = {1: 'title_article', 2: 'authors_article', 3: 'refs_article', 4: 'toc'}
expected_labels_name2i = {'title_article': 1, 'authors_article':2, 'refs_article': 3, 'toc': 4}

r = requests.get('http://localhost:8080/api/v1/projects', auth=("peb", pw))
if r.status_code != 200:
    print("problem with GET")
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
        print(f"   task={task_name}  count={image_count}  status={task_status}")
        if (task_status != "completed"):
            continue
        print(f"   task={task_name}  is completed, get zipfile of json...")
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
                    




