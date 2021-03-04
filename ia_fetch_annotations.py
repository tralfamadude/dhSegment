import random

import requests
from getpass import getpass
import json
import tempfile
import zipfile
import io
import os
from glob import glob
from pycocotools import coco as coco_loader

user = "peb"
pw = '******'

r = requests.get('http://localhost:8080/api/v1/projects', auth=("peb", pw))
if r.status_code != 200:
    print("problem with GET")
r.encoding = 'utf-8'
project_dict = r.json()
project_list = project_dict["results"]
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
        r_annotation2 = requests.get(annotation_url, auth=(user, pw))
        if r_annotation2.status_code != 201:
            print("error: r_annotation2 status is not 201")
        r_annotation3 = requests.get(annotation_url + "&action=download", auth=(user, pw))
        if r_annotation3.status_code != 200:
            print("error: r_annotation3 status is not 200")
        with tempfile.TemporaryDirectory() as tmp_dir:
            zipfile.ZipFile(io.BytesIO(r_annotation3.content)).extractall(tmp_dir)
            jsons = glob(os.path.join(tmp_dir, '**', '*.json'), recursive=True)
            for json in jsons:
                coco = coco_loader.COCO(json)
                print(f"FOUND:  {coco.getAnnIds()}")



