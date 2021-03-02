import requests
from getpass import getpass
import json
import zipfile

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
        annotation_url = task_url + "/annotations?format=COCO%201.0"
        r_annotation = requests.get(annotation_url, auth=(user, pw))
        if r_annotation.status_code != 202:
          print("error: r_annotation status is not 202")
        r_annotation2 = requests.get(annotation_url, auth=(user, pw))
        if r_annotation2.status_code != 201:
          print("error: r_annotation2 status is not 201")
        r_annotation3 = requests.get(annotation_url + "&action=download", auth=(user, pw))
