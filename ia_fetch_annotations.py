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
