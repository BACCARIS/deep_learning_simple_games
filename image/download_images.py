#!/usr/bin/python

import os
from sys import stdin

images_path = "urls.txt"


links = open(images_path, 'r')
lines = links.readlines()
for line in lines : 
    cmd = 'wget ' + line 
    os.system(cmd)
    print(cmd, "\n")

print("!!DONE!!")
