#!/usr/bin/python
#
# Author: Jim Fan
# Eona studio (c) 2015
#
# This script adds __NVCC__ and __CUDACC__ macros to .cproject

from sys import argv, stderr
import os

if not len(argv) >= 2:
    print >> stderr, "Usage: python {} <eclipse_project_dir (cmake build dir)>\n".format(argv[0])
    exit()

projpath = lambda f : os.path.join(argv[1], f)

def write_internal(f, l):
    print >> f, l.rstrip()

#================= correct .cproject file
os.rename(projpath(".cproject"), projpath(".cproject_bak"))
oldCproject = open(projpath(".cproject_bak"), 'r')
newCproject = open(projpath(".cproject"), 'w')

go = lambda : oldCproject.readline()
write = lambda l : write_internal(newCproject, l)

l = go()
write(l)
addOnce = False
while l:
    l = go()
    if "[Source" in l:
        if not addOnce:
            addOnce = True
            write('<pathentry kind="mac" name="__NVCC__" path="" value="1"/>')
            write('<pathentry kind="mac" name="__CUDACC__" path="" value="1"/>')
    write(l)

newCproject.close()
print "New .cproject written, old one backed up as .cproject_bak"
