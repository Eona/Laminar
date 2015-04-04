#!/usr/bin/python
#
# Author: Jim Fan
# Eona studio (c) 2015
#
# This script adds __NVCC__ and __CUDACC__ macros to .cproject
# adds nvcc.errorParser to .project

from sys import argv, stderr
import os

if not len(argv) >= 2:
    print >> stderr, "Usage: python {} <eclipse_project_dir (cmake build dir)>\n".format(argv[0])
    exit()

projpath = lambda f : os.path.join(argv[1], f)

def write_internal(f, l):
    print >> f, l.rstrip()

#================= correct .project file
os.rename(projpath(".project"), projpath(".project_bak"))
oldProject = open(projpath(".project_bak"), 'r')
newProject = open(projpath(".project"), 'w')

go = lambda : oldProject.readline()
write = lambda l : write_internal(newProject, l)

l = go()
write(l)
addOnce = False
while l:
    l = go()
    write(l)
    if "<key>org.eclipse.cdt.core.errorOutputParser</key>" in l:
        l = go()
        parservalues = l.strip()[len("<value>"):-len("</value>")]
        parservalues = 'nvcc.errorParser;' + parservalues
        write("<value>" + parservalues + "</value>")

newProject.close()
print "New .project written, old one backed up as .project_bak"

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
