#!/usr/bin/python
#
# Author: Jim Fan
# Eona studio (c) 2015
#
# Run inside the build dir
# ./lmncheck.py check <testsuite>
# For example 
# $> ./lmncheck.py vecmat_lstm
# will run test/vecmat_lstm_test executable and compare the result with 
# NOTE: automatically append _test to the executable
# ../test_out/vecmat_lstm.out file
#
# ./lmncheck.py gen <testsuite>
# will run test/vecmat_lstm_test and store the output in
# ../test_out/<testsuite>.out for future comparison

from sys import argv, stderr, exit
import os

assert len(argv) == 3

exepath = os.path.join('test', argv[2] + '_test')
exeoutpath = os.path.join('../test_out/', argv[2]+'.tmp')
stdpath = os.path.join('../test_out/', argv[2] + '.out')

if argv[1] == 'check':
    print 'Running', exepath, '...\n'
    # run the test first, store the output
    os.system(exepath + ' > ' + exeoutpath)
    # check against a stored standard
    for tmpline, stdline in zip(open(exeoutpath, 'r'), open(stdpath, 'r')):
        if tmpline.startswith('[==========]') or \
            tmpline.startswith('[----------]') or \
            tmpline.startswith('[ RUN      ]') or \
            tmpline.startswith('[       OK ]') or \
            tmpline.startswith('[  PASSED  ]'):
            continue

        if tmpline != stdline:
            print '??????  PYTHON LAMINAR CHECK FAILURE ???????\n'
            print exepath, ':\n', tmpline
            print '------ differs from ------\n'
            print stdpath, ':\n', stdline.rstrip()
            exit(1)

    print 'PYTHON LAMINAR CHECK SUCCESS'
    
# generate a standard 
elif argv[1] == 'gen':
    print 'Running', exepath, '...\n'
    os.system(exepath + ' > ' + stdpath)
    print 'Standard generated to', stdpath
