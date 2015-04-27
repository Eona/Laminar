from sys import argv
import os
from collections import Counter
import struct

# must end with '.corpus
fname = argv[1][:-len('.corpus')]

# map special chars to int index
SpecialCharDict = {
    ' ' : 52,
    '!' : 53,
    '"' : 54,
    ',' : 55,
    '(' : 56,
    ')' : 57,
    ',' : 58,
    '-' : 59,
    '.' : 60,
    ':' : 61,
    ';' : 62,
    '?' : 63
}

# convert to Laminar index
def to_idx(c):
    if 'A' <= c <= 'Z':
        return ord(c) - 65
    elif 'a' <= c <= 'z':
        return ord(c) - 71
    elif c in SpecialCharDict:
        return SpecialCharDict[c]
    elif c == '\n' or c == '\t' or c == '\r':
        return SpecialCharDict[' ']
    else: # skip 
        return -1

totalSize = 0

with open(fname + '.corpus', 'r') as corpus:
    # only a tmp file
    with open(fname + '.dattmp', 'wb') as outbintmp:
        while True:
            c = corpus.read(1)
            if c:
                idx = to_idx(c)
                if idx >= 0:
                    outbintmp.write(chr(idx))
                    totalSize += 1
            else: break

print 'Preprocessing done'

# write totalSize to the start of file
outbin = open(fname + '.dat', 'wb') 
outbin.write(struct.pack('i', totalSize))

with open(fname + '.dattmp', 'rb') as outbintmp:
    while True:
        dat = outbintmp.read(1024)
        if dat:
            outbin.write(dat)
        else: break
outbin.close()

print 'Write done'
os.remove(fname + '.dattmp')

# ==== print counting info
#   cc = Counter()

#   for l in open(fname, 'r'):
#       l = l.strip()
#       cc.update(l)

#   for entry in sorted(cc):
#       print entry, ord(entry), '=', cc[entry]



#   with open(origfname) as f:
#       while True:
#           c = f.read(1)
#           if not c:
#               print 'End of file'
#               break


