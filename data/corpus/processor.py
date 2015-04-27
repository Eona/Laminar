from sys import argv
import os
from collections import Counter

fname = argv[1]

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

with open(fname, 'r') as corpus:
    with open(fname[:-len('.corpus')] + '.bin', 'wb') as outbin:
        while True:
            c = corpus.read(1)
            if c:
                idx = to_idx(c)
                if idx >= 0:
                    outbin.write(chr(idx))
            else: break



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


