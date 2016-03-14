# Adjusted from https://github.com/MLWave/Kaggle-Ensemble-Guide/blob/master/kaggle_geomean.py


from __future__ import division
from collections import defaultdict
from os import listdir
import os
import sys
import math

glob_files = sys.argv[1]
loc_outfile = sys.argv[2]

def kaggle_bag(glob_files, loc_outfile, method="average", weights="uniform"):
  if method == "average":
    scores = defaultdict(float)
  with open(loc_outfile,"wb") as outfile:
    for i, glob_file in enumerate( [os.path.join(glob_files, file) for file in listdir(glob_files)]):
      print "parsing:", glob_file
      # sort glob_file by first column, ignoring the first line
      lines = open(glob_file).readlines()
      lines = [lines[0]] + sorted(lines[1:])
      for e, line in enumerate( lines ):
        if i == 0 and e == 0:
          outfile.write(line)
        if e > 0:
          row = line.strip().split(",")
          if scores[(e,row[0])] == 0:
            scores[(e,row[0])] = 1
          scores[(e,row[0])] *= float(row[1])
    for j,k in sorted(scores):
      outfile.write("%s,%f\n"%(k,math.pow(scores[(j,k)],1/(i+1))))
    print("wrote to %s"%loc_outfile)

kaggle_bag(glob_files, loc_outfile)
