#!/usr/bin/python

import cv2
import getopt
import lane
from moviepy.editor import VideoFileClip
import pickle
import sys


def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print('Input file is "', inputfile)
   print('Output file is "', outputfile)

   detector = lane.LaneDetector("calibration.p", debug=False)

   clip = VideoFileClip(inputfile)
   project_clip = clip.fl_image(detector.process)

   project_clip.write_videofile(outputfile, audio=False)

if __name__ == "__main__":
   main(sys.argv[1:])


