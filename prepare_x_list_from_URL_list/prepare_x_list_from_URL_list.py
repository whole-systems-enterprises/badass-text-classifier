#
# import useful libraries
#
import argparse
import os
import datetime
import requests
from boilerpipe.extract import Extractor

#
# get current date and time
#
timestamp = str(datetime.datetime.now()).replace(' ', '-').replace(':', '-')

#
# parse command line arguments
#
parser = argparse.ArgumentParser(description='Create a list of texts given a list of URLs.')
parser.add_argument('--output-directory', type=str, required=True, help='Name of directory to place output (will be erased/created).')
parser.add_argument('--url-list-file', type=str, required=True, help='Path and file name of lists of URLs, one URL per line.')
args = parser.parse_args()

output_directory = args.output_directory + '-' + timestamp
url_list_file = args.url_list_file

#
# erase and create path for output
#
if os.path.isdir(output_directory):
    os.system('rm -R ' + output_directory)
os.system('mkdir ' + output_directory)

#
# load list of urls
#
url_list = []
f = open(url_list_file)
for line in f:
    url_list.append(line.strip())
f.close()

print(url_list)
