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
parser.add_argument('--timeout', type=int, required=True, help='URLs download timeout time, in seconds (10 recommended).')
args = parser.parse_args()

output_directory = args.output_directory + '-' + timestamp
url_list_file = args.url_list_file
timeout = args.timeout

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

#
# iterate through the URLs
#
successful_url_list = []
successful_text_list = []
unsuccessful_url_list = []
for url in url_list:
    try:
        r = requests.get(url, timeout = timeout)
        if r.status_code != 200:
            unsuccessful_url_list.append(url)
            continue
        html = r.text
    except:
        unsuccessful_url_list.append(url)
        continue

    try:
        extractor = Extractor(extractor='ArticleExtractor', html=html)
        text = extractor.getText().replace('\\', '').strip().replace('\r', ' ').replace('\n', ' ')
    except:
        unsuccessful_url_list.append(url)
        continue

    successful_text_list.append(text)
    successful_url_list.append(url)

#
# write files
#
print()
print('There were ' + str(len(unsuccessful_url_list)) + ' unsuccessful webpage downloads. These URLs are listed in ' + output_directory + '/unsuccessful_url_list.txt')
f = open(output_directory + '/unsuccessful_url_list.txt', 'w')
f.write('\n'.join(unsuccessful_url_list) + '\n')
f.close()

print('There were ' + str(len(successful_url_list)) + ' successful webpage downloads. These URLs are listed in ' + output_directory + '/successful_url_list.txt')
f = open(output_directory + '/successful_url_list.txt', 'w')
f.write('\n'.join(successful_url_list) + '\n')
f.close()
print('Retrieved texts, in the order given in ' + output_directory + '/successful_url_list.txt, are listed in ' + output_directory + '/successful_text_list.txt')
f = open(output_directory + '/successful_text_list.txt', 'w')
f.write('\n'.join(successful_text_list) + '\n')
f.close()
print()


    
