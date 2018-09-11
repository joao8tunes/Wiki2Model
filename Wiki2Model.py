#!/usr/bin/python3.4
# -*- coding: utf-8 -*-
################################################################################
##              Laboratory of Computational Intelligence (LABIC)              ##
##             --------------------------------------------------             ##
##       Originally developed by: João Antunes  (joao8tunes@gmail.com)        ##
##       Laboratory: labic.icmc.usp.br    Personal: joaoantunes.esy.es        ##
##                                                                            ##
##   "Não há nada mais trabalhoso do que viver sem trabalhar". Seu Madruga    ##
################################################################################

import multiprocessing
import subprocess
import datetime
import argparse
import codecs
import logging
import nltk
import os
import sys
import time
import gensim
import math
import uuid
import warnings


################################################################################
### FUNCTIONS                                                                ###
################################################################################

# Print iterations progress: https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
def print_progress(iteration, total, estimation, prefix='   ', decimals=1, bar_length=100, final=False):
    columns = 32    #columns = os.popen('stty size', 'r').read().split()[1]    #Doesn't work with nohup.
    eta = str( datetime.timedelta(seconds=max(0, int( math.ceil(estimation) ))) )
    bar_length = int(columns)
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s %s%s |%s| %s' % (prefix, percents, '%', bar, eta))

    if final == True:    #iteration == total
        sys.stdout.write('\n')

    sys.stdout.flush()


class load_sentences(object):    #File iterator: line by line (memory-friendly).
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname):
            yield line.split()


#Format a value in seconds to "day, HH:mm:ss".
def format_time(seconds):
    return str( datetime.timedelta(seconds=max(0, int( math.ceil(seconds) ))) )


#Convert a string value to boolean:
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("invalid boolean value: " + "'" + v + "'")


#Verify if a value correspond to a natural number (it's an integer and bigger than 0):
def natural(v):
    try:
        v = int(v)

        if v > 0:
            return v
        else:
            raise argparse.ArgumentTypeError("invalid natural number value: " + "'" + v + "'")
    except ValueError:
        raise argparse.ArgumentTypeError("invalid natural number value: " + "'" + v + "'")

################################################################################


################################################################################

#URL: https://github.com/joao8tunes/Wiki2Model

#Examples usage:
#python3 Wiki2Model.py --language EN --download in/db/ --extractor tools/WikiExtractor.py --output in/models/
#python3 Wiki2Model.py --language EN --preprocess in/db/ --extractor tools/WikiExtractor.py --output in/models/

#Obs:
#Possible codes: https://meta.wikimedia.org/wiki/Template:List_of_language_names_ordered_by_code
#Dumps repository: https://dumps.wikimedia.org/<LANGUAGE_CODE>/
#Default directory: https://dumps.wikimedia.org/enwiki/

#Defining script arguments:
parser = argparse.ArgumentParser(description="Word2Vec CBoW based model from Wikipedia 'raw' dump (XML) generator\n===================================================================")
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
optional.add_argument("--log", metavar='BOOL', type=str2bool, action="store", dest="log", nargs="?", const=True, default=False, required=False, help='display log during the process: y, [N]')
optional.add_argument("--download", metavar='DIR_PATH', type=str, action="store", dest="download", default=None, nargs="?", const=True, required=False, help='output directory to download latest Wikipedia dump')
optional.add_argument("--ignore_case", metavar='BOOL', type=str2bool, action="store", dest="ignore_case", nargs="?", const=True, default=False, required=False, help='ignore case: y, [N]')
optional.add_argument("--preprocess", metavar='DIR_PATH', type=str, action="store", dest="preprocess", default=None, nargs="?", const=True, required=False, help='output directory to preprocess texts from Wikipedia dump')
required.add_argument("--extractor", metavar='FILE_PATH', type=str, action="store", dest="extractor", default=None, required=True, nargs="?", const=True, help='file path to "WikiExtractor.py" script')
required.add_argument("--language", metavar='STR', type=str, action="store", dest="language", nargs="?", const=True, required=True, help='language of Wikipedia dump: EN, ES, FR, DE, IT, PT')
optional.add_argument("--size", metavar='INT', type=natural, action="store", dest="size", default=300, nargs="?", const=True, required=False, help='num. (>= 1) of model dimensions (used by Word2Vec): [300]')
optional.add_argument("--min_count", metavar='INT', type=natural, action="store", dest="min_count", default=10, nargs="?", const=True, required=False, help='min. words count (>= 1, used by Word2Vec): [10]')
optional.add_argument("--epochs", metavar='INT', type=natural, action="store", dest="epochs", default=5, nargs="?", const=True, required=False, help='model epochs (used by Word2Vec): [5]')
optional.add_argument("--threads", "-t", metavar='INT', type=natural, action="store", dest="threads", default=multiprocessing.cpu_count(), nargs="?", const=True, required=False, help='num. (>= 1) of threads (used by Word2Vec): [<CPU_COUNT>]')
required.add_argument("--output", "-o", metavar='DIR_PATH', type=str, action="store", dest="output", required=True, nargs="?", const=True, help='output directory to save the model')
args = parser.parse_args()    #Verifying arguments.

################################################################################


################################################################################

#Setup logging:
if args.log:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if args.language == "ES":      #Spanish.
    language_code = "eswiki"
    nltk_language = "spanish"
elif args.language == "FR":    #French.
    language_code = "frwiki"
    nltk_language = "french"
elif args.language == "DE":    #Deutsch.
    language_code = "dewiki"
    nltk_language = "german"
elif args.language == "IT":    #Italian.
    language_code = "itwiki"
    nltk_language = "italian"
elif args.language == "PT":    #Portuguese.
    language_code = "ptwiki"
    nltk_language = "portuguese"
else:                          #English.
    args.language = "EN"
    language_code = "enwiki"
    nltk_language = "english"

warnings.simplefilter(action='ignore', category=FutureWarning)
total_start = time.time()

################################################################################


################################################################################

log = codecs.open("Wiki2Model-log_" + time.strftime("%Y-%m-%d") + "_" + time.strftime("%H-%M-%S") + "_" + str(uuid.uuid4().hex) + ".txt", "w", "utf-8")
print("\nWord2Vec CBoW based model from Wikipedia 'raw' dump (XML) generator\n===================================================================\n\n\n")
log.write("Word2Vec CBoW based model from Wikipedia 'raw' dump (XML) generator\n===================================================================\n\n\n")

if not args.download is None:    #Downloading latest Wikipedia dump.
    today = time.strftime("%Y_%m_%d")
    dump_location = args.download + today + "/"
    args.download += today + "/raw_texts/"
    args.output += today + "/"
    url = "http://dumps.wikimedia.org/" + language_code + "/latest/" + language_code + "-latest-pages-articles.xml.bz2"

    if not os.path.exists(dump_location):
        print("> Creating directory for Wikipedia dump...\n")
        os.makedirs(dump_location, mode=0o755)
    else:
        print("ERROR: Directory for Wikipedia dump already exists!\n\t!Directory: " + dump_location)
        log.write("ERROR: Directory for Wikipedia dump already exists!\n\t!Directory: " + dump_location)
        log.close()
        sys.exit()

    #For more informations, see: https://code.google.com/archive/p/word2vec/
    print("> Downloading latest Wikipedia '" + args.language  + "' dump:")
    print("..................................................")
    os.system("wget " + url + " -P " + dump_location)
    print("..................................................\n")
    print("> Decompressing dump file...\n\n\n")
    os.system("bzip2 -dk " + dump_location + language_code + "-latest-pages-articles.xml.bz2")
    input_path = dump_location
    log.write("> Download:\n")
    log.write("\t- Language:\t\t" + args.language + "\n")
    log.write("\t- URL:\t\t" + url + "\n")
    log.write("\t- Location:\t\t" + dump_location + "\n")
    log.write("\t- Dump:\t\t" + dump_location + language_code + "-latest-pages-articles.xml\n")
    log.write("\t- Models:\t\t" + args.output + "\n\n\n")
elif not args.preprocess is None:    #Preprocessing raw texts from Wikipedia dump.
    input_path = args.preprocess
    log.write("> Preprocess:\n")
    log.write("\t- Language:\t\t" + args.language + "\n")
    log.write("\t- Location:\t\t" + args.preprocess + "\n")
    log.write("\t- Dump:\t\t" + args.preprocess + language_code + "-latest-pages-articles.xml\n")
    log.write("\t- Models:\t\t" + args.output + "\n\n\n")
else:
    parser.error("ERROR: choose --download or --preprocess!")
    log.write("\nERROR: choose --download or --preprocess!")
    log.close()
    sys.exit()

################################################################################


################################################################################

print("> Extracting raw texts from Wikipedia dump:")
print("..................................................")
print("> Creating directory for raw texts...\n")
raw_texts_location = input_path + "raw_texts/"
os.makedirs(os.path.abspath(raw_texts_location), mode=0o755)
log.write("> Raw texts: " + raw_texts_location + "\n")
print("> Extracting raw texts:\n")
#For more informations, see: https://github.com/attardi/wikiextractor/pull/59/files?short_path=04c6e90

if args.log:
    os.system("time python " + args.extractor + " --processes " + str(args.threads) + " --no-doc --no-title -o " + raw_texts_location + " " + input_path + language_code + "-latest-pages-articles.xml")
else:
    os.system("time python " + args.extractor + " --processes " + str(args.threads) + " --no-doc --no-title -o " + raw_texts_location + " --quiet " + input_path + language_code + "-latest-pages-articles.xml")

print("..................................................\n\n\n")
print("> Removing empty lines:")
print("..................................................")
files_list = []

#Reading all filepaths from all root directories:
for directory in os.listdir(raw_texts_location):
    for file_item in os.listdir(raw_texts_location + "/" + directory):
        files_list.append(raw_texts_location + directory + "/" + file_item)

files_list.sort()
total_num_examples = len(files_list)
filepath_i = 0
eta = 0
print_progress(filepath_i, total_num_examples, eta)
operation_start = time.time()
log.write("\t# Files: " + str(total_num_examples) + "\n\n")

# #Reading database:
for filepath in files_list:
    start = time.time()
    log.write("\t" + filepath + "\n")
    file_item = codecs.open(filepath, "r", "utf-8")
    paragraphs = [p.strip() for p in file_item.readlines()]    #Removing extra spaces.
    file_item.close()
    file_item = codecs.open(filepath, "w", "utf-8")

    for paragraph in paragraphs:
        if not paragraph.strip(): continue    #Ignoring blank line.
        file_item.write(paragraph + "\n")

    file_item.close()
    filepath_i += 1
    end = time.time()
    eta = (total_num_examples-filepath_i)*(end-start)
    print_progress(filepath_i, total_num_examples, eta)

operation_end = time.time()
eta = operation_end-operation_start
print_progress(total_num_examples, total_num_examples, eta, final=True)
print("..................................................\n\n\n")
print("> Creating directory for raw texts...\n")
raw_texts_location = input_path + "raw_texts/"
log.write("\n\n> Raw texts (split by sentences): " + raw_texts_location + "\n")
print("> Tokenizing raw texts by sentences:")
print("..................................................")
log.write("\t# Files: " + str(total_num_examples) + "\n\n")
raw_files_list = []
total_num_paragraphs = 0
total_num_sentences = 0
filePath_i = 0
eta = 0
print_progress(filePath_i, total_num_examples, eta)
operation_start = time.time()
filepath_sentences = "sentences_" + time.strftime("%Y-%m-%d") + "_" + time.strftime("%H-%M-%S") + "_" + str(uuid.uuid4().hex) + ".tmp"
file_sentences = codecs.open(filepath_sentences, "w", "utf-8")

#Reading database:
for filepath in files_list:
    start = time.time()
    new_filepath = filepath.replace(raw_texts_location, raw_texts_location)
    raw_files_list.append(new_filepath)
    log.write("\t" + new_filepath + "\n")
    file_item = codecs.open(filepath, "r", "utf-8")
    paragraphs = [p.strip() for p in file_item.readlines()]    #Removing extra spaces.
    file_item.close()
    total_num_paragraphs += len(paragraphs)
    new_dir = '/'.join( new_filepath.split("/")[:-1] ) + "/"    #Writing raw content to new file.

    if not os.path.exists(new_dir):
        os.makedirs(os.path.abspath(new_dir), mode=0o755)    #Creating intermediated directories.

    new_file_item = codecs.open(new_filepath, "w", "utf-8")

    for paragraph_i, paragraph in enumerate(paragraphs):
        paragraphs[paragraph_i] = nltk.sent_tokenize(paragraph, nltk_language)    #Identifying sentences.
        total_num_sentences += len(paragraphs[paragraph_i])

        for sentence in paragraphs[paragraph_i]:
            tokens = nltk.tokenize.word_tokenize(sentence)    #Works well for many European languages.

            if args.ignore_case:
                tokens = [t.lower() for t in tokens]

            raw_sentence = " ".join( tokens )
            file_sentences.write(raw_sentence + "\n")
            new_file_item.write(raw_sentence + "\n")

    new_file_item.close()
    filePath_i += 1
    end = time.time()
    eta = (total_num_examples-filePath_i)*(end-start)
    print_progress(filePath_i, total_num_examples, eta)

file_sentences.close()
operation_end = time.time()
eta = operation_end-operation_start
print_progress(total_num_examples, total_num_examples, eta, final=True)
print("..................................................\n\n\n")

################################################################################


################################################################################
### TRAINING (CONSTRUCTING THE MODEL)                                        ###
################################################################################

if not os.path.exists(args.output):
    print("> Creating directory to save models...")
    os.makedirs(os.path.abspath(args.output), mode=0o755)

print("> Training new model...")
model = gensim.models.Word2Vec(load_sentences(filepath_sentences), size=args.size, min_count=args.min_count, workers=args.threads, iter=args.epochs)
os.remove(filepath_sentences)

################################################################################


################################################################################
### OUTPUT (SAVING MODEL AND VECTORS)                                        ###
################################################################################

if args.log:
    print("")

print("> Saving model and vectors...")
model.save(args.output + "model_raw")
model.wv.save_word2vec_format(args.output + "model_raw.txt", binary=False)
num_vectors = subprocess.check_output("head -n 1 " + args.output + "model_raw.txt", shell=True).decode("utf-8").split()[0]

################################################################################


################################################################################

if args.log:
    print("")

total_end = time.time()
time = format_time(total_end-total_start)
files = str(total_num_examples)
paragraphs = str(total_num_paragraphs)
sentences = str(total_num_sentences)
vectors = str(num_vectors)
print("> Log:")
print("..................................................")
print("- Time: " + time)
print("- Files: " + files)
print("- Paragraphs: " + paragraphs)
print("- Sentences: " + sentences)
print("- Model vectors: " + vectors)
print("..................................................\n")
log.write("\n\n> Log:\n")
log.write("\t- Time:\t\t\t" + time + "\n")
log.write("\t- Files:\t\t" + files + "\n")
log.write("\t- Paragraphs:\t\t" + paragraphs + "\n")
log.write("\t- Sentences:\t\t" + sentences + "\n")
log.write("\t- Model vectors:\t" + vectors + "\n")
log.close()