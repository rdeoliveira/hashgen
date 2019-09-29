#!/usr/bin/env

'''HashGen: Generates hashtags from the most frequent (and relevant) words in a collection of documents.
'''

import argparse as ap
import json
import nltk
import os
import pandas as pd
import re
import sys

POS_TO_LEMMATISE = ['n','v']

BAD_HOMOGRAPHS = {
	'us':'PRP' # the pronoun (PRP), not the proper name (NNP) meaning the country U.S/US 
	}

def main(args):
	"""Generates hash tags from the most frequent words, excluding blacklisted ones, from a set of text files. Always uses NLTK's built-in stopword list as blacklist, in addition to any custom stopword list supplied.
    
    Args:
        Command line arguments. See ``parse_args`` for details.

    Produces:
        A JSON file with the hashtags and, for each tag, a list of documents and sentences where the tag appears. Or nothing, if no hash selected could be selected with the data and selection criteria supplied.
    """
	print("HashGen started...")
	update_nltk()
	data, counts = parse_texts(args.i, args.e, args.s)
	selected_tags = select_tags(make_df(counts), args.l, args.m)
	write_tags(selected_tags, data, args.o)
	print("HashGen done")

def update_nltk():
	"""Updates the NLTK module in the local environment.
    """
	nltk.download('stopwords')
	nltk.download('punkt')
	nltk.download('wordnet')
	nltk.download('averaged_perceptron_tagger')

def parse_texts(input_dir, extension, custom_stopwords):
	"""Parses all text files in an input directory, collecting tags, doc names and sentences for each tag.
    
    Args:
        input_dir (str): Path to input directory.
        extension (str): Extension of files to read.
        custom_stopwords (str): Path to the custom set of strings to ignore, i.e. to not generate has tags for.

    Returns:
        data: A dictionary with the generated hash tags as keys. Values are the locations (documents and sentences) where the tags appear. Format:
        	{
				tag : {
					docs : [],
					sents : []
				}
        	}
        counts: A dictionary with the generated hash tags as keys, and the total number of occurrences of each tag as values. Multiple occurrences of a word in the same sentence count muplitple times. Format:
        	{
				tag : count
        	} 
    """
	stops = build_stopword_list(custom_stopwords)
	data, counts = {}, {}
	for file in get_files(input_dir, extension):
		doc = os.path.basename(file)
		for line in open(file).readlines():
			for sent in nltk.sent_tokenize(line):
				for tag in get_tags(nltk.word_tokenize(sent), stops):
					update_data(doc, sent, tag, data, counts)
	return data, counts

def build_stopword_list(custom_stopwords):
	"""Combines NLTK's bult-in stopword set with the supplied custom set, if any.
    
    Args:
        custom_stopwords (str): Path to the custom set of strings to ignore, i.e. to not generate has tags for.

    Returns:
        stops: The combined set of NLTK's built-in stopwords plus custom ones.
    """
	stops = set(nltk.corpus.stopwords.words('english'))
	if custom_stopwords is not None:
		stops = stops.union(stops_from_file(custom_stopwords))
	return stops

def stops_from_file(custom_stopwords):
	"""Builds a set of custom stop wrods from file. Are lines are trimmed of leading/trailing spaces , and, after trimming, empty or lines starting with # are ignored.
    
    Args:
        custom_stopwords (str): Path to the custom set of strings to ignore, i.e. to not generate has tags for.

    Returns:
        A set of custom stopwords.

    Note:
    	If the file is invalid, print warning on console and exit programme.
    """
	if os.path.isfile(custom_stopwords):
		return set([line.rstrip('\n') for line in open(custom_stopwords) if not line.strip().startswith('#') and len(line.strip()) > 0])
	else:
		print("\"%s\" is not a valid text file.\nNo hashtags generated." % path)
		sys.exit(-1)

def get_files(input_dir, extension):
	"""Gets all files with the supplied extension from the supplied directory.
    
    Args:
        input_dir (str): Path to input directory.
        extension (str): Extension of files to read.

    Returns:
        A list of file paths.

    Note:
    	If the directory is invalid, print warning on console and exit programme.
    """
	if os.path.isdir(input_dir):
		return [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.' + extension)]
	else:
		print("\"%s\" is not a valid directory.\nNo hashtags generated." % path)
		sys.exit(-1)

def get_tags(tokens, stops):
	"""Generates hash tags by:

		1. Parsing a sequence of tokens, where parts-of-speech are associated with each token.
		2. Normalising tokens so that only 1 hash tag is generated for different orthographic styles of the same word (e.g. Car, car -> car).
		3. Lemmatising certain tokens so that only 1 tag is generated for different inflections of the same word (e.g. cars, car -> car).
		4. Excluding tokens that are empty ater normalisation or not interesting for different reasons. See ``good_tag`` for details.
    
    Args:
        tokens (list): A list of tokens pertaining to 1 sentence.
        stops (set): A set of stopwords.

    Returns:
        A list of hash tags.
    """
	tags = []
	for item in nltk.pos_tag(tokens):
		token = normalise_token(item[0])
		pos = item[1]
		if len(token) > 0:
			tag = lemmatise_tag(token, pos)
			if good_tag(tag, pos, stops):
				tags.append(tag)
	return tags

def normalise_token(token):
	"""Normalises a token by:

		1. Removing any non-alpha-numeric character and white spaces.
		2. Lower-casing letters.
    
    Args:
        token (str): A token.

    Returns:
        The normalised token.
    """
	return re.sub(r'[^\w\s]','',token).lower()

def lemmatise_tag(token, pos):
	"""Lemmatises a token using NLTK's WordNet lemmatiser, if the token's part-of-speech tag starts with the characters in ``POS_TO_LEMMATISE``.
    
    Args:
        token (str): A token.
        pos (str): A part-of-speech tag.

    Returns:
        The lemma of the token, if it should be lemmatised; otherwise, the token as is.
    """
	pos_start = pos[0].lower()
	return nltk.WordNetLemmatizer().lemmatize(token, pos_start) if pos_start in POS_TO_LEMMATISE else token

def good_tag(token, pos, stops):
	"""Checks whether a token should be turned into a hash tag.
    
    Args:
        token (str): A token.
        pos (str): A part-of-speech tag.
        stops (set): A set of stopwords.

    Returns:
        True if:
        	1. The token/part-of-speech combination is not black-listed in ``BAD_HOMOGRAPHS``.
        	2. The token is not composed of only numbers.
        	3. Token is not black-listed.
    """
	return not bad_homograph(token, pos) and not only_numeric(token) and not in_stop_list(token, stops)

def bad_homograph(token, pos):
	"""Checks whether the token/part-of-speech combination is black-listed in ``BAD_HOMOGRAPHS``.
    
    Args:
        token (str): A token.
        pos (str): A part-of-speech tag.

    Returns:
        True if:
        	1. The token is a key in the ``BAD_HOMOGRAPHS`` dictionary.
        	2. The value of the key equals the part-of-speech of the token.
    """
	return token in BAD_HOMOGRAPHS and BAD_HOMOGRAPHS[token] == pos

def only_numeric(tag):
	"""Checks whether the token contains only numbers.
    
    Args:
        token (str): A token.

    Returns:
        True if:
        	1. The token is empty after removing all numbers.
    """
	return len(re.sub(r'\d','',tag)) == 0

def in_stop_list(tag, stops):
	"""Checks whether the token is a stopword.
    
    Args:
        token (str): A token.
        stops (set): A set of stopwords.

    Returns:
        True if:
        	1. The stopword set contains the token.
    """
	return tag in stops

def update_data(doc, sent, tag, data, counts):
	"""Updates the dictionaries ``data`` and ``counts`` as follows:
		
		data: If the hash tag is not yet a key in the dictionary, create the key and start a list of documents and sentences for the tag containing the document and sentence supplied. If the tag already exists, add the document and the sentence to the appropriate lists.
		
		counts: If the hash tag is not yet a key in the dictionary, create the key with 1 as value. If the tag already exists, increment the count with 1.
    	
    Args:
    	doc (str): The name of the current file.
    	sent (str): The current sentence.
        tag (str): A hash tag.
        data (dict): A dictionary with the following structure:
        	{
				tag : {
					docs : [],
					sents : []
				}
        	}
        count (dict): A dictionary with the following structure:
        	{
				tag : count
        	}
    """
	if tag in data:
		data[tag]['docs'].append(doc) if doc not in data[tag]['docs'] else None 
		data[tag]['sents'].append(sent) if sent not in data[tag]['sents'] else None
		counts[tag] += 1
	else:
		data[tag] = {'docs':[doc], 'sents':[sent]}
		counts[tag] = 1

def make_df(counts):
	"""Creates a data frame with the dictionary of hash tags and counts.
    	
    Args:
        count (dict): A dictionary with the following structure:
        	{
				tag : count
        	}

    Returns:
    	A data frame with columns [tag, count], sorted by count, highest first.
    """
	df = pd.DataFrame(list(counts.items()), columns=["tag", "count"])
	return df.sort_values(by=["count"], ascending=False)

def select_tags(df, limit, metric):
	"""Selects hash tags based on a metric and limit. Examples:

		1. limit=10, metric='abs': Tags within the absolute top 10 counts are selected.
		2. limit=10, metric='pct': Tags within the top 10% counts are selected.
		3. limit=10, metric='min': Tags with at least 10 occurrences are selected.
    	
    Args:
    	df: A data frame with columns [tag, count].
    	limit (int): A number to be used when selecting tags. 
        metric (str): How to use the limit when selecting tags. 

    Returns:
    	A list of selected hash tags.
    """
	if metric == 'min':
		return get_tags_with_min(df, limit)
	else:
		return get_top_tags(df, limit, metric)

def get_tags_with_min(df, limit):
	"""Selects tags with a minimum value of occurrences. 
    	
    Args:
    	df: A data frame with columns [tag, count].
    	limit (int): The minimum number of occurrences of a tag to be selected.

    Returns:
    	A list of selected hash tags.
    """
	return df[df['count'] >= limit]['tag'].tolist()

def get_top_tags(df, limit, metric):
	"""Selects tags with the top number of occurrences. 
    	
    Args:
    	df: A data frame with columns [tag, count].
    	limit (int): The minimum number of occurrences of a tag to be selected.
    	metric (str): The metric to use when computing 'top' occurrences.

    Returns:
    	A list of selected hash tags.
    """
	return df.nlargest(get_n(df, limit, metric), 'count')['tag'].tolist()

def get_n(df, limit, metric):
	"""Computes the actual number of occurences to consider as 'top' occurences. 
    	
    Args:
    	df: A data frame with columns [tag, count].
    	limit (int): The limit to use when computing 'top'.
    	metric (str): The metric to use when computing 'top' occurrences.

    Returns:
    	The percent limit if the metric is 'pct'; otherwise, the absolute limit.
    """
	total = len(df)
	if metric is not None and metric == "pct":
		limit = total if limit >= 100 else int(limit * total / 100)
	return limit

def write_tags(selected_tags, data, out_file):
	"""Writes a JSON file with the generated and selected hash tags. 
    	
    Args:
    	selected_tags: A list of hash tags to be written to file.
    	data (dict): A dictionary with all generated tags. Format:
    		{
				tag : {
					docs : [],
					sents : []
				}
        	}
        out_file (str): The path of the file to write.

    Produces:
    	A JSON file with the selected hash tags and their associated locations (documents and sentences). Format:
			{
				tag : {
					docs : [],
					sents : []
				}
        	}
        If no tags could be selected, skip writing the JSON and print warning to console.
    """
	if (len(selected_tags) > 0):
		with open(out_file, 'w') as o:
			json.dump(filter_data(selected_tags, data), o, indent=2, ensure_ascii=False, sort_keys=True)
	else:
		print("No hashtags could be generated with the supplied data and selection criteria.")

def filter_data(selected_tags, data):
	"""Creates a subset of the generated hash tag dictionary. 
    	
    Args:
    	selected_tags: A list of hash tags to be used when creating the subset of all generated hash tags.
    	data (dict): A dictionary with all generated tags. Format:
    		{
				tag : {
					docs : [],
					sents : []
				}
        	}

    Returns:
    	A dictionary with only the generated tags and same format as the original set.
    """
	return {tag : data[tag] for tag in selected_tags}

def parse_args():
	"""Parses command line arguments.

    Returns:
    	A dictionary of arguments.
    """
	parser = ap.ArgumentParser(description='HashGen : Generates hashtags from a input directory with text files. Accepts custom filtering criteria (see args -l -m and -s).')
	parser.add_argument('i',
		metavar='input_dir',
		type=str,
		help='the path to the directory with the input files')
	parser.add_argument('-e',
		metavar='file_extension',
		type=str,
		default='txt',
		help='the extension of files to look for; txt is default')
	parser.add_argument('-l',
		metavar='limit',
		type=int,
		default=10,
		help='the number of hash tags to generate; see param -m for further details; 10 is default')
	parser.add_argument('-m',
		metavar='metric',
		type=str,
		choices=['abs', 'pct', 'min'],
		default='abs',
		help='whether you want the absolute (\'abs\') top -l, the percent (\'pct\') top -l or a minimum (min) -l occurences; \'abs\' is default')
	parser.add_argument('-s',
		metavar='stop_words',
		type=str,
		help='a custom list of stop words')
	parser.add_argument('-o',
		metavar='out_file',
		type=str,
		default='./out.json',
		help='the output file path; \'./out.json\' is default')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	main(parse_args())