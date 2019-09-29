## Introduction

This explains how to set up and use _HashGen_, a tool to generate hashtags from text.

HashGen uses some NLTK tools, such as the built-in stopword list and the WordNet lemmatiser.

You can configure the application, such as supplying additional stopwords and how to slice the results with your preferences (see how to use optional arguments below).

## Set-up
```
$ pip install -r requirements.txt
```

## Usage
Running the tool:
```
$ python hashgen.py input_dir [optional_args]
```

Learning about arguments (mandatory and optional):
```
$ python hashgen.py -h
```

If all went well, a file ("out.json" as per default) will be generated:

```
{
	"challenge": {
	    "docs": [
	      "doc1.txt",
	      "doc2.txt",
	      "doc3.txt",
	      "doc5.txt",
	      "doc6.txt"
	    ],
	    "sents": [
	      "As some of you know, Senator Lugar and I recently traveled to Russia, Ukraine, and Azerbaijan to witness firsthand both the progress we're making in securing the world's most dangerous weapons, as well as the serious challenges that lie ahead.",
	      "Now, few people understand these challenges better than the co-founder of the Cooperative Threat Reduction Program, Dick Lugar, and this is something that became particularly clear to me during one incident on the trip.",
	      "Throughout American history, there have been moments that call on us to meet the challenges of an uncertain world, and pay whatever price is required to secure our freedom.",
	      ...
	    ]
	}
	...
}
```

Where:

- Hash tags are the keys (e.g. "challenge").
- Hash tags indicate:
	- The documents they are associated with.
	- The sentences they are associated with.

Note that verbs and nouns are lemmatised (cars -> car, slept -> sleep) before becoming hash tags.

## Testing
Running unit tests:
```
$ python tests.py
```

(!) Note that the unit test suite is *far from complete*. It is here just for demonstration.  