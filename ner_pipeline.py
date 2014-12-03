import nltk
from os import listdir
from os.path import isfile, join
import subprocess
import os
from xml.dom import minidom
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.corpus import wordnet


def create_gold_standard(onlyannotations, onlyfiles):
	'''
	Create gold standard-marked sentence split/tokenized files.
	'''
	dicts_for_files = {}
	training_with_gold_tags_file = open('training_with_gold_tags', 'w')
	# look at each training file
	for f in onlyannotations:
		# get just the identifier for the file
		identifier = f[f.rfind('/') + 1:f.rfind('.') - 4]
		dicts_for_files[identifier] = {}
		# parse the annotation xml
		xmldoc = minidom.parse(f)
		# find all entities
		entitylist = xmldoc.getElementsByTagName('entity')
		for entity in entitylist:
			# find all entity mentions
			entitymention_list = entity.getElementsByTagName('entity_mention')
			# get the mention type
			entity_type = entity.attributes['type'].value
			# add the entity mention, the character begin offset in the 
			# text, and the entity type to dictionary of files to entity mention
			for entitymention in entitymention_list:
				begin_index = entitymention.attributes['offset'].value
				dicts_for_files[identifier][begin_index] = (entity_type, entitymention.attributes['length'].value)

	named_entity_tags = []
	tokens = []
	# look at the raw text data
	for f in onlyfiles:
			with open (f, "r") as myfile:
					data = myfile.read().encode('utf-8')
			# identifier is up to the index of the first '.'
			identifier = f[f.rfind('/') + 1:f.rfind('.')]
			# get the dictionary of entity mention offset to 
			# info about mention from the general dictionary for all files
			f_ne_dict = dicts_for_files[identifier]
			
			tokenized_spans = []
			
			for start, end in PunktSentenceTokenizer().span_tokenize(data):
				# tokenize the raw file
				tokenized = list(WhitespaceTokenizer().span_tokenize(data[start:end]))
				modified_tokenized = []
				for tup in tokenized:
					new_tup = tup[0]+start, tup[1]+start, tup[0] == 0
					print data[new_tup[0]:new_tup[1]]
					modified_tokenized.append(new_tup)
				tokenized_spans.append(modified_tokenized)
			
			i = 0
			for sent in tokenized_spans:
				while i < len(sent):
					token = sent[i]
					begin_of_sent = token[2]
					# token was marked as a named entity
					if unicode(token[0]) in f_ne_dict:
						len_of_entity = f_ne_dict[unicode(token[0])][1]
						begin_of_entity = token[0]
						named_entity = f_ne_dict[unicode(token[0])][0]
						named_entity_tags.append("B-" + named_entity)
						text_and_begin = data[token[0]:token[1]], begin_of_sent
						tokens.append(text_and_begin)
						# group all tokens that are part of this mention
						lookahead_index = i + 1
						if lookahead_index < len(sent):
							lookahead_token = sent[lookahead_index]
							span_covered = lookahead_token[0] - begin_of_entity
							while(lookahead_index < len(sent) and int(span_covered) < int(len_of_entity)):
								lookahead_token = sent[lookahead_index]
								span_covered = lookahead_token[0] - begin_of_entity
								if int(span_covered) < int(len_of_entity):
									named_entity_tags.append("I-" + named_entity)
									text_and_begin=data[lookahead_token[0]:lookahead_token[1]],begin_of_sent
									print text_and_begin
									tokens.append(text_and_begin)
								lookahead_index += 1
						i = lookahead_index - 1
					else:
						named_entity_tags.append("O")
						text_and_begin=data[token[0]:token[1]],begin_of_sent
						print text_and_begin
						tokens.append(text_and_begin)
					i += 1
				i = 0
			
			# write the gold standard file
	all_elements = zip(tokens, named_entity_tags)
	i = 0
	done = False
	while i < len(all_elements) and not done:
		element = all_elements[i] 
		if element[0][1] and i != 0:
			training_with_gold_tags_file.write("\n")
		if element[0][0] != "HEADER":
			training_with_gold_tags_file.write(element[0][0]+"\t"+element[1]+"\n")
		else: 
			done = True
		i+=1
					
	training_with_gold_tags_file.close()
	extract_features(tokens, 'nltk_features.conll')

	
def extract_features(tokens, fname):
	'''
	Extract features given a list of tuples with the token
	and whether or not it's the beginning of a sentence.
	Write to given file.
	'''
	
	pos_tags = [tup[1] for tup in nltk.pos_tag([token[0] for token in tokens])]
	all_elements = zip(tokens, pos_tags)
	conll_file = open(fname, 'w')
	i = 0
	for j, token in enumerate(all_elements):
		if token[0][1]:
			i = 0
			if j != 0:
 				conll_file.write("\n")
		token_line = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t".format(i,token[0][0],token[0][0],"_",token[1],"_","0","_","_","_")
 		conll_file.write(token_line+"\n")
 		i+=1
 	conll_file.close()


def read_dep_tagged_features(dep_tagged_file):
	'''
	Given a file that has tokenized/dependency tagged data,
	read the data in.
	'''
	token_to_features = []
	synset_to_hint = {wordnet.synset('position.n.05'):"title",wordnet.synset('position.n.06'):"title",
					wordnet.synset('organization.n.03'):"organization",
					wordnet.synset('person.n.01'):"person",
					wordnet.synset('place.n.01'):"place"}
	for line in dep_tagged_file:
		if line != "\n":
			parts = line.split()    #token, POS, dep rel
			new_token_to_features = parts[1], parts[4], parts[7]
			token_to_features.append(new_token_to_features)
		else:
			token_to_features.append(None)
	final_token_to_features = []
	for item in token_to_features:
		if item != None:
			synsets = wordnet.synsets(item[0])
			marked_synsets = synset_to_hint.keys()
			for marked_synset in marked_synsets:
				lch = marked_synset.hyponyms()
				matches = False
				for synset in synsets:
				# not the case that nothing was returned, 
				# and if something was returned then it has only 1 elemenet which is entity
					if synset in lch:
						matches = True
						break
				
				item += matches,
			final_token_to_features.append(item)
		else:
			final_token_to_features.append(None)
	return final_token_to_features
			
			
def put_together_final_training(final_token_to_features, gold_file):
	'''
	Given the final token/feature data and the file
	containing the same tokens mapped to the gold
	standard answers, write to the final training file.
	'''
	training_file = open('training_file','w')
	for i, line in enumerate(gold_file):
		gold_label = ""
		if line != "\n":
			write_line = ""
			gold_label = line.split()[1]
			for item in final_token_to_features[i]:
				write_line+=str(item)+"\t"
			write_line+=gold_label+"\n"
			training_file.write(write_line)
		else:
			training_file.write("\n")
	training_file.close()
	
def put_together_final_test(final_token_to_features):
	'''
	Given a list of tokens+features (and None if sentence break),
	write to test file.
	'''
	training_file = open('test_file','w')
	for element in final_token_to_features:
		if element != None:
			write_line = ""
			for item in element:
				write_line += str(item)+"\t"
			write_line+="\n"
			training_file.write(write_line)
		else:
			training_file.write("\n")
	training_file.close()
	
def train():
	'''
	Perform training process, given the raw training file and
	the annotations file.
	'''
	mypath = "ToolsForNLP/Tools-IE-assignment/Annotated/"
	onlyfiles = [ mypath + f for f in listdir(mypath) if isfile(join(mypath, f)) and "tokenized" not in f ]
	myannotations = "ToolsForNLP/Tools-IE-assignment/Raw/"
	# # get all annotated training data
	onlyannotations = [ myannotations + f for f in listdir(myannotations) if isfile(join(myannotations, f))]
	create_gold_standard(onlyannotations, onlyfiles)
	os.chdir('/home/mtydykov/ToolsForNLP/HW4/malt/')
	subprocess.call('java -Xmx1024m -jar malt.jar -c engmalt.linear-1.7 -i ../../HW8NER/nltk_features.conll -o ../../HW8NER/outfile.conll -m parse', shell=True)
	os.chdir('/home/mtydykov/ToolsForNLP/HW8NER/')
	final_token_to_features = read_dep_tagged_features(open('outfile.conll','r'))
	put_together_final_training(final_token_to_features, open('training_with_gold_tags','r'))
	print os.getcwd()
	subprocess.call('wine CRF++-0.58/crf_learn.exe CRF++-0.58/hw8_template training_file crf_model', shell=True)

def test():
	'''
	Use the trained model to tag the test file.
	'''
	with open ('/home/mtydykov/ToolsForNLP/HW8NER/Test/muc3-tst1.txt', "r") as myfile:
		data = myfile.read().encode('utf-8')
	sentences = sent_tokenize(data)
	tokens = []
	for sentence in sentences:
		token_list = PunktWordTokenizer().tokenize(sentence)
		new_tup = token_list[0], True
		tokens.append(new_tup)
		tokens.extend((token, False) for token in token_list[1:])
	extract_features(tokens, 'nltk_test_features.conll')
	os.chdir('/home/mtydykov/ToolsForNLP/HW4/malt/')
	subprocess.call('java -Xmx1024m -jar malt.jar -c engmalt.linear-1.7 -i ../../HW8NER/nltk_test_features.conll -o ../../HW8NER/test_outfile.conll -m parse', shell=True)
	os.chdir('/home/mtydykov/ToolsForNLP/HW8NER/')
	
	final_token_to_features = read_dep_tagged_features(open('test_outfile.conll','r'))
	put_together_final_test(final_token_to_features)
	subprocess.call('wine CRF++-0.58/crf_test.exe -m crf_model test_file > test_result 2>&1', shell = True)

train()
test()
