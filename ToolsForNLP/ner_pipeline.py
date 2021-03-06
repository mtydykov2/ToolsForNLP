import nltk
from os import listdir
from os.path import isfile, join
from xml.dom import minidom
from nltk.tokenize import WhitespaceTokenizer
mypath = "Tools-IE-assignment/Annotated/"
onlyfiles = [ mypath+f for f in listdir(mypath) if isfile(join(mypath,f)) and "tokenized" not in f ]
myannotations="Tools-IE-assignment/Raw/"
onlyannotations = [ myannotations+f for f in listdir(myannotations) if isfile(join(myannotations,f))]
dicts_for_files = {}
for f in onlyannotations:
	identifier = f[f.rfind('/')+1:f.rfind('.')-4]
	dicts_for_files[identifier] = {}
	xmldoc = minidom.parse(f)
	entitylist = xmldoc.getElementsByTagName('entity')
	for entity in entitylist:
		entitymention_list = entity.getElementsByTagName('entity_mention')
		entity_type = entity.attributes['type'].value
		for entitymention in entitymention_list:
			begin_index = entitymention.attributes['offset'].value
			dicts_for_files[identifier][begin_index]=(entity_type, entitymention.attributes['length'].value)
			#print entitymention.firstChild.nodeValue


newf = open('training','w')
for f in onlyfiles:
		with open (f, "r") as myfile:
				data=myfile.read().encode('utf-8')
		# identifier is up to the index of the first '.'
		identifier = f[f.rfind('/')+1:f.rfind('.')]
		f_ne_dict = dicts_for_files[identifier]
		tokenized = list(WhitespaceTokenizer().span_tokenize(data))
		
		i = 0
		named_entity_tags = []
		tokens = []
		while i < len(tokenized):
			token = tokenized[i]
			# token was marked as a named entity
			if unicode(token[0]) in f_ne_dict:
				len_of_entity = f_ne_dict[unicode(token[0])][1]
				begin_of_entity = token[0]
				named_entity = f_ne_dict[unicode(token[0])][0]
				named_entity_tags.append("B-"+named_entity)
				tokens.append(data[token[0]:token[1]])
				lookahead_index = i+1
				if lookahead_index < len(tokenized):
					lookahead_token = tokenized[lookahead_index]
					span_covered = lookahead_token[0] - begin_of_entity
					while(lookahead_index < len(tokenized) and int(span_covered) < int(len_of_entity)):
						lookahead_token = tokenized[lookahead_index]
						span_covered = lookahead_token[0] - begin_of_entity
						if int(span_covered) < int(len_of_entity):
							named_entity_tags.append("I-"+named_entity)
							tokens.append(data[lookahead_token[0]:lookahead_token[1]])
						lookahead_index+=1
				i = lookahead_index-1
			else:
				named_entity_tags.append("O")
				tokens.append(data[token[0]:token[1]])
			i+=1
		i = 0
		
		pos_tags = [tup[1] for tup in nltk.pos_tag(tokens)]
		all_elements = zip(tokens, pos_tags, named_entity_tags)
		while i < len(all_elements) and all_elements[i][0] != "HEADER:":
			for element in all_elements[i]:
				newf.write(element + "\t")
			newf.write("\n")
			i+=1
newf.close()


