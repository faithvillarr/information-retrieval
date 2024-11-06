import nltk
import math
from nltk.tokenize import word_tokenize, sent_tokenize

# GENERAL FUNCTIONS


# FUNCTIONS FOR IMPORTING INFORMATION
def get_queries_dict(filepath):
    with open(filepath) as file:
        lines = file.readlines()
    
    queries = {}
    i = 0 # Will signify the line number

    while i < len(lines):

        if lines[i][0:2] == '.I':
            # Add query number to dict queries
            query_num = int(lines[i][2:].strip())

            queries[query_num] = ""
            i += 1 # Go to next line

            while i < len(lines) and lines[i][0:2] != '.W':
                print('Might be issue with parsing.')
                i += 1

            i += 1

            # Now we should be at the actual string of the query
            while i < len(lines) and lines[i][0:2] != '.I':
                # print(i, len(lines))
                queries[query_num] += lines[i].strip() + " "
                i += 1
                
    return queries     

def get_all_docs_text(filepath):
    with open(filepath) as file:
        lines = file.readlines()

    docs = {}
    current_doc_id = None
    title = ""
    author = ""
    abstract = ""

    field_type = -1  # Indicates the current field being recorded
    # Mapping: 0 - id, 1 - title, 2 - author, 3 - bibi, 4 - abstract, -1 - INVALID

    for line in lines:
        # Check if type indicator line
        if line.startswith('.'):
            ind = line[1]
            
            # If we are storing a new doc, store and reset info
            if ind == 'I':
                # Store
                if current_doc_id is not None:
                    docs[current_doc_id] = f"{title.strip()} {author.strip()} {abstract.strip()}"
                
                # Reset
                current_doc_id = int(line[2:].strip())
                title = ""
                author = ""
                abstract = ""
                field_type = 0
                
            elif ind == 'T':
                field_type = 1
            elif ind == 'A':
                field_type = 2
            elif ind == 'B':
                field_type = 3  # ignoring
            elif ind == 'W':
                field_type = 4

        # Append text to the current field based on the field_type
        elif current_doc_id is not None and field_type != -1:
            if field_type == 1:
                title += line.strip() + " "
            elif field_type == 2:
                author += line.strip() + " "
            elif field_type == 4: 
                abstract += line.strip() + " "

    # Add the last doc
    if current_doc_id is not None:
        docs[current_doc_id] = f"{title.strip()} {author.strip()} {abstract.strip()}"

    return docs

# FUNCTIONS FOR PROCESSING VECTORS
def cosine_similarity(vec1, vec2): # vec1 must be query 
    # Calculate the dot product
    dot_product = sum(vec1[word] * vec2.get(word, 0) for word in vec1) # only look for query words
    
    # Calculate magnitudes of each vector
    magnitude1 = math.sqrt(sum(value ** 2 for value in vec1.values()))
    magnitude2 = math.sqrt(sum(value ** 2 for value in vec2.values()))
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Calculate cosine similarity
    return dot_product / (magnitude1 * magnitude2)

def normalize_vec(vec):
    # Normalize word counts for query length
    # ? Make nouns weighted more?
    inst_total = sum(vec.values())
    for key in vec.keys():
        vec[key] = vec[key] / inst_total
    return vec

def vec_from_text(text):

    # Grab nouns with POS tagging and use just nouns in vector to lower processing load
    wordsList = nltk.word_tokenize(text)
    # print(query, '\n', wordsList)
    tagged = nltk.pos_tag(wordsList)

    # 'tagged' holds pairs of ('word', 'POS tag')
    tags_to_keep = ['FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NP', 'NPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    res_vec = {} # {'word' : # of instances in this query}

    # Count frequency of words and store in query_vecs
    for word, tag in tagged:
        if tag in tags_to_keep:
            # We have found a word we want to count at this point
            if word not in res_vec:
                res_vec[word] = 0
            res_vec[word] += 1

    return res_vec #dictionary ('word' : count)    


if __name__ == "__main__":
    doc_filepath = 'Cranfield_collection_HW\cran.all.1400'
    query_filepath = 'Cranfield_collection_HW/cran.qry'

    # Import all query and document text
    doc_dict = get_all_docs_text(doc_filepath)  # {document_number  : document_text}
    que_dict = get_queries_dict(query_filepath) # {query_number     : query_text}

    # Create frequency vectors for each text
    for key in doc_dict.keys():
        doc_dict[key] = normalize_vec(vec_from_text(doc_dict[key]))
    for key in que_dict.keys():
        que_dict[key] = normalize_vec(vec_from_text(que_dict[key]))

    '''
    At this point, we have two nested dictionaries. 
        que_dict = { query_id : {'word': frequency in decimal }}
        doc_dict = { doc_id   : {'word': frequency in decimal }}
    '''

    s = 0
    print("Writing similarity scores...")
    for q_key, q_vec in que_dict.items():
        for d_key, d_vec in doc_dict.items():
            # Get cosine sim of each vector pair
            cos_sim = cosine_similarity(q_vec, d_vec)

            # Write to output
            f = open("output.txt", 'a')

            # Define what we want to write
            new_line = str(q_key) + " " + str(d_key) + " " + str(round(cos_sim, 3)) + "\n"

            # Write to file
            if cos_sim != 0: f.write(new_line)
            f.close()

    
        
    # for i in range(1, 4):
    #     if i in doc_dict.keys():
    #         print("DOC: ", doc_dict[i])
    #     if i in que_dict.keys():
    #         print("QUE: ", que_dict[i])
    

    '''
    QueryID DocID Cosine Score
    '''

    # doc_vecs = {} # { int doc_id : {words : count}}""
    # for i in range(1, 1401):
    #     # if i % 100 == 0: print(f"On doc {i}...")
    #     doc_text = get_doc_text('Cranfield_collection_HW\cran.all.1400', i)

    #     doc_vec = vec_from_text(doc_text)
    #     if 1 < i < 5: print(i, doc_vec)
    #     doc_vecs[i] = doc_vec

    # print(f'Num of Docs Found: {len(doc_vecs.keys())}')
    # query_num = 10
    # for key in doc_vec.keys():
    #     sim_score = cosine_similarity(doc_vecs[key], queries[query_num])

    #     f = open("train-output.txt", "a")
    #     output_str = str(int(query_num)) + " " + str(key) + " " + str(round(sim_score, 3)) + "\n"
    #     print(f'Adding line: {output_str}')
    #     f.write(output_str)
    #     f.close()

    # Now queries = {'query_num' : {'word_1': freq as percent of q_len }}
    # We can be done with queries for now. 

