import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np

def cossim(X,Y):
    # tokenization
    X_list = word_tokenize(X) 
    Y_list = word_tokenize(Y)

    # sw contains the list of stopwords
    sw = stopwords.words('english') 
    l1 =[];l2 =[]

    # remove stop words from the string
    X_set = {w for w in X_list if not w in sw} 
    Y_set = {w for w in Y_list if not w in sw}

    # form a set containing keywords of both strings 
    rvector = X_set.union(Y_set) 
    for w in rvector:
        if w in X_set: l1.append(1) # create a vector
        else: l1.append(0)
        if w in Y_set: l2.append(1)
        else: l2.append(0)
    c = 0

    # cosine formula 
    for i in range(len(rvector)):
            c+= l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5+1e-9)
    
    return cosine

class T5_WSD(object):
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained("jpelhaw/t5-word-sense-disambiguation").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("jpelhaw/t5-word-sense-disambiguation")
    
    def predict(self, context, target, definitions):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = self.model
        tokenizer = self.tokenizer
        
        temp_context = context.replace(target, '\" '+target+' \"')

        temp_description = ",\n".join(['\" '+definition+' \"' for definition in definitions])
        temp_description = '[ %s ]'%temp_description

        input_template = '''question: which description describes the word " %s "\
        best in the following context?
        descriptions:%s
        context: %s'''%(target, temp_description,temp_context)
        
        example = tokenizer.encode_plus(input_template) #tokenizer.tokenize(input, add_special_tokens=True)
        input_ids = torch.tensor([example['input_ids']]).to(device)
        attention_mask = torch.tensor([example['attention_mask']]).to(device)
        answer = model.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        max_length=135)
        answer = tokenizer.decode(answer[0][1:-1])
        
        cossims = [cossim(answer, definition) for definition in definitions]
        max_index = np.argmax(cossims)
        answer_definition = definitions[max_index]
        
        return answer_definition, max_index