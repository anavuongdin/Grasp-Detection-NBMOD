import os
import pickle

def read_prompts(file_name):
    with open(file_name, 'rb') as f:
        x = pickle.load(f)
        x = x.strip('\n')

    for line in x.split('\n'):
        try:
            prompt, queries = line.split('.')
        except:
            print(line)
        queries = queries.strip(' []')
        queries = queries.split(' ')
        queries = list(map(lambda x: x.strip(','), queries))
        yield prompt, queries