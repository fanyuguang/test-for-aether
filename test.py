import argparse
from gensim.models import Word2Vec


parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', '-t', default='test_data.txt') # required=True
parser.add_argument('--model_path', '-m', default='word2vec.model')
parser.add_argument('--result_path', '-r', default='result.txt')
args = parser.parse_args()

test_data_path = args.test_data_path
model_path = args.model_path
result_path = args.result_path

# test
model = Word2Vec.load(model_path)
words = []
with open(result_path, encoding='utf-8', mode='w') as result_data_file:
  with open(test_data_path, encoding='utf-8', mode='r') as data_file:
    for line in data_file:
      word = line.strip()
      if word:
        similar_words = model.most_similar(word)
        result_data_file.write('{}, similar words:'.format(word) + '\n')
        for items in similar_words:
          result_data_file.write('    ({}, {})'.format(items[0], items[1]) + '\n')
