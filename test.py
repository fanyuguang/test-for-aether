from gensim.models import Word2Vec

test_data_path = 'test_data.txt'
model_path = 'word2vec.model'
result_path = 'result.txt'

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
