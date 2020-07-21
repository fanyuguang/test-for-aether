import argparse
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


# parser = argparse.ArgumentParser()
# parser.add_argument('--train_data_path', '-t', default='train_data.txt') # required=True
# parser.add_argument('--model_path', '-m', default='word2vec.model')
# parser.add_argument('--embedding_path', '-e', default='embedding.txt')
# args = parser.parse_args()
#
# train_data_path = args.train_data_path
# model_path = args.model_path
# embedding_path = args.embedding_path

train_data_path = 'train_data.txt'
model_path = 'word2vec.model'
embedding_path = 'embedding.txt'

# train
model = Word2Vec(LineSentence(train_data_path), size=50, window=2, min_count=1, workers=1, sg=0, hs=0, negative=20, sample=1e-4, iter=5)
model.save(model_path)
model.wv.save_word2vec_format(embedding_path, binary=False)
