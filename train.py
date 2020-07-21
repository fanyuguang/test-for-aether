from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


train_data_path = 'train_data.txt'
model_path = 'word2vec.model'
embedding_path = 'embedding.txt'

# train
model = Word2Vec(LineSentence(train_data_path), size=300, window=2, min_count=0, workers=1, sg=0, hs=0, negative=20, sample=1e-4, iter=5)
model.save(model_path)
model.wv.save_word2vec_format(embedding_path, binary=False)
