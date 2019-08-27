import os
import time
import numpy as np
import tensorflow as tf
from datetime import timedelta
from collections import Counter
import tensorflow.contrib.keras as kr
import jieba as jb
from sklearn import metrics
import matplotlib.pyplot as plt

# GPU训练
# os.environ['CUDA_VISIBLE_DEVICES']="7"
"""
任务目标：情感分类
任务流程：
1. 加载数据
2. 数据预处理
3. 构建词汇表
4. 学习embedding表示
"""

# 数据预处理
def cat_to_id(classes=None):
    '''
    :param classes: 分类标签：默认为0--positive 1--negative
    :return: {分类标签：id}
    '''
    if not classes:
        classes = ['0','1']
    cat2id = {cat: idx for (idx, cat) in enumerate(classes)}
    # {pos:0, neg:1}
    return classes, cat2id

# 构建词汇表并存储，{word: id}
# only one start
def build_word2id(file):
    """
    :param file: word2id 保存地址
    :return: None
    """
    word2id = {'_PAD_': 0}
    path = ['./data/train.txt', './data/validation.txt']
    print(path)
    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:   # 0位置为标签
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)  # id等于当前词典长度
                        """
                        raw_input()          #' insert 0 5     '
                        raw_input().strip()  #'insert 0 5'
                        raw_input().strip().split()  #['insert', '0', '5']
                        """

    with open(file, 'w', encoding='utf-8') as f:
        for w in word2id:
            f.write(w+'\t')
            f.write(str(word2id[w]))
            f.write('\n')

# 加载词汇表
def load_word2id(path):
    """

    :param path: word_to_id 词汇表路径
    :return: word_to_id: {wprd: id}
    """
    word_to_id = {}  # 定义空词汇表
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            word = sp[0]
            idx = int(sp[1])
            if word not in word_to_id:
                word_to_id[word] = idx
    return word_to_id

# 基于预训练好的word2vec构建训练语料中所含词语的word2vec
def build_word2vec(fname, word2id, save_to_path=None):
    """

    :param fname: 预训练的word2vec
    :param word2id: 预料文本中包含的词汇集
    :param save_to_path: 保存训练语料库中词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}
    """
    import gensim
    n_words = max(word2id.values()) + 1 # 词的个数
    # 使用预训练模型学习word的表示  embedding
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    """
     numpy.random.uniform(low,high,size)
     功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    """
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]  #向量表示
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(''.join(vec))
                f.write('\n')  # 每一行是一个word 的embedding。
    return word_vecs  # 学到所有word的embedding

def load_corpus_word2vec(path):
    """
    加载语料库 word2vec词向量，相对wiki词向量较小
    :param path:
    :return:
    """
    word2vec = []
    with open(path, encoding = 'utf-8') as f:
        for line in f.readlines():
            sp = [float(w) for w in line.strip().split()]
            word2vec.append(sp)
    return np.asarray(word2vec)

def load_corpus(path, word2id, max_sen_len = 70):
    """
    :param path: 样本语料库的文件   train/dev/test
    :param word2id:
    :param max_sen_len:
    :return: 文本内容contents  分类标签labels (onehot)
    """

    _, cat2id = cat_to_id()
    contents, labels = [], []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            label = sp[0]
            content = [word2id.get(w, 0) for w in sp[1:]]
            content = content[:max_sen_len]  #截断
            if len(content) < max_sen_len: #补0
                content += [word2id['_PAD_']] * (max_sen_len - len(content))
            labels.append(label)
            contents.append(content)
    counter = Counter(labels)
    print('总样本数：%d' % (len(labels)))
    print('各类别样本数：')
    for w in counter:
        print(w, counter[w])

    contents = np.asarray(contents)
    labels = [cat2id[1] for 1 in labels]
    labels = kr.utils.to_categorical(labels, len(cat2id))

    return contents, labels   # id和label

def batch_index(length, batch_size, is_shuffle=True):
    """
    生成批处理样本序列id
    :param length: 样本总数
    :param batch_size: 批处理大小
    :param is_shuffle: 是否打乱样本顺序
    :return:
    """
    index = [idx for idx in range(length)]
    if is_shuffle:
        np.random.shuffle(index)
    for i in range(int(np.ceil(length/batch_size))):
        yield index[i * batch_size:(i+1)*batch_size]
        """
        np.ceil(ndarray) 
        计算大于等于改值的最小整数
        
        >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
        >>> np.ceil(a)
        array([-1., -1., -0.,  1.,  2.,  2.,  2.])
        """


"""
经过数据预处理，数据的格式如下：
x: [1434, 5454, 2323, ..., 0, 0, 0]
y: [0, 1]
x为构成一条评论的词所对应的id。 y为onehot编码: pos-[1, 0], neg-[0, 1]

"""

# 建立Text-CNN模型
# 配置参数
class CONFIG():
    update_w2v = True    # 是否在训练中更新w2v
    vocab_size = 59290    # 词汇量，与word2id中的词汇量一致
    n_class = 2  # 分类数：分别为pos和neg
    max_sen_len = 70  # 句子最大长度  75?
    embedding_dim = 50  # 词向量维度
    batch_size = 100  # 批处理尺寸
    n_hidden = 256  # 隐藏层节点数
    n_epoch = 10  # 训练迭代周期，即遍历整个训练样本的次数
    opt = 'adam'  # 训练优化器：adam或者adadelta
    learning_rate = 0.001  # 学习率；若opt=‘adadelta'，则不需要定义学习率
    drop_keep_prob = 0.5  # dropout层，参数keep的比例
    num_filters = 256  # 卷积层filter的数量
    kernel_size = 4  # 卷积核的尺寸；nlp任务中通常选择2,3,4,5
    print_per_batch = 100  # 训练过程中,每100词batch迭代，打印训练信息
    save_dir = './checkpoints/'  # 训练模型保存的地址
    train_path = './data/train.txt'
    dev_path = './data/validation.txt'
    test_path = './data/test.txt'
    word2id_path = './data/word_to_id.txt'
    pre_word2vec_path = './data/wiki_word2vec_50.bin'
    corpus_word2vec_path = './data/corpus_word2vec.txt'

# 定义时间函数，供计算模型迭代时间使用
def time_diff(start_time):
    """当前距初始时间已花费的时间"""
    end_time = time.time()
    diff = end_time - start_time
    return timedelta(seconds=int(round(diff)))
#timedelta是用于对间隔进行规范化输出，间隔10秒的输出为：00:00:10

# 建立Text-CNN模型
class TextCNN(object):
    def __int__(self, config, embeddings=None):
        self.update_w2v = config.update_w2v
        self.vocab_size = config.vocab_size
        self.n_class = config.n_class
        self.max_sen_len = config.max_sen_len
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.num_filters = config.num_filters
        self.kernel_size = config.kernel_size
        self.n_hidden = config.n_hidden
        self.n_epoch = config.n_epoch
        self.opt = config.opt
        self.learning_rate = config.learning_rate
        self.drop_keep_prob = config.drop_keep_prob

        self.x = tf.placeholder(tf.int32, [None, self.max_sen_len], name='x')
        self.y = tf.placeholder(tf.int32, [None, self.n_class], name='y')

        if embeddings is not None:
            # tf.Variable(initializer,name),参数initializer是初始化参数，name是可自定义的变量名称
            self.word_embeddings = tf.Variable(embeddings, dtype=tf.float32, trainable=self.update_w2v)




