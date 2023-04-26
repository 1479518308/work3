import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
import os
import re
from tqdm import tqdm
import jieba
import math
import thulac
import pandas as pd
import numpy as np
import sys
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
jieba.load_userdict("cn_stopwords.txt")
origin_path = (r".\jyxstxtqj_downcc.com")

def lda(data,data_cut,n_topics):
    tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                    max_features=500,
                                    stop_words='english',
                                    max_df=0.5,
                                    min_df=10,analyzer="char")
    tf = tf_vectorizer.fit_transform(data_cut)
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                    learning_method='batch',
                                    learning_offset=50,
                                    doc_topic_prior=0.1,
                                    topic_word_prior=0.01,
                                    random_state=0)
    lda.fit(tf)

    n_top_words = 15
    tf_feature_names = tf_vectorizer.get_feature_names()
    topic_word = print_top_words(lda, tf_feature_names, n_top_words)
    topics = lda.transform(tf)

    topic = []
    for t in topics:
        topic.append("Topic #" + str(list(t).index(np.max(t))))
    data['概率最大的主题序号'] = topic
    data['每个主题对应概率'] = list(topics)

    data.to_csv("data_topic.csv",index=False,encoding='utf-8-sig')
    return lda,tf,data,topics

def getpath(origin_path):#获取全部txt文件
    names = os.listdir(origin_path)
    pathlist = []
    labellist = []
    for name in tqdm(names):
        if name.lower().endswith('.txt'):
            path = os.path.join(origin_path, name)
            pathlist.append(path)
            labellist.append(name.replace('.txt','',1))
    return pathlist,labellist

def getcontext(path):
    with open(path, "r", encoding="ANSI") as file:
        filecontext = file.read()
        seg_list = jieba.lcut_for_search(filecontext)
        for num in range(len(seg_list)):
            seg_list[num] = nonoise(seg_list[num])
        seg_list = [i for i in seg_list if i != '']
    return seg_list

def nonoise(filecontext):
    english = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;「<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    str1 = u'[②③④⑤⑥⑦⑧⑨⑩_“”、。《》！，：；？‘’」「…『』（）<>【】．·.—*-~﹏]'
    filecontext = re.sub(english, '', filecontext)
    filecontext = re.sub(str1, '', filecontext)
    filecontext = filecontext.replace("\n", '')
    filecontext = filecontext.replace(" ", '')
    filecontext = filecontext.replace(u'\u3000', '')
    return filecontext

def get_data():
    pathlist, labellist = getpath(origin_path)
    data = {}
    count = 0
    for i in range(len(pathlist)):
        seg_list = getcontext(pathlist[i])
        label = labellist[i]
        cut = int(len(seg_list)//13)
        start = 0
        for j in range(13):
            sectioncontent = ''
            section_cutci = ''
            section_cutzi = ''
            section = seg_list[start:start+500]
            for t in section:
                sectioncontent += t
                section_cutci += t
                section_cutci += ' '
            for m in sectioncontent:
                section_cutzi += m
                section_cutzi += ' '
            start += cut
            data["section_count"] = j + count
            data['section'] = sectioncontent
            data['label'] = label
            data['section_cutci'] = section_cutci
            data['section_cutzi'] = section_cutzi
            data = pd.DataFrame(data,index = [1])
            print(data)
            try:
                data_ci = pd.concat([data_ci, data])

            except:
                data_ci = data
        count += 13
    data_ci = data_ci.reset_index(drop=True)
    print(data_ci)
    data_ci.to_csv("data.csv",index=False,encoding='utf-8-sig')

def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword

def main():
    data = pd.read_csv("data.csv")
    data_ci = data['section_cutci']
    data_zi = data['section_cutzi']
    n_topics = 16
    lda1,tf,data,topics = lda(data,data_ci,n_topics)
    #lda1, tf, data, topics = lda(data, data_zi)
    perplesity(tf)
    svm(data,topics)

def drawmain():
    data = pd.read_csv("data.csv")
    data_ci = data['section_cutci']
    data_zi = data['section_cutzi']
    score = []
    x = []
    for i in range(10):
        n_topics = i*2+30
        #lda1, tf, data, topics = lda(data, data_ci, n_topics)
        lda1, tf, data, topics = lda(data, data_zi,n_topics)
        # perplesity(tf)
        score.append(svm(data, topics))
        x.append(n_topics)
    plt.plot(x,score)
    plt.xlabel("n_topics")
    plt.ylabel("score")
    plt.show()


def test(strtest,lda):
    str = transform(strtest)
    tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                    max_features=500,
                                    stop_words='english',
                                    max_df=0.5,
                                    min_df=10)
    tf = tf_vectorizer.fit_transform(str)
    topics = lda.transform(tf)
    for t in topics:
        print("Topic #" + str(list(t).index(np.max(topics))))

def transform(strtest):
    seg_list = jieba.lcut_for_search(strtest)
    for num in range(len(seg_list)):
        seg_list[num] = nonoise(seg_list[num])
    seg_list = [i for i in seg_list if i != '']
    return seg_list

def perplesity(tf):
    plexs = []
    scores = []
    n_max_topics = 50
    for i in range(1, n_max_topics):
        print(i)
        lda = LatentDirichletAllocation(n_components=i)
        lda.fit(tf)
        plexs.append(lda.perplexity(tf))
        scores.append(lda.score(tf))
    n_t = 49  # 区间最右侧的值。注意：不能大于n_max_topics
    x = list(range(1, n_t + 1))
    plt.plot(x, plexs[0:n_t])
    plt.xlabel("number of topics")
    plt.ylabel("perplexity")
    plt.show()
    plt.plot(x, scores[0:n_t])
    plt.xlabel("number of topics")
    plt.ylabel("scores")
    plt.show()

def svm(data,topics):
    X = pd.DataFrame(topics)
    print(type(X),len(X))
    Y = data['label']
    print(type(Y),len(Y))

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=42)
    # 恢复分割后的索引
    for i in [Xtrain, Xtest, Ytrain, Ytest]:
        i.index = range(i.shape[0])

    clf = DecisionTreeClassifier(class_weight='balanced', random_state=37)
    clf = clf.fit(Xtrain, Ytrain)  # 拟合训练集
    score_c = clf.score(Xtest, Ytest)  # 输出测试集准确率
    print(score_c)
    return score_c

if __name__ == "__main__":
    #get_data()
    #main()
    drawmain()