"coding = utf-8"
# 删除语料中的生僻字
import codecs
import progressbar
import numpy as np
# import mycrnn_pc.config.cfg as cfg

import re,string

punctuation = """！，。？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"""

#np.random.seed(0)
dictfile = "dict5990.txt"
corpus_file = "jingji.txt"
output = "filted_jian_sentences.txt"
filiter_txt="fileter.txt"
dict = []
max_row_len = 20


#mode = 'split'
mode = 'filter'

if mode == 'filter':
    with codecs.open(dictfile, mode='r', encoding='utf-8') as f:
        # 按行读取字典
        for line in f:
            # 当前行单词去除结尾，为了正常读取空格，第一行两个空格
            word = line.strip('\r\n')
            # 只要没超出上限就继续添加单词
            dict.append(word)

    corpus = []
    with codecs.open(corpus_file, mode='r', encoding='utf-8') as f:
        # 按行读取语料
        print('正在读取语料...')
        for line in f:
            corpus.append(line)

    with codecs.open(output, mode='w', encoding='utf-8') as output:
        widgets = ["正在过滤语料: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(corpus), widgets=widgets).start()
        np.random.shuffle(corpus)
        f = codecs.open(filiter_txt, mode='w', encoding='utf-8')
        for i, line in enumerate(corpus):
            line = line.strip().replace(" ", "")  # 去除空格
            sentence = ''
            for each in line:
                # 去除生僻字
                try:
                    dict.index(each)
                    sentence += each
                except:
                    pass
                    print('i',i)
                    #print("字典不包含{}, 忽略".format(each，i))
                    print('a={0} b={1}'.format(each,i))
                    f.write(each+"\t"+str(i+1)+"\n")
            if sentence != '\n' and sentence != ' ' and sentence!='':  # 不写空行
                output.write(sentence+'\n')
            pbar.update(i)
        pbar.finish()

elif mode == 'split':
    corpus = []
    with codecs.open(output, mode='r', encoding='utf-8') as f:
        # 按行读取语料
        print('正在读取语料...')
        for line in f:
            corpus.append(line)

    with codecs.open('split_sentences.txt', mode='w', encoding='utf-8') as output:
        widgets = ["正在分行语料: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(corpus), widgets=widgets).start()

        for i, line in enumerate(corpus):
            row = line
            # if np.random.randint(0, 1000) < 2:  # 0.2%的概率加空白行
            #     output.write('\n')
                # 对大于max_row_len的句子进行分行，直到最后小于max_row_len
            while len(row) > max_row_len:
                # 长句子分行
                # 偶尔出现单字
                spliter = np.random.random_integers(1, max_row_len-1)
                output.write(row[0:spliter] + '\n')
                # if np.random.randint(0, 1000) < 2:  # 0.2%的概率加空白行
                #     output.write('\n')
                row = row[spliter:]
                
            #每行会含有一个换行符，去掉换行符    
            if len(row)>=6:
                re_punctuation = "[{}]+".format(punctuation)
                row = re.sub(re_punctuation, "", row)
                
               
                if len(row)>=6:
                    output.write(row)
                    pbar.update(i)
        pbar.finish()
        
        
        
        

#语料文件在切分时，长度为5-15        
'''
elif mode == 'split':
    corpus = []
    with codecs.open(output, mode='r', encoding='utf-8') as f:
        # 按行读取语料
        print('正在读取语料...')
        for line in f:
            corpus.append(line)

    with codecs.open('sentences.txt', mode='w', encoding='utf-8') as output:
        widgets = ["正在分行语料: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(corpus), widgets=widgets).start()

        for i, line in enumerate(corpus):
            row = line
            # if np.random.randint(0, 1000) < 2:  # 0.2%的概率加空白行
            #     output.write('\n')
                # 对大于max_row_len的句子进行分行，直到最后小于max_row_len
            while len(row) > max_row_len:
                # 长句子分行
                # 偶尔出现单字
                spliter = np.random.random_integers(6, max_row_len-1)
                output.write(row[0:spliter] + '\n')
                # if np.random.randint(0, 1000) < 2:  # 0.2%的概率加空白行
                #     output.write('\n')
                row = row[spliter:]
                
            #每行会含有一个换行符，去掉换行符    
            if len(row)>=6:
                re_punctuation = "[{}]+".format(punctuation)
                row = re.sub(re_punctuation, "", row)
                
               
                if len(row)>=6:
                    output.write(row)
                    pbar.update(i)
        pbar.finish()        
        
'''        
        