import os 
import json
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



def load_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open (os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                docs.extend(paragraphs)
    return docs


def build_index(docs):

    def cut_words(text):
        return ' '.join(jieba.cut(text))
    
    cut_docs = [cut_words(doc) for doc in docs]
    my_stop_words = ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这']
    
    vectorizer = TfidfVectorizer(max_features=100, stop_words=my_stop_words)
    
    

    tfidf_matrix = vectorizer.fit_transform(cut_docs)
    print(vectorizer.get_feature_names_out())


    return vectorizer, tfidf_matrix


def retrieve(query, docs, vectorize, tfidf_matrix, top_k=3):
    cut_query = ' '.join(jieba.cut(query))

    query_vec = vectorizer.transform([cut_query])

    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            'content': docs[idx],
            'score': float(similarities[idx])
        })
    return results

def generate_answer(query, retrieved_docs):
    """模拟生成回答:拼装成Prompt, 打印出来供你手动复制给大模型"""
    context = '\n\n'.join([doc['content'] for doc in retrieved_docs])
    prompt = f"""
  【任务】 请根据以下参考资料回答问题。

  【参考资料】
{context}

  【问题】
{query}

【回答】
"""
    return prompt


if __name__ == '__main__':
    print("正在加载文档...")
    docs = load_documents('./data')
    if not docs:
        print("请现在 data 文件夹下放入至少一个 .txt 文件！")
        exit()
    print(f"共加载{len(docs)}个段落")


    print("正在建立索引...")
    vectorizer, tfidf_matrix = build_index(docs)
    print("索引建立完成")

    while True:
        query = input("\n 请输入你的问题(输入exit退出) : ")
        if query.lower() == 'exit':
            break


        results = retrieve(query, docs, vectorizer, tfidf_matrix)


        prompt = generate_answer(query, results)
        

        print("\n" + "="*50)
        print("请将一下Prompt复制到任意大模型(如DeepSeek/通义千问) 获取回答：")
        print("="*50)


        print("\n 召回段落摘要：")
        for i, doc in enumerate(results):
            print(f" {i+1}.(相似度：{doc['score']:.4f}){doc['content'][:50]}...")