import numpy as np
from collections import Counter

from typing import List, Tuple

def_n_vocab=100
def_alpha=1
def_beta=1

def gen_crp(
        doc_len:List[int],
        alpha:float = def_alpha,
        beta:float = def_beta,
        n_vocab:int = def_n_vocab,
) -> Tuple[List[int],List[List[int]]]:

    rng = np.random.default_rng()

    documents = []
    topics = []
    vocab_dist = {}
    for i_doc, length in enumerate(doc_len):
        n_doc_for_topic = Counter(topics)
        n_topic = len(n_doc_for_topic)
        denom = len(documents)+alpha
        prob = [n_doc/denom for n_doc in n_doc_for_topic.values()]
        prob.append(alpha/denom)
        topic_index = rng.choice(n_topic+1,p=prob)

        topics.append(topic_index)
        if topic_index==n_topic:
            vocab_dist[n_topic] = rng.dirichlet([beta]*n_vocab)

        documents.append([
            rng.choice(
                n_vocab,
                p=vocab_dist[topic_index],
            ) for _ in range(length)
        ])

    return topics, documents

if __name__ == '__main__':

    topics, documents = gen_crp(
        doc_len=[1,1,1,1,2,2,2,3,3,4,4,5,5,6,6,7,7,8,8],
    )

    for topic, document in zip(topics, documents):
        print( topic, document )
