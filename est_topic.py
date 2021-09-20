import numpy as np
import math
from collections import Counter
import argparse

import gen_crp

from typing import List

def_n_vocab=gen_crp.def_n_vocab
def_alpha=gen_crp.def_alpha
def_beta=gen_crp.def_beta

def est_topic(
        topics:List[int],
        documents:List[List[int]],
        alpha:float = def_alpha,
        beta:float = def_beta,
        n_vocab:int = def_n_vocab,
) -> List[int]:

    n_doc = len(topics)
    assert len(documents)==n_doc, "# of topics and documents is different"

    rng = np.random.default_rng()
    
    est_topics = [0]*len(topics)
    n_loop = 0
    n_consecutive_unchanged = 0
    while True:
        for i_doc in range(n_doc):
            other_topics = {
                j_doc:topic_id for j_doc, topic_id in enumerate(est_topics)
                if topic_id>0 and i_doc!=j_doc
            }
            
            counter_other_topics = Counter(other_topics.values()) 
            i_topic_to_topic_id = [topic_id for topic_id
                                   in counter_other_topics.keys()]
            n_doc_for_each_topic = [n_doc for n_doc
                                    in counter_other_topics.values()]
            n_topic = len(n_doc_for_each_topic)
            probs = [n_doc_for_this_topic
                     for n_doc_for_this_topic in n_doc_for_each_topic]
            this_doc_length = len(documents[i_doc])
            n_word_for_this_doc = Counter(documents[i_doc])
            for i_topic, topic_id in enumerate(i_topic_to_topic_id):
                words_for_this_topic = []
                for j_doc in other_topics.keys():
                    if other_topics[j_doc]==topic_id:
                        words_for_this_topic.extend(documents[j_doc])
                n_all_word_for_this_topic = len(words_for_this_topic)
                N = n_all_word_for_this_topic+beta*n_vocab
                log_prob = math.lgamma(N)
                log_prob -= math.lgamma(N+this_doc_length)
                n_this_word_for_this_topic = Counter(words_for_this_topic)
                for vocab, n_this_word_for_this_doc in n_word_for_this_doc.items():
                    n_this_word_for_other_doc = n_this_word_for_this_topic[vocab]
                    N = n_this_word_for_other_doc+beta
                    log_prob += math.lgamma(N+n_this_word_for_this_doc)
                    log_prob -= math.lgamma(N)
                probs[i_topic] *= math.exp(log_prob)    
            probs.append(alpha)
            log_prob = math.lgamma(beta*n_vocab)
            log_prob -= math.lgamma(this_doc_length+beta*n_vocab) 
            for vocab, n_this_word_for_this_doc in n_word_for_this_doc.items():
                log_prob += math.lgamma(n_this_word_for_this_doc+beta)
                log_prob -= math.lgamma(beta)
            probs[-1] *= math.exp(log_prob)

            denom = sum(probs)
            probs = [ unnormed/denom for unnormed in probs ]

            i_topic = rng.choice( len(probs), p=probs )
            orig_topic = est_topics[i_doc]
            if i_topic<n_topic:
                est_topics[i_doc] = i_topic_to_topic_id[i_topic]
            else:
                est_topics[i_doc] = (
                    max(i_topic_to_topic_id)+1
                    if n_topic>0 else 1
                )
            print(i_doc, n_loop, other_topics, probs)
                
        n_loop += 1
        if est_topics[i_doc]==orig_topic:
            n_consecutive_unchanged += 1
        else:
            n_consecutive_unchanged += 0
        if n_consecutive_unchanged>5*n_doc or n_loop>100*n_doc:
            break

    return est_topics
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "doc_len",
        nargs="+",
        type=int,
        help="list of each document length",
    )

    args = parser.parse_args()
    doc_len = args.doc_len

    topics, documents = gen_crp.gen_crp(
        doc_len=doc_len,
    )

    est_topics = est_topic(
        topics=topics,
        documents=documents,
    )

    print(topics)
    print(est_topics)
    
    
