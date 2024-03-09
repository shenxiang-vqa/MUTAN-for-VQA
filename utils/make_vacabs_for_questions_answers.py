import os
import argparse
import numpy as np
import json
import re
from collections import defaultdict
from tqdm import tqdm

def make_vocab_questions(input_dir):
    r"The purpose of fuctional is to gain a vocab of question"
    """
    This is done in the following steps:
    
    step1 : Get all the words to the question.Then do the splitting of words
    
    step2 : Count the frequency of occurrence of each word
    
    step3 : disjunction words
    
    step4 : Create an indexed dictionary with the most frequent words in the first position
    """
    vocab_set = set()
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    """
    text = "Hello! How are you? I'm fine, thank you."
    sentences = SENTENCE_SPLIT_REGEX.split(text)
    out:['Hello', '!', ' How are you', '?', " I'm fine, thank you", '.', '']
    """
    question_length = []
    datasets = os.listdir(input_dir)#["v2_OpenEnded_mscoco_test2015_questions.json","v2_OpenEnded_mscoco_test-dev2015_questions.json","v2_OpenEnded_mscoco_train2014_questions.json","v2_OpenEnded_mscoco_val2014_questions.json"]
    for dataset in datasets:
        print("Start processing document ing...")
        with open(os.path.join(input_dir+'/'+dataset))as f: # f is a object of document
            questions = json.load(f)["questions"]#Load json file
        set_question_length = [None] * len(questions)
        for iquestion, question in tqdm(enumerate(questions)):
            words = SENTENCE_SPLIT_REGEX.split(question["question"].lower())
            words = [w.strip() for w in words if len(w.strip()) > 0]
            """
            s = "  hello world   "
            result = s.strip()
            print(result)  # 输出："hello world"
            """
            vocab_set.update(words)
            set_question_length[iquestion] = len(words)
            if iquestion % 40000 == 0:
                print("It's been dealt with.{}/{}".format(iquestion, len(questions)))
        question_length += set_question_length
    vocab_list = list(vocab_set)
    vocab_list.sort()
    vocab_list.insert(0, '<pad>')
    vocab_list.insert(1, '<unk>')
    with open(os.path.join(input_dir, 'vocab_questions.txt'), 'w') as f:
        f.writelines([w + '\n' for w in vocab_list])
    print('Make vocab for questions')
    print('The number of total words of questions: %d' % len(vocab_set))
    print('Maximum length of question: %d' % np.max(question_length))
def make_vocab_answers(input_dir,n_answers):
    """
    text = "hello world hello python world"

    # 创建一个 defaultdict 对象，初始值为 0
    word_counts = defaultdict(lambda: 0)

    # 统计单词出现的次数
    for word in text.split():
        word_counts[word] += 1

    # 输出统计结果
    for word, count in word_counts.items():
        print(f"{word}: {count}")
    out:hello: 2
        world: 2
        python: 1
    """
    answers = defaultdict(lambda:0)
    datasets = os.listdir(input_dir)
    for dataset in datasets:
        with open(input_dir+'/'+dataset) as f:
            annotations = json.load(f)["annotations"]
        for i,annotation in tqdm(enumerate(annotations)):
            if i % 40000 == 0:
                print("It's been dealt with.{}/{}".format(i, len(annotations)))
            for answer in annotation['answers']:
                word = answer['answer']
                if re.search(r"[^\w\s]", word):
                    continue
                answers[word] += 1
    answers = sorted(answers, key=answers.get, reverse=True)
    assert ('<unk>' not in answers)
    top_answers = ['<unk>'] + answers[:n_answers - 1]  # '-1' is due to '<unk>'

    with open(os.path.join(input_dir, 'vocab_answers.txt'), 'w') as f:
        f.writelines([w + '\n' for w in top_answers])

    print('Make vocabulary for answers')
    print('The number of total words of answers: %d' % len(answers))
    print('Keep top %d answers into vocab' % n_answers)
def main(args):
    input_dir = args.input_dir
    n_answers = args.n_answers
    make_vocab_questions(input_dir+'/questions')
    make_vocab_answers(input_dir+'/annotations', n_answers)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='D:\data\VQA_data\VQAv2',
                        help='directory for input questions and answers')
    parser.add_argument('--n_answers', type=int, default=1000,
                        help='the number of answers to be kept in vocab')
    args = parser.parse_args()
    main(args)