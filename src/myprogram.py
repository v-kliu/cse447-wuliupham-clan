#!/usr/bin/env python
import os
import random
import json
import re
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class MyModel:
    """
     n-gram language model of characters with interpolation
    basically the same thing from assignment 1b but for characters instead of words
    uses up to 5-gram context to predict next character
    trained on 10 language datasets and sentences, see info.txt
    """

    def __init__(self):
        # max n-gram order, so we look at up to 4 characters of context
        self.N = 5

        # same structure as WordNGramLM from hw1b
        # ngramCounts[n] is a dict: context -> {next_char: count}
        # contextCounts[n] is a dict: context -> total count
        # e.g. ngramCounts[3]["th"] = {"e": 500, "a": 200, ...}
        self.ngramCounts = {}
        self.contextCounts = {}
        for n in range(1, self.N + 1):
            self.ngramCounts[n] = {}
            self.contextCounts[n] = {}

        # interpolation weights just like assignment 1b exercise 3.3
        # P(next) = lambda1*P_unigram + lambda2*P_bigram + ... + lambda5*P_5gram
        # higher n = more context = better prediction but sparser data
        # so we weight higher orders more when they have data
        self.lambdas = [0.05, 0.10, 0.20, 0.25, 0.40]


    @classmethod
    def preprocess_line(cls, line):
        """
        each line looks like "123\tThe actual sentence here"
        so we strip the leading number + tab
        """
        # strip leading number and tab that leipzig corpus adds
        line = re.sub(r'^\d+\t', '', line)
        line = line.strip()
        return line


    @classmethod
    def load_training_data(cls):
        """
        load all training data from the training_data folder
        each file is a different language ~10k sentences each
        also loads the og training_data_one.txt
        """
        all_data = []

        # load the multilingual datasets
        data_dir = "./src/training_data"
        if os.path.isdir(data_dir):
            for fname in os.listdir(data_dir):
                fpath = os.path.join(data_dir, fname)
                if os.path.isfile(fpath) and fname.endswith('.txt'):
                    print(f'  loading {fname}...')
                    with open(fpath, encoding='utf-8') as f:
                        for line in f:
                            cleaned = cls.preprocess_line(line)
                            if cleaned:
                                all_data.append(cleaned)

        # don't use old data of uw > oregon dataset

        print(f'  total training sentences: {len(all_data)}')
        return all_data


    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname, encoding='utf-8') as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data


    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write('{}\n'.format(p))


    def run_train(self, data, work_dir):
        """
        build n-gram counts from training data
        basically the same as WordNGramLM.fit() from hw1b
        but instead of words we use characters, and instead of tuples we use strings as context

        for each character in the text, we look at the previous n-1 characters (the context)
        and count how many times this character follows that context
        """

        for line in data:
            for i in range(len(line)):
                next_char = line[i]
                # go through each n-gram order, same loop structure as hw
                for n in range(1, self.N + 1):
                    if n == 1:
                        # unigram - no context, just overall char frequency
                        context = ""
                    else:
                        # need n-1 chars of history
                        context_start = i - (n - 1)
                        if context_start < 0:
                            continue  # not enough context yet skip
                        context = line[context_start:i]

                    # same pattern as self.ngramCounts[context][targetWord] += 1 from hw
                    if context not in self.ngramCounts[n]:
                        self.ngramCounts[n][context] = {}
                        self.contextCounts[n][context] = 0

                    if next_char not in self.ngramCounts[n][context]:
                        self.ngramCounts[n][context][next_char] = 0

                    self.ngramCounts[n][context][next_char] += 1
                    self.contextCounts[n][context] += 1


    def _get_interpolated_scores(self, prompt):
        """
        compute interpolated probabilities for the next character
        this is  same math from WordNGramLMWithInterpolation.eval_perplexity

        from hw1b:
        interpolatedProb = 0.0
        for n_idx, m in enumerate(self.models):
            lambda_val = self.lambdas[n_idx]
            ngram_count = m.ngramCounts.get(curr_context, {}).get(target, 0)
            context_count = m.contextCounts.get(curr_context, 0)
            if context_count > 0:
                interpolatedProb += lambda_val * (ngram_count / context_count)

        we do the same thing but for every possible next character
        instead of just one target word
        """
        scores = defaultdict(float)

        for n in range(1, self.N + 1):
            n_idx = n - 1  # index into lambdas list
            lambda_val = self.lambdas[n_idx]

            if n == 1:
                context = ""
            else:
                # grab last n-1 chars as context
                if len(prompt) < n - 1:
                    continue # prompt too short for this order
                context = prompt[-(n - 1):]

            # check if we ever saw this context  (same as m.ngramCounts.get(curr_context, {}))
            if context not in self.ngramCounts[n]:
                continue  # never seen this context, contributes 0

            charCounts = self.ngramCounts[n][context]
            contextCount = self.contextCounts[n][context]

            if contextCount == 0:
                continue

            # add weighted prob for each possible next char
            # same formula: lambda_val * (ngram_count / context_count)
            for char, count in charCounts.items():
                scores[char] += lambda_val * (count / contextCount)

        return scores


    def run_pred(self, data):
        """
        for each test prompt use interpolated n-gram scores to pick top 3 chars
        """
        preds = []

        for prompt in data:
            try:
                scores = self._get_interpolated_scores(prompt)

                if scores:
                    # sort by score descending take top 3  (same as how we sorted in old version)
                    topCharsSorted = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
                    pred = "".join([ch for ch, _ in topCharsSorted])
                else:
                    # fallback if we got nothing
                    pred = "e a"

                # pad to 3 chars if we dont have enough predictions
                fallbacks = "e atonirsl"
                idx = 0
                while len(pred) < 3 and idx < len(fallbacks):
                    if fallbacks[idx] not in pred:
                        pred += fallbacks[idx]
                    idx += 1

                preds.append(pred[:3])

            except Exception:
                # fall back
                preds.append("e a")

        return preds


    def save(self, work_dir):
        """
        save model to json, have to convert to regular dicts bc json doesnt like defaultdict
        """
        save_data = {
            'N': self.N,
            'lambdas': self.lambdas,
            'ngramCounts': {},
            'contextCounts': {}
        }

        for n in range(1, self.N + 1):
            n_str = str(n)
            save_data['ngramCounts'][n_str] = {}
            save_data['contextCounts'][n_str] = {}

            for context, charCounts in self.ngramCounts[n].items():
                save_data['ngramCounts'][n_str][context] = dict(charCounts)

            for context, total in self.contextCounts[n].items():
                save_data['contextCounts'][n_str][context] = total

        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False)

        print(f'  model saved, stats:')
        for n in range(1, self.N + 1):
            print(f'    {n}-gram: {len(self.ngramCounts[n])} unique contexts')


    @classmethod
    def load(cls, work_dir):
        """
        load saved model back from json checkpoint
        """
        model = MyModel()

        with open(os.path.join(work_dir, 'model.checkpoint'), encoding='utf-8') as f:
            save_data = json.load(f)

        model.N = save_data['N']
        model.lambdas = save_data['lambdas']

        # rebuild the dicts from json
        model.ngramCounts = {}
        model.contextCounts = {}

        for n in range(1, model.N + 1):
            model.ngramCounts[n] = {}
            model.contextCounts[n] = {}
            n_str = str(n)

            if n_str in save_data['ngramCounts']:
                for context, charCounts in save_data['ngramCounts'][n_str].items():
                    model.ngramCounts[n][context] = {}
                    for char, count in charCounts.items():
                        model.ngramCounts[n][context][char] = count

            if n_str in save_data['contextCounts']:
                for context, total in save_data['contextCounts'][n_str].items():
                    model.contextCounts[n][context] = total

        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
