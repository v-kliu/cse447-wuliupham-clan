#!/usr/bin/env python
import os
import string
import random
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        file_name = "./src/training_data_one.txt"
        data = []
        with open(file_name) as file:
            for line in file:
                inp = line[:-1] # we want to cut new line character but keep last char so last char - 1 still has data
                data.append(inp)
        return data

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        self.nextCharCountsDict = {} # get a dict to store char -> next char counts
        for line in data:
            for idx in range(len(line) - 1):
                char = line[idx]
                nextChar = line[idx + 1]
                # make sure char has a dict inside it
                if char not in self.nextCharCountsDict:
                    self.nextCharCountsDict[char] = {}
                # get curr count or default to 0, increment by 0
                self.nextCharCountsDict[char][nextChar] = self.nextCharCountsDict[char].get(nextChar, 0) + 1


    def run_pred(self, data):
        # your code here
        preds = []
        # all_chars = string.ascii_letters <- keeping so we know to implement multilanguage stuff later
        for prompt in data:
            if not prompt or not prompt[len(prompt) - 1] in self.nextCharCountsDict:
                preds.append("aei") # just append 3 vowels which are likely in english as base case
                continue 
            
            lastChar = prompt[len(prompt) - 1] # curr logic only cares about last char (will be improved LOL)
            lastCharDict = self.nextCharCountsDict[lastChar] # we know it has to exist based on our case above
            topThreeCountsSorted = sorted(lastCharDict.items(), key = lambda item: item[1], reverse=True)[:3]
            # topThreeCountsSorted stores 3 tuples of len 2, need to just append the literal chars
            preds.append("".join([item[0] for item in topThreeCountsSorted]))
        # return our predictions
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            json.dump(self.nextCharCountsDict, f)

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        model = MyModel()
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            model.nextCharCountsDict = json.load(f)
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
