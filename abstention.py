import json
from engine import Engine
from tqdm import tqdm
import numpy as np
from scipy.special import softmax
import tiktoken
import argparse
from sklearn.metrics import brier_score_loss

def qa_to_prompt(query, context, choices, schema, answer=''):
    context = context.replace('“', '"').replace('”', '"').replace('’', "'")
    if schema == 'base':
        prompt = '{}\n\nQ: {}\nChoices: {}\nA: {}'.format(context, query, choices, answer)
    elif schema == 'opin':
        context = context.replace('"', "")
        prompt = 'Bob said, "{}"\n\nQ: {} in Bob\'s opinion?\nChoices: {}\nA: {}'.format(context, query[:-1], choices, answer)
    elif schema == 'attr':
        prompt = '{}\n\nQ:{} based on the given text?\nChoices: {}\nA: {}'.format(context, query[:-1], choices, answer)
    elif schema == 'instr':
        prompt = '{}\n\nQ: {}\nChoices: {}\nA: {}'.format(context, query, choices, answer)
    elif schema == 'instr+opin':
        context = context.replace('"', "")
        prompt = 'Bob said, "{}"\n\nQ: {} in Bob\'s opinion?\nChoices: {}\nA: {}'.format(context, query[:-1], choices, answer)
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./datasets/realtime_qa/realtime_qa_data.json", type=str)
    parser.add_argument("--demo_path", default="./datasets/realtime_qa/realtime_qa_demo_data.json", type=str)
    parser.add_argument("--engine", default="text-davinci-003", type=str)
    parser.add_argument("--schema", default="base", type=str, help="Choose from the following prompting templates: base, attr, instr, opin, instr+opin.")
    parser.add_argument("--demo_mode", default="none", help="Choose from the following demonstrations: none, original.")
    parser.add_argument("--log_path", default='', type=str)
    args = parser.parse_args()
    with open(args.data_path, 'r') as fh:
        test_data = json.load(fh)
    with open(args.demo_path, 'r') as fh:
        demo_data = json.load(fh)
    engine = Engine(args.engine)
    tokenizer = tiktoken.encoding_for_model(args.engine)
    abs_golds, abs_probs, preds, golds = [], [], [], []
    for d in tqdm(test_data):
        context, question, choices, answer = d['context'], d['question'], d['choices'], d['answer']
        probs = []
        for choice in choices.split(';'):
            choice = choice.strip()
            assert len(choice) > 0
            prompt = ''
            if args.schema in ('instr', 'instr+opin'):
                prompt = 'Instruction: answer a question based on the provided input-output pairs.\n\n'
            if args.demo_mode == 'original':
                for demo in demo_data:
                    prompt += (qa_to_prompt(demo['question'], demo['context'], demo['choices'], args.schema, answer=demo['answer']) + '\n\n')
            choice = choice.strip() + '.'
            prompt += qa_to_prompt(question, context, choices, args.schema)
            prompt = prompt + choice
            if engine.check_prompt_length(prompt):
                continue
            num_tokens = len(tokenizer.encode(' ' + choice))
            prob = engine.get_prob(prompt, num_tokens)
            if prob is not None:
                probs.append(prob)
        if len(probs) != len(choices.split(';')):
            continue
        choice_probs = softmax(np.array(probs))
        choices = [s.strip() for s in choices.split(';')]
        pred = choices[probs.index(max(probs))]
        d['pred'] = pred
        d['probs'] = choice_probs.tolist()
        abs_gold = 1 if answer == 'I don\'t know' else 0
        abs_golds.append(abs_gold)
        abs_probs.append(choice_probs.tolist()[-1])
        preds.append(pred)
        golds.append(answer)
    # Evaluation
    has_ans_correct, no_ans_correct, has_ans_wrong, no_ans_wrong = 0, 0, 0, 0
    for pred, gold in zip(preds, golds):
        if pred == gold:
            if gold != 'I don\'t know':
                has_ans_correct += 1
            else:
                no_ans_correct += 1
        else:
            if gold != 'I don\'t know':
                has_ans_wrong += 1
            else:
                no_ans_wrong += 1
    hasans_acc = has_ans_correct / (has_ans_correct + has_ans_wrong)
    noans_acc = no_ans_correct / (no_ans_correct + no_ans_wrong)
    acc = (has_ans_correct + no_ans_correct) / (has_ans_correct + has_ans_wrong + no_ans_correct + no_ans_wrong)
    brier = brier_score_loss(np.array(abs_golds), np.array(abs_probs))
    print("HasAns Acc {}, NoAns Acc {}, Acc {}, Brier {}.".format(hasans_acc, noans_acc, acc, brier))
    if args.log_path:
        with open(args.log_path, 'w') as fh:
            json.dump(test_data, fh)

if __name__ == '__main__':
    main()