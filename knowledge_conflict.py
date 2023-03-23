import json
import re
import argparse
from tqdm import tqdm
from engine import Engine
from evaluation import get_score


def qa_to_prompt(query, context, schema, demos=[], num_demos=16):
    def get_prompt(query, context, schema, answer=''):
        if schema == 'base':
            prompt = '{}\nQ:{}\nA:{}'.format(context, query, answer)
        elif schema == 'opin':
            context = context.replace('"', "")
            prompt = 'Bob said "{}"\nQ: {} in Bob\'s opinion?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'instr+opin':
            context = context.replace('"', "")
            prompt = 'Bob said "{}"\nQ: {} in Bob\'s opinion?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'attr':
            prompt = '{}\nQ:{} based on the given tex?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'instr':
            prompt = '{}\nQ:{}\nA:{}'.format(context, query, answer)
        return prompt
    prompt = ''
    if schema in ('instr', 'instr+opin'):
        prompt = 'Instruction: read the given information and answer the corresponding question.\n\n'
    for demo in demos[-num_demos:]:
        answer = demo['answer'] if isinstance(demo['answer'], str) else demo['answer'][0]
        demo_prompt = get_prompt(demo['question'], demo['context'], schema=schema, answer=answer)
        prompt = prompt + demo_prompt + '\n\n'
    prompt = prompt + get_prompt(query, context, schema=schema)
    return prompt

def eval(pred_answers, orig_answers, gold_answers):
    em, ps = get_score(pred_answers, gold_answers)
    _, po = get_score(pred_answers, orig_answers)
    mr = po / (ps + po + 1e-10) * 100
    print('ps {}, po {}, mr {}, em {}.'.format(ps, po, mr, em))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_path", default="./datasets/nq/orig_dev_filtered.json", type=str)
    parser.add_argument("--counter_path", default="./datasets/nq/conflict_dev_filtered.json", type=str)
    parser.add_argument("--engine", default="text-davinci-003", type=str)
    parser.add_argument("--schema", default="base", type=str, help="Choose from the following prompting templates: base, attr, instr, opin, instr+opin.")
    parser.add_argument("--demo_mode", default="none", help="Choose from the following demonstrations: none, original, counter.")
    parser.add_argument("--num_demos", default=16, type=int)
    parser.add_argument("--log_path", default='', type=str)
    args = parser.parse_args()
    with open(args.orig_path, 'r') as fh:
        orig_examples = json.load(fh)
    with open(args.counter_path, 'r') as fh:
        counter_examples = json.load(fh)
    print('Loaded {} instances.'.format(len(counter_examples)))
    engine = Engine(args.engine)

    step = 0
    gold_answers, pred_answers, orig_answers = [], [], []
    for oe, ce in tqdm(zip(orig_examples, counter_examples), total=len(orig_examples)):
        if step % 100 == 0:
            eval(pred_answers, orig_answers, gold_answers)
        step += 1
        query, context, answer = ce['question'], ce['context'], ce['answer']
        orig_answer = oe['answer']
        if orig_answer is None:
            continue
        if args.demo_mode == 'none':
            demos = []
        elif args.demo_mode == 'counter':
            demos = ce['ic_examples']
        elif args.demo_mode == 'original':
            demos = ce['ico_examples']
        for num_demos in range(args.num_demos, 1, -1):  # Use fewer demos if prompt is too long
            prompt = qa_to_prompt(query, context, schema=args.schema, demos=demos, num_demos=num_demos)
            if not engine.check_prompt_length(prompt):
                break
        if engine.check_prompt_length(prompt):
            continue
        pred = engine.complete(prompt)
        if pred is None:
            continue
        pred_answers.append(pred)
        gold_answers.append(answer)
        orig_answers.append(orig_answer)
        # Logs
        ce['prediction'] = pred
        ce['orig_answer'] = orig_answer
        ce['schema'] = args.schema
        ce['demo_mode'] = args.demo_mode
    if args.log_path:
        with open(args.log_path, 'w') as fh:
            json.dump(counter_examples, fh)
    eval(pred_answers, orig_answers, gold_answers)

if __name__ == '__main__':
    main()