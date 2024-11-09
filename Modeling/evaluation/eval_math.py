import os
import pprint

import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_equivalence import is_equiv
from Dataset.mathematica import Mathematica


def remove_boxed(s):
    """
    :param s: sample string
    :return: string with removed boxed latex expression
    """
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def get_real_sol_idxs(tokens_sol, tokenizer):
    """
    Return the start and stop indexes (inclusive) for everything inside \\boxed{...}
    """
    left_idx, right_idx = None, None
    for i in range(tokens_sol.shape[1]):
        if i < 3:
            continue

        if tokens_sol[0, i].item() and \
                tokens_sol[0, i - 1].item() == 276 and \
                tokens_sol[0, i - 2].item() == 3524:
            # at index i, we have the { of \\boxed{
            left_idx = i + 1  # Don't include the {

        if tokens_sol[0, i].item() == 50256:
            right_idx = i - 2  # don't include the token before the current one (usually the } from \boxed{})

    return left_idx, right_idx


def run_eval(args):
    argsdict = vars(args)
    tokenizer = None
    print(pprint.pformat(argsdict))

    # Load a trained model for evaluation
    if args.load:
        if args.arch in {'gpt2'}:
            print(f"Loading model from {args.load}")
            model = transformers.GPT2LMHeadModel.from_pretrained(args.load)
            print(f"Successfully loaded model from {args.load}")
            tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.arch)
        elif args.arch in {'EleutherAI/gpt-neo-125m'}:
            print(f"Loading model from {args.load}")
            model = AutoModelForCausalLM.from_pretrained(args.load)
            print(f"Successfully loaded model from {args.load}")
            tokenizer = AutoTokenizer.from_pretrained(args.arch)
    else:
        if args.arch in {'gpt2'}:
            model = transformers.GPT2LMHeadModel.from_pretrained(args.arch)
        elif args.arch in {'EleutherAI/gpt-neo-125m'}:
            model = AutoModelForCausalLM.from_pretrained(args.arch)

    eval_data = get_dataset(args)
    for inner_dset in eval_data.datasets:
        inner_dset.tokenizer = tokenizer

    dataloader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=args.batch_size_per_replica,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = model.eval()

    outputs = []
    answers = []
    fnames_list = []

    with torch.no_grad():
        correct = 0
        total = 0
        skipped = 0
        mean_max_probs_correct = []
        mean_max_probs_wrong = []
        print("cuda_device_count: " + str(torch.cuda.device_count()))
        for i, batch in enumerate(tqdm(dataloader)):
            print("batch: " + str(batch))

            if torch.sum(batch['input_ids']) == 0:
                skipped += 1
                print("SKIPPING", batch['fnames'][0])
                continue
            fname = batch['fnames'][0]
            assert len(fname) == 1
            fnames_list.append(fname[0])

            # ids upon the input_ids from the loaded model
            output_ids = model.generate(
                batch['input_ids'],
                num_beams=args.num_beams,
                early_stopping=True,
                temperature=1.0,
                max_length=512,
            )
            print("output_ids: " + str(output_ids))

            mean_probs_sol = 0

            #  return the tokens which shall be decoded to a word
            output_tokens = get_model_output(batch['input_ids'][0], output_ids[0], tokenizer)
            print("output_tokens: " + str(output_tokens))

            output_str = tokenizer.decode(output_tokens)
            correct_ans = tokenizer.decode(batch['labels'][0])
            # Truncate after the first line
            first_line_end = output_str.find('\n')  # or any other specific pattern
            output_full = None
            if first_line_end != -1:
                output_full = output_str[:first_line_end]
            # output_str = last_boxed_only_string(output_str)
            output_full = output_str

            """if args.math_mode == "eval_peeking":
                answer_str = last_boxed_only_string(tokenizer.decode(batch['labels'][0]))
            else:
                answer_str = tokenizer.decode(batch['labels'][0])

            output, answer = remove_boxed(output_str), remove_boxed(answer_str)"""

            print("Problem String:")
            print(tokenizer.decode(batch['input_ids'][0]) + "\n")
            print("Model output:")
            print(output_full)
            print("Correct answer:")
            print(correct_ans + "\n")
            print("fname")
            print(fname)
            print("--------------------------------------------")

            outputs.append(output_full)
            answers.append(correct_ans)

            # Check for answer equality and append it to either the amount of correct or wrong answer from the model
            equiv = is_equiv(output_full, correct_ans)
            if equiv:
                correct += 1
                mean_max_probs_correct.append(mean_probs_sol)
            else:
                mean_max_probs_wrong.append(mean_probs_sol)

            total += 1

    """print(f"Average of mean_max_probs_correct = {sum(mean_max_probs_correct)}/{len(mean_max_probs_correct)} = ",
          sum(mean_max_probs_correct) / len(mean_max_probs_correct))
    print(f"Average of mean_max_probs_wrong   = {sum(mean_max_probs_wrong)}/{len(mean_max_probs_wrong)} = ",
          sum(mean_max_probs_wrong) / len(mean_max_probs_wrong))"""

    # Saving the outputs and answers
    output_file = "Outputs/gpt2/Outputs_400_Find_Roots_Detailed"

    with open(output_file, "w+") as f:
        for k, (output, answer, fname) in enumerate(
                zip(outputs, answers, fnames_list)):
            f.write("{} OUTPUT: {} | ANSWER: {} | FNAME: {}\n".format(k, output, answer, fname))

        print("#####################")
        f.write("#####################\n")

        print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct / total))
        print("Skipped = {}".format(skipped))
        f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct / total))
        f.write("Skipped = {}".format(skipped))

    print()


def get_model_output(context, full_output, tokenizer):
    """Given the context and the full model output (context + generated),
    extract just the generated tokens.
    Remove the last token if it is <|endoftext|>"""

    ret = full_output[len(context):]
    print("ret:" + str(ret))
    if ret[-1] == tokenizer.eos_token_id:
        ret = ret[:-1]
    return ret


"""def get_model_output(context, full_output, tokenizer):
    ret = full_output[len(context):]
    if ret[-1] == tokenizer.eos_token_id:
        ret = ret[:-1]
    return ret"""


def get_tokenizer(args):
    """
    :param args: the command line arguments (for the tokenizer we only need to specify the language model name
    :return: the tokenizer for encoding the samples and decoding generated ids back to text
    """
    tokenizer = None

    if args.arch in {'gpt2'}:
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.arch, return_tensors='pt')
    elif args.arch in {'EleutherAI/gpt-neo-125m'}:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m", return_tensors='pt')

    return tokenizer


def get_dataset(args):
    tokenizer = get_tokenizer(args)
    """
    A Key difference to the training dataset is that here the tokenizer is set to None
    :param args: Command line arguments, specifically the dataroot argument
    :return: the test dataset
    """
    eval_datasets = []

    if args.math_dataroot:
        # for math_dr in args.math_dataroot:
        flist_find_roots = args.math_dataroot + "\\test_data\\algebra\\find_roots"
        flist_invert_function = args.math_dataroot + "\\test_data\\algebra\\invert_function"
        flist_derivatives = args.math_dataroot + "\\test_data\\calculus\\derivatives"
        flist_integrals = args.math_dataroot + "\\test_data\\calculus\\integrals"
        flist_polygons = args.math_dataroot + "\\test_data\\geometry\\polygons"
        flist_triangles = args.math_dataroot + "\\test_data\\geometry\\triangles"
        flist_determinant = args.math_dataroot + "\\test_data\\linear_algebra\\determinant"
        flist_orthogonalize_vectors = args.math_dataroot + "\\test_data\\linear_algebra\\orthogonalize_vectors"

        with open(flist_find_roots, "r") as f:
            find_roots_num_files = len(f.readlines())

        with open(flist_invert_function, "r") as f:
            invert_function_num_files = len(f.readlines())

        with open(flist_derivatives, "r") as f:
            derivatives_num_files = len(f.readlines())

        with open(flist_integrals, "r") as f:
            integrals_num_files = len(f.readlines())

        with open(flist_polygons, "r") as f:
            polygons_num_files = len(f.readlines())

        with open(flist_triangles, "r") as f:
            triangles_num_files = len(f.readlines())

        with open(flist_determinant, "r") as f:
            determinant_files = len(f.readlines())

        with open(flist_orthogonalize_vectors, "r") as f:
            orthogonalize_vectors_num_files = len(f.readlines())

        if find_roots_num_files:
            eval_datasets.append(Mathematica(
                dataroot=flist_find_roots,
                tokenizer=tokenizer,
                max_tokens=1024,
                # max_tokens=2048 if args.arch == 'EleutherAI/gpt-neo-125m' else 1024,
                mode='gpt2-eval',
            ))
        if invert_function_num_files:
            eval_datasets.append(Mathematica(
                dataroot=flist_invert_function,
                tokenizer=tokenizer,
                max_tokens=1024,
                # max_tokens=2048 if args.arch == 'EleutherAI/gpt-neo-125m' else 1024,
                mode='gpt2-eval',
            ))
        if derivatives_num_files:
            eval_datasets.append(Mathematica(
                dataroot=flist_derivatives,
                tokenizer=tokenizer,
                max_tokens=1024,
                # max_tokens=2048 if args.arch == 'EleutherAI/gpt-neo-125m' else 1024,
                mode='gpt2-eval',
            ))
        if integrals_num_files:
            eval_datasets.append(Mathematica(
                dataroot=flist_integrals,
                tokenizer=tokenizer,
                max_tokens=1024,
                # max_tokens=2048 if args.arch == 'EleutherAI/gpt-neo-125m' else 1024,
                mode='gpt2-eval',
            ))
        if polygons_num_files:
            eval_datasets.append(Mathematica(
                dataroot=flist_polygons,
                tokenizer=tokenizer,
                max_tokens=1024,
                # max_tokens=2048 if args.arch == 'EleutherAI/gpt-neo-125m' else 1024,
                mode='gpt2-eval',
            ))
        if triangles_num_files:
            eval_datasets.append(Mathematica(
                dataroot=flist_triangles,
                tokenizer=tokenizer,
                max_tokens=1024,
                # max_tokens=2048 if args.arch == 'EleutherAI/gpt-neo-125m' else 1024,
                mode='gpt2-eval',
            ))
        if determinant_files:
            eval_datasets.append(Mathematica(
                dataroot=flist_determinant,
                tokenizer=tokenizer,
                max_tokens=1024,
                # max_tokens=2048 if args.arch == 'EleutherAI/gpt-neo-125m' else 1024,
                mode='gpt2-eval',
            ))
        if orthogonalize_vectors_num_files:
            eval_datasets.append(Mathematica(
                dataroot=flist_orthogonalize_vectors,
                tokenizer=tokenizer,
                max_tokens=1024,
                # max_tokens=2048 if args.arch == 'EleutherAI/gpt-neo-125m' else 1024,
                mode='gpt2-eval',
            ))

        eval_data = torch.utils.data.ConcatDataset(eval_datasets)
        return eval_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Language Modelling on Code")
    parser.add_argument('--arch', default='gpt2', help="The name of the model to be used")
    parser.add_argument('--load', default=None, type=str, help="Model to be loaded for evaluation")
    parser.add_argument('--num-beams', default=10, type=int)

    # Dataloading
    parser.add_argument('--math_dataroot', default=None, type=str,
                        help="To specify the path where the test data is stored")
    parser.add_argument('--math_mode', default='gpt2-eval', type=str,
                        help="Specify upon which pretrained model the evaluation shall be done")
    parser.add_argument('--peek-fraction', type=float, default=1.0)

    # Others
    parser.add_argument('--batch-size-per-replica', default=8, type=int, help="Specifying the Batch size")
    parser.add_argument('--workers', default=1, type=int)

    args = parser.parse_args()

    run_eval(args)
