import argparse
import os
import pprint
import torch
import transformers
from ray.tune.experiment import Trial
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_math import get_model_output, get_dataset
from math_equivalence import is_equiv
from Dataset.mathematica import Mathematica
from ray import tune
from ray.tune.schedulers import ASHAScheduler


def custom_trial_dirname_creator(trial):
    return f"trial_{trial.trial_id}"


# Ray Tune search space for hyperparameters
def tune_eval(config, checkpoint_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = argparse.Namespace(**config)

    trial_id = Trial.generate_id()
    trial_output_dir = "<path to the ray_temp directory>"

    # Ensure the directory exists
    os.makedirs(trial_output_dir, exist_ok=True)

    # Pass the trial ID to args so it's accessible in run_eval
    run_eval(args, trial_id)


def run_eval(args, trial_id):
    argsdict = vars(args)
    tokenizer = None
    print(pprint.pformat(argsdict))

    if args.arch in {'gpt2'}:
        print("Loading model from <path to the model>")
        model = transformers.GPT2LMHeadModel.from_pretrained('<path to the model>')
        print(f"Successfully loaded model from <path to the model>")
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.arch)
    elif args.arch in {'EleutherAI/gpt-neo-125m'}:
        print("Loading model from <path to the model>")
        model = AutoModelForCausalLM.from_pretrained('<path to the model>')
        print(f"Successfully loaded model from <path to the model>")
        tokenizer = AutoTokenizer.from_pretrained(args.arch)

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

            num_samples = 5
            generated_outputs = []
            for _ in range(num_samples):
                output_ids = model.generate(
                    batch['input_ids'],
                    num_beams=args.num_beams,
                    early_stopping=True if args.num_beams > 1 else False,
                    temperature=args.temperature,
                    max_length=args.max_length,
                    do_sample=True,  # Sampling statt deterministischer Generierung
                )
                output_tokens = get_model_output(batch['input_ids'][0], output_ids[0], tokenizer)
                output_str = tokenizer.decode(output_tokens)
                generated_outputs.append(output_str)

            # Konsistenz durch Mehrheitswahl (häufigste Antwort)
            most_common_output = max(set(generated_outputs), key=generated_outputs.count)
            correct_ans = tokenizer.decode(batch['labels'][0])

            print("Problem String:")
            print(tokenizer.decode(batch['input_ids'][0]) + "\n")
            print("Most Common Model Output:")
            print(most_common_output)
            print("Correct Answer:")
            print(correct_ans + "\n")
            print("fname")
            print(fname)
            print("--------------------------------------------")

            outputs.append(most_common_output)
            answers.append(correct_ans)

            equiv = is_equiv(most_common_output, correct_ans)
            if equiv:
                correct += 1
                mean_max_probs_correct.append(1)
            else:
                mean_max_probs_wrong.append(1)

            total += 1

    # Dynamisch den Output für jedes Trial festlegen
    output_file = os.path.join(args.output_dir, f"Outputs_400_1_v2_{trial_id}.txt")

    with open(output_file, "w+") as f:
        for k, (output, answer, fname) in enumerate(zip(outputs, answers, fnames_list)):
            f.write("{} OUTPUT: {} | ANSWER: {} | FNAME: {}\n".format(k, output, answer, fname))

        print("#####################")
        f.write("#####################\n")

        print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct / total))
        print("Skipped = {}".format(skipped))
        f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct / total))
        f.write("Skipped = {}".format(skipped))

    print()



# Define hyperparameters search space
search_space = {
    'arch': tune.choice(['gpt2']),
    'batch_size_per_replica': tune.choice([1]),
    'num_beams': tune.choice([1, 3, 5, 10]),
    'max_length': tune.choice([1024]),
    'temperature': tune.choice([1]),
    'workers': tune.choice([1]),
    'math_dataroot': tune.choice(['C:\\Users\\angel\\PycharmProjects\\pythonProject\\data_file_lists']),
    'output_dir': tune.choice(['<path to the ray_temp directory>']),
}


# Setup Ray Tune
def main():
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2)

    tune.run(
        tune_eval,
        config=search_space,
        scheduler=scheduler,
        num_samples=5,  # Number of hyperparameter configurations to sample
        trial_dirname_creator=custom_trial_dirname_creator,
        resources_per_trial={"cpu": 24, "gpu": 1},  # Allocate 1 GPU per trial
    )


if __name__ == "__main__":
    main()
