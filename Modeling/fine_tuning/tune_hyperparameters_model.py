import argparse
import json
import os
import pprint
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import TrainingArguments, Seq2SeqTrainingArguments, AutoModelForCausalLM, AutoTokenizer, \
    BitsAndBytesConfig, Trainer
from ray import tune
from ray.air import session

from Dataset.mathematica import Mathematica


def custom_trial_dirname_creator(trial):
    return f"trial_{trial.trial_id}"


class GPTTrainer(Trainer):
    """
    Using the AdamW Optimizer to hinder overfitting and utilize the weight decay to resolve the weight reduction problem.
    Also specifying the learning rate strategy (constant or linear warmup where the learning rate decreases linearly)
    """

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            print("Making AdamW Optimizer")
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0
                },
            ]
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

        if self.lr_scheduler is None:
            if self.args.warmup_steps == -1:
                print("Using constant LR")
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda steps: 1.0)
            else:
                print("Using Linear warmup LR")
                self.lr_scheduler = self.get_linear_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
                )

    @staticmethod
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / (float(max(1, num_warmup_steps)))
            min_lr_multiplier = 0.1
            return max(
                min_lr_multiplier,
                ((1 - min_lr_multiplier) * float(num_training_steps - current_step) / float(
                    max(1, num_training_steps - num_warmup_steps))) + min_lr_multiplier
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def train_model(args, train_data):
    """
     Fine-tune a pretrained language model on mathematical tasks.
    :param args: several arguments needed to train the model
    :param train_data: the training dataset from get_dataset
    :return: None
    """

    model = None

    print("cuda_device_count: " + str(torch.cuda.device_count()))
    if not args.save_steps:
        save_steps = len(train_data)
        save_steps = int(save_steps / args.grad_acc_steps)
        save_steps = int(save_steps / args.batch_size_per_replica)
        if not args.tpu_num_cores:
            if torch.cuda.is_available():
                save_steps = int(save_steps / torch.cuda.device_count())
        else:
            save_steps = int(save_steps / 8)
    else:
        save_steps = args.save_steps
    print("Save Steps= ", save_steps)

    if args.load:
        if args.arch == 'gpt2':
            model = transformers.GPT2LMHeadModel.from_pretrained(args.load)
            print(f"Loaded GPT2 model from {args.load}")
        elif args.arch == 'EleutherAI/gpt-neo-125m':
            model = AutoModelForCausalLM.from_pretrained(args.load)
            print(f"Loaded GPT-NEO model from {args.load}")
    else:
        if args.arch == 'gpt2':
            model = transformers.GPT2LMHeadModel.from_pretrained(args.arch, return_dict=True)
            print(f"Loaded GPT2 model from {args.arch}")
        elif args.arch == 'EleutherAI/gpt-neo-125m':
            model = AutoModelForCausalLM.from_pretrained(args.arch, return_dict=True)
            print(f"Loaded GPT-NEO model from {args.arch}")

    start_epoch = 0
    start_iteration = 0

    train_data.start_iteration = start_iteration

    print(f"Setting up Trainer")

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=False,
        do_predict=True,
        evaluation_strategy='no',
        eval_steps=0,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.lr_warmup_steps,
        max_grad_norm=100000.0,
        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_steps=save_steps,
        save_total_limit=10,
        dataloader_drop_last=True,
        dataloader_num_workers=args.dataloader_num_workers,
        local_rank=args.local_rank,
        tpu_num_cores=args.tpu_num_cores,
    )

    trainer = GPTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )
    trainer.remove_callback(transformers.integrations.TensorBoardCallback)

    print(f"STARTING TRAINING, save_steps={save_steps}")
    trainer.train()

    # Calculate and report loss for Ray Tune
    final_loss = trainer.state.log_history[-1]['loss']  # Extract the last reported loss
    session.report(loss=final_loss)

    trainer.save_model(os.path.join(args.save_dir, "results"))
    print("Finished")


def get_tokenizer(args):
    if args.arch == 'gpt2':
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.arch, return_tensors='pt')
    elif args.arch == 'EleutherAI/gpt-neo-125m':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", return_dict=True)

    return tokenizer


def get_dataset(args):
    """
    :param args: the command line arguments  to retrieve the training data from the specified dataroot
    :return: the training dataset
    """
    tokenizer = get_tokenizer(args)

    train_data = []

    if args.math_dataroot:
        # for math_dr in args.math_dataroot:

        flist_find_roots = args.math_dataroot + "\\train_data\\algebra\\find_roots"
        flist_invert_function = args.math_dataroot + "\\train_data\\algebra\\invert_function"
        flist_derivatives = args.math_dataroot + "\\train_data\\calculus\\derivatives"
        flist_integrals = args.math_dataroot + "\\train_data\\calculus\\integrals"
        flist_polygons = args.math_dataroot + "\\train_data\\geometry\\polygons"
        flist_triangles = args.math_dataroot + "\\train_data\\geometry\\triangles"
        flist_determinant = args.math_dataroot + "\\train_data\\linear_algebra\\determinant"
        flist_orthogonalize_vectors = args.math_dataroot + "\\train_data\\linear_algebra\\orthogonalize_vectors"

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
            train_data.append(Mathematica(
                dataroot=flist_find_roots,
                tokenizer=tokenizer,
                # max_tokens=2048 if args.arch == 'EleutherAI/gpt-neo-125m' else 1024,
                max_tokens=1024,
                mode=args.arch,
            ))
        if invert_function_num_files:
            train_data.append(Mathematica(
                dataroot=flist_invert_function,
                tokenizer=tokenizer,
                # max_tokens=2048 if args.arch == 'EleutherAI/gpt-neo-125m' else 1024,
                max_tokens=1024,
                mode=args.arch,
            ))
        if derivatives_num_files:
            train_data.append(Mathematica(
                dataroot=flist_derivatives,
                tokenizer=tokenizer,
                # max_tokens=512 if args.arch == 'bert-base-uncased' else 1024,
                max_tokens=1024,
                mode=args.arch,
            ))
        if integrals_num_files:
            train_data.append(Mathematica(
                dataroot=flist_integrals,
                tokenizer=tokenizer,
                # max_tokens=512 if args.arch == 'bert-base-uncased' else 1024,
                max_tokens=1024,
                mode=args.arch,
            ))
        if polygons_num_files:
            train_data.append(Mathematica(
                dataroot=flist_polygons,
                tokenizer=tokenizer,
                # max_tokens=512 if args.arch == 'bert-base-uncased' else 1024,
                max_tokens=1024,
                mode=args.arch,
            ))
        if triangles_num_files:
            train_data.append(Mathematica(
                dataroot=flist_triangles,
                tokenizer=tokenizer,
                max_tokens=512 if args.arch == 'bert-base-uncased' else 1024,
                mode=args.arch,
            ))
        if determinant_files:
            train_data.append(Mathematica(
                dataroot=flist_determinant,
                tokenizer=tokenizer,
                max_tokens=512 if args.arch == 'bert-base-uncased' else 1024,
                mode=args.arch,
            ))
        if orthogonalize_vectors_num_files:
            train_data.append(Mathematica(
                dataroot=flist_orthogonalize_vectors,
                tokenizer=tokenizer,
                max_tokens=512 if args.arch == 'bert-base-uncased' else 1024,
                mode=args.arch,
            ))

    for dset in train_data:
        print(f"{dset.__class__.__name__}: __len__ = {len(dset)}")

    return torch.utils.data.ConcatDataset(train_data)


def tune_hyperparameters(config, args, train_data):
    args.lr = config["lr"]
    args.epochs = config["epochs"]
    args.weight_decay = config["weight_decay"]

    train_model(args, train_data)


from ray.tune import CLIReporter


def main():
    ######### Arg parsing ###############################################################

    parser = argparse.ArgumentParser(description="Language Modelling on Code")
    parser.add_argument('--arch', default='gpt2', help="The name of the model to be used")
    parser.add_argument('--load', default=None, type=str, help="Model to be loaded after training is completed.")

    parser.add_argument('--math_dataroot', default=None, type=str,
                        help="To specify the path where the train data is stored")
    parser.add_argument('--MATH-peek-min', default=0.1, type=float)
    parser.add_argument('--MATH-peek-max', default=1.0, type=float)
    parser.add_argument('--dataloader-num-workers', default=1, type=int)

    parser.add_argument('--epochs', default=1, type=int, help="Specifying the epochs being run during the training")
    parser.add_argument('--lr', default=5e-5, type=float, help="Specifying the learning rate strategy")
    parser.add_argument('--weight-decay', default=0.05, type=float,
                        help="Regularization parameter used by the AdamW Optimizer")
    parser.add_argument('--lr-warmup-steps', default=0, type=int)
    parser.add_argument('--batch-size-per-replica', default=1, type=int, help="Specifying the Batch size")
    parser.add_argument('--grad-acc-steps', default=4, type=int, help="Used to accelerate the training process")
    parser.add_argument('--log-freq', default=50, type=int)

    parser.add_argument('--save_dir', default="./saved_models", type=str)
    parser.add_argument('--save_steps', default=None, type=int)

    parser.add_argument('--local_rank', type=int, default=-1, help="local_rank for distributed training on GPUs")
    parser.add_argument('--tpu-num-cores', type=int, default=None, help="TPU training if available")

    parser.add_argument('--tune', action='store_true', help="Set this flag to run hyperparameter tuning")

    args = parser.parse_args()

    if args.tune:
        search_space = {
            "lr": tune.loguniform(1e-5, 5e-4),
            "weight_decay": tune.loguniform(1e-6, 1e-2),
            "epochs": tune.choice([1, 3]),
            "batch_size_per_replica": tune.choice([1]),
        }

        train_data = get_dataset(args)

        reporter = CLIReporter(
            parameter_columns=["lr", "epochs", "weight_decay"],
            metric_columns=["loss", "accuracy", "training_iteration"]
        )

        analysis = tune.run(
            tune.with_parameters(tune_hyperparameters, args=args, train_data=train_data),
            resources_per_trial={"cpu": 4, "gpu": 1},
            config=search_space,
            num_samples=5,
            scheduler=tune.schedulers.ASHAScheduler(metric="loss", mode="min"),
            progress_reporter=reporter,
            name="tune_gpt_model",
            trial_dirname_creator=custom_trial_dirname_creator,
        )

        # Save the best model after tuning
        best_trial = analysis.get_best_trial("loss", "min", "last")
        if best_trial is None:
            print(
                "No best trial found. Please check if the metric is being logged correctly and trials are completing.")
        else:
            print("Best trial evaluated params: {}".format(best_trial.evaluated_params))
            print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))

            # Load and save the model from the best checkpoint
            best_checkpoint_dir = analysis.get_best_checkpoint(best_trial)
            print(f"Best checkpoint directory: {best_checkpoint_dir}")

            # Reload the model from the best checkpoint
            best_model = transformers.GPT2LMHeadModel.from_pretrained(best_checkpoint_dir)

            model_save_path = os.path.join(args.save_dir, "best_model")
            os.makedirs(model_save_path, exist_ok=True)

            # Save the model properly
            best_model.save_pretrained(model_save_path)
            print(f"Best model saved to: {model_save_path}")


if __name__ == "__main__":
    main()
