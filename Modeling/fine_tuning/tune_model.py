import argparse
import os
import pprint
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import TrainingArguments, Seq2SeqTrainingArguments, AutoModelForCausalLM, AutoTokenizer, \
    BitsAndBytesConfig, Trainer

from Dataset.mathematica import Mathematica


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
     Here a pretrained language model for example gpt2 will be fine-tuned on mathematical tasks
    :param args: several arguments needed to train the model
    :param train_data: the training dataset from get_dataset
    :return: None
    """

    model = None

    # Calculate the steps needed for the training process. Amount depends on the cpu or gpu being used
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

    # Specify if an existing trained model should be extended with additional training data for
    # training or specify a particular pretrained language model for initial training
    if args.load:
        if args.arch in {'gpt2'}:
            model = transformers.GPT2LMHeadModel.from_pretrained(args.load)
            print(f"Loaded GPT2 model from {args.load}")
        elif args.arch in {'EleutherAI/gpt-neo-125m'}:
            model = AutoModelForCausalLM.from_pretrained(args.load)
            print(f"Loaded GPT-NEO model from {args.load}")
    else:
        if args.arch in {'gpt2'}:
            model = transformers.GPT2LMHeadModel.from_pretrained(args.arch, return_dict=True)
            print(f"Loaded GPT2 model from {args.arch}")
        elif args.arch in {'EleutherAI/gpt-neo-125m'}:
            model = AutoModelForCausalLM.from_pretrained(args.arch, return_dict=True)
            print(f"Loaded GPT-NEO model from {args.arch}")

    start_epoch = 0
    start_iteration = 0

    train_data.start_iteration = start_iteration

    print(f"Setting up Trainer")

    # All the training arguments used for training
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
    # Save the model for later extension or evaluation
    trainer.save_model(os.path.join(args.save_dir, "results"))
    print("Finished")


def get_tokenizer(args):
    """
    :param args: the command line arguments (for the tokenizer we only need to specify the language model name
    :return: the tokenizer for encoding the samples and decoding generated ids back to text
    """
    tokenizer = None

    if args.arch in {'gpt2'}:
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.arch, return_tensors='pt')
    elif args.arch in {'EleutherAI/gpt-neo-125m'}:
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
        # flist_invert_function = args.math_dataroot + "\\train_data\\algebra\\invert_function"
        # flist_derivatives = args.math_dataroot + "\\train_data\\calculus\\derivatives"
        # flist_integrals = args.math_dataroot + "\\train_data\\calculus\\integrals"
        # flist_polygons = args.math_dataroot + "\\train_data\\geometry\\polygons"
        # flist_triangles = args.math_dataroot + "\\train_data\\geometry\\triangles"
        # flist_determinant = args.math_dataroot + "\\train_data\\linear_algebra\\determinant"
        # flist_orthogonalize_vectors = args.math_dataroot + "\\train_data\\linear_algebra\\orthogonalize_vectors"

        flist_derivatives_steps = args.math_dataroot + "\\train_data\\calculus\\derivatives_steps"
        # flist_determinant_steps = args.math_dataroot + "\\train_data\\linear_algebra\\determinant_steps"

        with open(flist_find_roots, "r") as f:
            find_roots_num_files = len(f.readlines())

        # with open(flist_invert_function, "r") as f:
        # invert_function_num_files = len(f.readlines())

        # with open(flist_derivatives, "r") as f:
        # derivatives_num_files = len(f.readlines())

        # with open(flist_integrals, "r") as f:
        # integrals_num_files = len(f.readlines())

        # with open(flist_polygons, "r") as f:
        # polygons_num_files = len(f.readlines())

        # with open(flist_triangles, "r") as f:
        # triangles_num_files = len(f.readlines())

        # with open(flist_determinant, "r") as f:
        # determinant_files = len(f.readlines())

        # with open(flist_orthogonalize_vectors, "r") as f:
        # orthogonalize_vectors_num_files = len(f.readlines())

        # with open(flist_derivatives_steps, "r") as f:
        # derivatives_steps_num_files = len(f.readlines())

        # with open(flist_determinant_steps, "r") as f:
        # determinant_steps_num_files = len(f.readlines())

        if find_roots_num_files:
            train_data.append(Mathematica(
                dataroot=flist_find_roots,
                tokenizer=tokenizer,
                # max_tokens=2048 if args.arch == 'EleutherAI/gpt-neo-125m' else 1024,
                max_tokens=1024,
                mode=args.arch,
            ))
        """if invert_function_num_files:
            train_data.append(Mathematica(
                dataroot=flist_invert_function,
                tokenizer=tokenizer,
                # max_tokens=2048 if args.arch == 'EleutherAI/gpt-neo-125m' else 1024,
                max_tokens=1024,
                mode=args.arch,
            ))"""
        """if derivatives_num_files:
            train_data.append(Mathematica(
                dataroot=flist_derivatives,
                tokenizer=tokenizer,
                # max_tokens=512 if args.arch == 'bert-base-uncased' else 1024,
                max_tokens=1024,
                mode=args.arch,
            ))"""
        """if integrals_num_files:
            train_data.append(Mathematica(
                dataroot=flist_integrals,
                tokenizer=tokenizer,
                # max_tokens=512 if args.arch == 'bert-base-uncased' else 1024,
                max_tokens=1024,
                mode=args.arch,
            ))"""
        """if polygons_num_files:
            train_data.append(Mathematica(
                dataroot=flist_polygons,
                tokenizer=tokenizer,
                # max_tokens=512 if args.arch == 'bert-base-uncased' else 1024,
                max_tokens=1024,
                mode=args.arch,
            ))"""
        """if triangles_num_files:
            train_data.append(Mathematica(
                dataroot=flist_triangles,
                tokenizer=tokenizer,
                max_tokens=512 if args.arch == 'bert-base-uncased' else 1024,
                mode=args.arch,
            ))"""
        """if determinant_files:
            train_data.append(Mathematica(
                dataroot=flist_determinant,
                tokenizer=tokenizer,
                max_tokens=512 if args.arch == 'bert-base-uncased' else 1024,
                mode=args.arch,
            ))"""
        """if orthogonalize_vectors_num_files:
            train_data.append(Mathematica(
                dataroot=flist_orthogonalize_vectors,
                tokenizer=tokenizer,
                max_tokens=512 if args.arch == 'bert-base-uncased' else 1024,
                mode=args.arch,
            ))"""
        """if derivatives_steps_num_files:
            train_data.append(Mathematica(
                dataroot=flist_derivatives_steps,
                tokenizer=tokenizer,
                max_tokens=512 if args.arch == 'bert-base-uncased' else 1024,
                mode=args.arch,
            ))"""
        """if determinant_steps_num_files:
            train_data.append(Mathematica(
                dataroot=flist_determinant_steps,
                tokenizer=tokenizer,
                max_tokens=512 if args.arch == 'bert-base-uncased' else 1024,
                mode=args.arch,
            ))"""

    for dset in train_data:
        print(f"{dset.__class__.__name__}: __len__ = {len(dset)}")

    return torch.utils.data.ConcatDataset(train_data)


def main():
    ######### Arg parsing ###############################################################

    parser = argparse.ArgumentParser(description="Language Modelling on Code")
    parser.add_argument('--arch', default='gpt2', help="The name of the model to be used")
    parser.add_argument('--load', default=None, type=str, help="Model to be loaded after training is completed.")

    # Dataloading
    parser.add_argument('--math_dataroot', default=None, type=str,
                        help="To specify the path where the train data is stored")
    parser.add_argument('--MATH-peek-min', default=0.1, type=float)
    parser.add_argument('--MATH-peek-max', default=1.0, type=float)
    parser.add_argument('--dataloader-num-workers', default=1, type=int)

    # Training
    parser.add_argument('--epochs', default=1, type=int, help="Specifying the epochs being run during the training")
    parser.add_argument('--lr', default=5e-5, type=float, help="Specifying the learning rate strategy")
    parser.add_argument('--weight-decay', default=0.05, type=float,
                        help="Regularization parameter used by the AdamW Optimizer")
    parser.add_argument('--lr-warmup-steps', default=0, type=int)
    parser.add_argument('--batch-size-per-replica', default=8, type=int, help="Specifying the Batch size")
    parser.add_argument('--grad-acc-steps', default=4, type=int, help="Used to accelerate the training process")
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--tpu_num_cores', default=None, type=int,
                        help="Setting the amount of tpu cores available to accelerate processing")

    # Logging and stuff
    parser.add_argument('--save-dir', default="trained_models\\GPT-NEO", type=str,
                        help="Specify the directory where to save the model after training")
    parser.add_argument('--save-steps', default=0, type=int, help="Save steps to not start all over again when "
                                                                  "rerunning the training")
    parser.add_argument('--log-freq', default=5, type=int)

    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%m-%d-%Y__%H-%M-%S"))

    ######### Start training ##########################################################

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    train_data = get_dataset(args)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    train_model(args, train_data)


if __name__ == "__main__":
    main()
