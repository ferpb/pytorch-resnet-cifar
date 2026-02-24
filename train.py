import os
import argparse
import random
import uuid
import json

import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import resnet_cifar


MODELS = {
    f.__name__: f
    for f in [
        resnet_cifar.resnet20,
        resnet_cifar.resnet32,
        resnet_cifar.resnet44,
        resnet_cifar.resnet56,
        resnet_cifar.resnet110,
        resnet_cifar.resnet1202,
    ]
}

DEFAULT_CONFIG = {
    "model": "resnet20",
    "batch_size": 128,
    "max_steps": 64000,
    "initial_lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "warmup_lr": False,
    "log_every": 100,
    "eval_every": 1000,
    "save_every": 10000,
}

CONFIGS = {
    "resnet20": {
        **DEFAULT_CONFIG,
        "model": "resnet20",
    },
    "resnet32": {
        **DEFAULT_CONFIG,
        "model": "resnet32",
    },
    "resnet44": {
        **DEFAULT_CONFIG,
        "model": "resnet44",
    },
    "resnet56": {
        **DEFAULT_CONFIG,
        "model": "resnet56",
    },
    "resnet110": {
        **DEFAULT_CONFIG,
        "model": "resnet110",
        "warmup_lr": True,
    },
    "resnet1202": {
        **DEFAULT_CONFIG,
        "model": "resnet1202",
        "warmup_lr": True,
    },
}


def log_string(string, log_file=None):
    if log_file is not None:
        log_file.write(string + "\n")
        log_file.flush()
    print(string)


def get_lr(step, initial_lr, warmup=False):
    lr = 0.1 * initial_lr if warmup and step < 400 else initial_lr
    if step >= 32000:
        lr *= 0.1
    if step >= 48000:
        lr *= 0.1
    return lr


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    wrong, total = 0, 0
    for input, target in loader:
        input, target = input.to(device), target.to(device)
        wrong += (model(input).argmax(dim=1) != target).sum().item()
        total += target.size(0)
    return wrong / total


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # CIFAR-10 mean and std
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_data = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
    train_eval_data = datasets.CIFAR10("./data", train=True, download=True, transform=eval_transform)
    test_data = datasets.CIFAR10("./data", train=False, download=True, transform=eval_transform)

    orig_model = MODELS[args.model]()
    orig_model.to(args.device)
    model = torch.compile(orig_model)

    optimizer = torch.optim.SGD(orig_model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    train_eval_loader = torch.utils.data.DataLoader(train_eval_data, args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # logging
    args.experiment_name = f"{args.model}-{str(uuid.uuid4())[:4]}"

    if args.log:
        results_dir = os.path.join("results", args.experiment_name)
        os.makedirs(results_dir)
        log_file = open(os.path.join(results_dir, "log.jsonl"), "w")
        # save code snapshot
        os.system(f"tar zcvf {results_dir}/src.tgz *.py")
    else:
        results_dir = None
        log_file = None

    log_string(str(vars(args)), log_file)

    step = 0
    last_step = False
    ema_beta = 0.9
    smooth_train_loss = 0

    while not last_step:
        for input, target in train_loader:
            last_step = step == args.max_steps

            # evaluate periodically
            if (step > 0 and step % args.eval_every == 0) or last_step:
                train_error = evaluate(model, train_eval_loader, args.device)
                test_error = evaluate(model, test_loader, args.device)
                log_metrics = {
                    "step": step,
                    "train_error": train_error,
                    "test_error": test_error,
                }
                log_string(json.dumps(log_metrics), log_file)

            # save checkpoint
            if (step > 0 and step % args.save_every == 0) or last_step:
                checkpoint = {
                    "args": vars(args),
                    "step": step,
                    "state_dict": orig_model.state_dict(),
                }
                if args.log:
                    torch.save(checkpoint, os.path.join(results_dir, f"checkpoint-{step:05d}.pth"))

            if last_step:
                break

            # training
            model.train()

            input, target = input.to(args.device), target.to(args.device)

            lr = get_lr(step, args.initial_lr, args.warmup_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            logits = model(input)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss.item()
            debiased_train_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))

            if step % args.log_every == 0:
                log_metrics = {
                    "step": step,
                    "lr": lr,
                    "train_loss": debiased_train_loss,
                }
                log_string(json.dumps(log_metrics), log_file)

            step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="resnet20", choices=CONFIGS.keys())
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    vars(args).update(CONFIGS[args.config])
    main(args)
