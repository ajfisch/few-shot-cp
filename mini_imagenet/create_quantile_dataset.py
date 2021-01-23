import pprint
import tqdm
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.data.distributed
import torch.utils.data
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.nn as nn
import torch
import argparse
import os
import json


from prototypical_networks.samplers.episodic_batch_sampler import EpisodicBatchSamplerWithClass
from prototypical_networks.dataloaders.mini_imagenet_loader import MiniImageNet, ROOT_PATH
from prototypical_networks.models.convnet_mini import ConvNet
from prototypical_networks.models.identity import Identity
from prototypical_networks.utils import AverageMeter, euclidean_dist, mkdir, compute_accuracy
from torch.utils.data import DataLoader


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names.append('default_convnet')

parser = argparse.ArgumentParser(description='Pytorch Prototypical Networks Testing')
parser.add_argument("--output_dir", type=str, default="results/")
parser.add_argument('--splits_path', type=str, default="/data/rsg/nlp/tals/coverage/models/prototypical-networks/mini_imagenet_5_splits", help='path to dir with per fold subdirs of csv files containing train/dev/test examples')
parser.add_argument('--full_path', type=str, default="/data/rsg/nlp/tals/coverage/models/prototypical-networks/mini_imagenet", help='path to dir with full data csv files containing train/dev/test examples')
parser.add_argument('--images_path', type=str, default="/data/rsg/nlp/tals/coverage/models/prototypical-networks/mini_imagenet", help='path to parent dir with "images" dir.')
parser.add_argument('-a', '--arch', metavar='ARCH', default='default_convnet',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--evaluation_name', type=str, help='Evaluation name')
parser.add_argument('--n_episodes', default=2000, type=int, help='Number of episodes to average')
parser.add_argument('--n_way', default=10, type=int, help='Number of classes per episode')
parser.add_argument('--n_support', default=16, type=int, help='Number of support samples per class')
parser.add_argument('--n_query', default=100, type=int, help='Number of query samples')
parser.add_argument("--fold_ckpts", type=str, nargs="+",
                    default=[
                    "/data/rsg/nlp/tals/coverage/models/prototypical-networks/models_trained/mini_imagenet_10_shot_10_way_split_0/model_best_acc.pth.tar",
                    "/data/rsg/nlp/tals/coverage/models/prototypical-networks/models_trained/mini_imagenet_10_shot_10_way_split_1/model_best_acc.pth.tar",
                    "/data/rsg/nlp/tals/coverage/models/prototypical-networks/models_trained/mini_imagenet_10_shot_10_way_split_2/model_best_acc.pth.tar",
                    "/data/rsg/nlp/tals/coverage/models/prototypical-networks/models_trained/mini_imagenet_10_shot_10_way_split_3/model_best_acc.pth.tar",
                    "/data/rsg/nlp/tals/coverage/models/prototypical-networks/models_trained/mini_imagenet_10_shot_10_way_split_4/model_best_acc.pth.tar",
                             ])
parser.add_argument("--full_ckpt", type=str,
                    default="/data/rsg/nlp/tals/coverage/models/prototypical-networks/models_trained/mini_imagenet_10_shot_10_way/model_best_acc.pth.tar")

def run_fold(dataset_dir, images_dir, ckpt, split, args):
    # Load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = ConvNet()

    print("=> loading checkpoint '{}'".format(ckpt))
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['state_dict'])

    model = model.to(device)

    model.eval()

    cudnn.benchmark = True

    test_dataset = MiniImageNet(split, dataset_dir, images_dir)
    test_sampler = EpisodicBatchSamplerWithClass(test_dataset.labels,
                                                 args.n_episodes, args.n_way,
                                                 args.n_support + args.n_query)
    test_loader = DataLoader(dataset=test_dataset, batch_sampler=test_sampler,
                             num_workers=args.workers, pin_memory=True)

    accuracy = AverageMeter()
    s_accuracy = AverageMeter()

    # Evaluate
    all_outputs = []
    with torch.no_grad():
        for n_episode, batch in tqdm.tqdm(enumerate(test_loader, 1), total=args.n_episodes):
            data, tasks = [_.cuda(non_blocking=True) for _ in batch]
            tasks = tasks.tolist()
            p = args.n_support * args.n_way
            data_support, data_query = data[:p], data[p:]

            # [n_support, n_way, model.out_channels]
            support_encodings = model(data_support).view(args.n_support, args.n_way, -1)

            # === Evaluate on Kth support using K-1 support set. ===
            # Construct K-fold batch.
            labels = torch.arange(args.n_way).long().to(support_encodings.device)
            all_s_scores = [[] for _ in  range(args.n_way)]
            all_s_probs = [[] for _ in  range(args.n_way)]
            for i in range(args.n_support):
                support_idx = [j for j in range(args.n_support) if j != i]
                support_idx = torch.LongTensor(support_idx).to(
                    support_encodings.device)

                # Query.
                # [n_way, model.output_channels]
                query = support_encodings[i:i + 1, :, :].squeeze()

                # Support.
                # [n_way, model.output_channels]
                support_proto = support_encodings.index_select(
                    0, support_idx).mean(dim=0)
                s_logits = euclidean_dist(query, support_proto)
                s_scores = s_logits.diag().tolist()
                s_probs = torch.softmax(s_logits, dim=1).diag().tolist()
                for i, s in enumerate(s_scores):
                    all_s_scores[i].append(s)
                    all_s_probs[i].append(s_probs[i])

                # Labels.
                s_acc = compute_accuracy(s_logits, labels)
                s_accuracy.update(s_acc, labels.size(0))

            # === Evaluate on all queries using full support set. ===
            # Generate labels [n_way, n_query]
            labels = torch.arange(args.n_way).long().repeat(args.n_query).to(
                support_encodings.device)

            class_prototypes = model(data_support).reshape(
                args.n_support, args.n_way, -1).mean(dim=0)

            # Compute accuracy.
            logits = euclidean_dist(model(data_query), class_prototypes)
            probs = torch.softmax(logits, dim=1)
            acc = compute_accuracy(logits, labels)
            accuracy.update(acc, data_query.size(0))

            # Collect per task scores from all queries.
            all_f_scores = [
                logits[torch.arange(data_query.size(0)), labels][torch.arange(
                    i, data_query.size(0), args.n_way)].tolist()
                for i in range(args.n_way)
            ]
            all_f_probs = [
                probs[torch.arange(data_query.size(0)), labels][torch.arange(
                    i, data_query.size(0), args.n_way)].tolist()
                for i in range(args.n_way)
            ]

            for i in range(args.n_way):
                all_outputs.append(
                    dict(s_scores=all_s_scores[i],
                         f_scores=all_f_scores[i],
                         s_probs=all_s_probs[i],
                         f_probs=all_f_probs[i],
                         task=tasks[i],
                         ))
                         #TODO: add representation

        print(f"K-1 accuracy: {s_accuracy.avg}")
        print(f"Test accuracy: {accuracy.avg}")

        return all_outputs


def extract_k_folds(args):
    all_outputs = []
    for i, ckpt in tqdm.tqdm(enumerate(args.fold_ckpts), desc="evaluating folds"):
        fold_outputs = run_fold(
            dataset_dir=os.path.join(args.splits_path, str(i)),
            images_dir=args.images_path,
            ckpt=ckpt,
            split="test",
            args=args)
        all_outputs.extend(fold_outputs)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_quantile.jsonl"), "w") as f:
        for output in all_outputs:
            f.write(json.dumps(output, sort_keys=True) + "\n")

def extract_full(args):
    val_outputs = run_fold(
         dataset_dir=args.full_path,
         images_dir=args.images_path,
         ckpt=args.full_ckpt,
         split="val",
         args=args)


    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "val_quantile.jsonl"), "w") as f:
        for output in val_outputs:
            f.write(json.dumps(output, sort_keys=True) + "\n")


def main():
    args = parser.parse_args()

    options = vars(args)
    printer = pprint.PrettyPrinter()
    printer.pprint(options)

    # Create model
    print("=> creating model '{}'".format(args.arch))

    assert args.arch == 'default_convnet'

    extract_full(args)
    extract_k_folds(args)


if __name__ == '__main__':
    main()
