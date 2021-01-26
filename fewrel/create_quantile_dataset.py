import pprint
from tqdm import tqdm
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
import sys
import numpy as np


from FewRel.fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, BERTPAIRSentenceEncoder, RobertaSentenceEncoder, RobertaPAIRSentenceEncoder
from FewRel.fewshot_re_kit.data_loader import get_loader_wclass
from FewRel.models.proto import Proto
from FewRel.models.gnn import GNN
from FewRel.models.snail import SNAIL
from FewRel.models.metanet import MetaNet
from FewRel.models.siamese import Siamese
from FewRel.models.pair import Pair
from FewRel.models.d import Discriminator
from FewRel.models.mtb import Mtb


parser = argparse.ArgumentParser(description='Pytorch Prototypical Networks Testing')
parser.add_argument("--output_dir", type=str, default="results/")
parser.add_argument('--splits_path', type=str, default="FewRel/data/wiki_5_splits/", help='path to dir with per fold subdirs of json files containing train/dev/test examples')
parser.add_argument('--full_path', type=str, default="FewRel/data/", help='path to dir with full data json files containing train/dev/test examples')
parser.add_argument('--model', default='proto',
        help='model name')
parser.add_argument('--encoder', default='cnn',
        help='encoder: cnn or bert or roberta')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--n_episodes', default=2000, type=int, help='Number of episodes to average')
parser.add_argument('--n_way', default=10, type=int, help='Number of classes per episode')
parser.add_argument('--n_support', default=16, type=int, help='Number of support samples per class')
parser.add_argument('--n_query', default=100, type=int, help='Number of query samples')
parser.add_argument("--fold_ckpts", type=str, nargs="+",
                    default=[
                    "/data/rsg/nlp/tals/coverage/meta_cp/FewRel/checkpoint/proto-cnn-wiki_5_splits__0__train-wiki_5_splits__0__val-10-10-0.pth.tar",
                    "/data/rsg/nlp/tals/coverage/meta_cp/FewRel/checkpoint/proto-cnn-wiki_5_splits__1__train-wiki_5_splits__1__val-10-10-1.pth.tar",
                    "/data/rsg/nlp/tals/coverage/meta_cp/FewRel/checkpoint/proto-cnn-wiki_5_splits__2__train-wiki_5_splits__2__val-10-10-2.pth.tar",
                    "/data/rsg/nlp/tals/coverage/meta_cp/FewRel/checkpoint/proto-cnn-wiki_5_splits__3__train-wiki_5_splits__3__val-10-10-3.pth.tar",
                    "/data/rsg/nlp/tals/coverage/meta_cp/FewRel/checkpoint/proto-cnn-wiki_5_splits__4__train-wiki_5_splits__4__val-10-10-4.pth.tar"
                             ])
parser.add_argument("--full_ckpt", type=str,
                    default="/data/rsg/nlp/tals/coverage/meta_cp/FewRel/checkpoint/proto-cnn-train_wiki-val_wiki-10-10.pth.tar")

parser.add_argument('--batch_size', default=1, type=int,
        help='batch size')
parser.add_argument('--max_length', default=128, type=int,
       help='max length')
parser.add_argument('--dot', action='store_true', 
       help='use dot instead of L2 distance for proto')
parser.add_argument('--pair', action='store_true',
       help='use pair model')
parser.add_argument('--hidden_size', default=230, type=int,
       help='hidden size')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def euclidean_dist(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


def load_model(ckpt, device, args):
    # Note: only cnn works for now...
    if args.encoder == 'cnn':
        try:
            glove_mat = np.load('./FewRel/pretrain/glove/glove_mat.npy')
            glove_word2id = json.load(open('./FewRel/pretrain/glove/glove_word2id.json'))
        except:
            raise Exception("Cannot find glove files. see README file to download.")
        sentence_encoder = CNNSentenceEncoder(
                glove_mat,
                glove_word2id,
                args.max_length)
    elif args.encoder == 'bert':
        raise NotImplementedError
        pretrain_ckpt = args.pretrain_ckpt or 'bert-base-uncased'
        if args.pair:
            sentence_encoder = BERTPAIRSentenceEncoder(
                    pretrain_ckpt,
                    args.max_length)
        else:
            sentence_encoder = BERTSentenceEncoder(
                    pretrain_ckpt,
                    args.max_length,
                    cat_entity_rep=args.cat_entity_rep,
                    mask_entity=args.mask_entity)
    elif args.encoder == 'roberta':
        raise NotImplementedError
        pretrain_ckpt = args.pretrain_ckpt or 'roberta-base'
        if args.pair:
            sentence_encoder = RobertaPAIRSentenceEncoder(
                    pretrain_ckpt,
                    args.max_length)
        else:
            sentence_encoder = RobertaSentenceEncoder(
                    pretrain_ckpt,
                    args.max_length,
                    cat_entity_rep=args.cat_entity_rep)
    else:
        raise NotImplementedError
    
    model_name = args.model
    if model_name == 'proto':
        model = Proto(sentence_encoder, dot=args.dot)
    elif model_name == 'gnn':
        model = GNN(sentence_encoder, N, hidden_size=args.hidden_size)
    elif model_name == 'snail':
        model = SNAIL(sentence_encoder, N, K, hidden_size=args.hidden_size)
    elif model_name == 'metanet':
        model = MetaNet(N, K, sentence_encoder.embedding, args.max_length)
    elif model_name == 'siamese':
        model = Siamese(sentence_encoder, hidden_size=args.hidden_size)
    elif model_name == 'pair':
        model = Pair(sentence_encoder, hidden_size=args.hidden_size)
    elif model_name == 'mtb':
        model = Mtb(sentence_encoder)
    else:
        raise NotImplementedError

    print("=> loading checkpoint '{}'".format(ckpt))
    state_dict = torch.load(ckpt)['state_dict']
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)

    model = model.to(device)
    return model, sentence_encoder


def run_fold(dataset_dir, ckpt, args, f_name='val_wiki'):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, sentence_encoder = load_model(ckpt, device, args)
    model.eval()

    test_data_loader = get_loader_wclass(f_name, sentence_encoder,
            N=args.n_way, K=args.n_support, Q=args.n_query,
            batch_size=args.batch_size, root=dataset_dir, num_workers=args.workers)

    # Evaluate
    accuracy = AverageMeter()
    s_accuracy = AverageMeter()
    all_outputs = []
    with torch.no_grad():
        for it in tqdm(range(args.n_episodes)):
            support, query, labels, rels = next(test_data_loader)
            if torch.cuda.is_available():
                for k in support:
                    support[k] = support[k].cuda()
                for k in query:
                    query[k] = query[k].cuda()
                labels = labels.cuda()

            # [n_way, n_support, model.hidden_size]
            support_emb = model.sentence_encoder(support).view(args.n_way, args.n_support, -1)

            # === Evaluate on Kth support using K-1 support set. ===
            s_labels = torch.arange(args.n_way).long().to(support_emb.device)
            all_s_scores = [[] for _ in  range(args.n_way)]
            all_s_probs = [[] for _ in  range(args.n_way)]
            for i in range(args.n_support):
                support_idx = [j for j in range(args.n_support) if j != i]
                support_idx = torch.LongTensor(support_idx).to(
                    support_emb.device)

                # Query.
                # [n_way, model.hidden_size]
                query_emb = support_emb[:, i:i + 1, :].squeeze()

                # Support.
                # [n_way, model.output_channels]
                support_proto = support_emb.index_select(
                    1, support_idx).mean(dim=1)

                s_logits = euclidean_dist(query_emb, support_proto)
                s_scores = s_logits.diag().tolist()
                s_probs = torch.softmax(s_logits, dim=-1).diag().tolist()

                for i, s in enumerate(s_scores):
                    all_s_scores[i].append(s)
                    all_s_probs[i].append(s_probs[i])

                preds = torch.argmax(s_logits, dim=1)
                s_acc = model.accuracy(preds, s_labels)
                s_accuracy.update(s_acc, s_labels.size(0))

            # === Evaluate on all queries using full support set. ===

            # [n_way, model.hidden_size]
            class_prototypes = support_emb.mean(-2)

            # [n_way * n_query, model.hidden_size]
            query_emb = model.sentence_encoder(query)

            # [n_way * n_query, n_way]
            logits = euclidean_dist(query_emb, class_prototypes)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=1)

            acc = model.accuracy(preds, labels).mean().item()
            accuracy.update(acc, query_emb.size(0))

            f_gold_probs = probs[torch.arange(args.n_query * args.n_way), labels]
            f_gold_scores = logits[torch.arange(args.n_query * args.n_way), labels]

            f_gold_probs = f_gold_probs.view(args.n_way, args.n_query)
            f_gold_scores = f_gold_scores.view(args.n_way, args.n_query)

            for i in range(args.n_way):
                all_outputs.append(
                    dict(s_scores=all_s_scores[i],
                         f_scores=f_gold_scores[i].tolist(),
                         s_probs=all_s_probs[i],
                         f_probs=f_gold_probs[i].tolist(),
                         task=rels[i * args.n_query],
                         ))

        print(f"K-1 accuracy: {s_accuracy.avg}")
        print(f"Test accuracy: {accuracy.avg}")

        return all_outputs


def extract_k_folds(args):
    all_outputs = []
    for i, ckpt in tqdm(enumerate(args.fold_ckpts), desc="evaluating folds"):
        fold_outputs = run_fold(
             dataset_dir=os.path.join(args.splits_path, str(i)),
             ckpt=ckpt,
             args=args,
             f_name='val')

        all_outputs.extend(fold_outputs)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_quantile.jsonl"), "w") as f:
        for output in all_outputs:
            f.write(json.dumps(output, sort_keys=True) + "\n")

def extract_full(args):
    val_outputs = run_fold(
         dataset_dir=args.full_path,
         ckpt=args.full_ckpt,
         args=args)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "val_quantile.jsonl"), "w") as f:
        for output in val_outputs:
            f.write(json.dumps(output, sort_keys=True) + "\n")


def main():
    args = parser.parse_args()
    if args.pair:
        raise NotImplementedError
    assert args.batch_size == 1
    assert not args.dot

    options = vars(args)
    printer = pprint.PrettyPrinter()
    printer.pprint(options)

    # Create model
    print("=> creating model '{}'".format(args.model))

    extract_full(args)
    extract_k_folds(args)


if __name__ == '__main__':
    main()
