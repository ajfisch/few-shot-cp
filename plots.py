import matplotlib as mpl
mpl.use("Agg")
import json
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import os
import collections

from absl import app
from absl import flags

sns.set_context("paper", font_scale=2.0)
sns.set_style("white")
sns.set_style("ticks")
sns.despine()

LINES = ["--", "-.", ":"]
OURS_LINE = "-"
SINGLE_LINE = ":"

COLORS = [u"#1f77b4", u"#2ca02c", u"#9467bd", u"#8c564b",
          u"#e377c2", u"#7f7f7f", u"#bcbd22", u"#17becf"]
OURS_COLOR = u"#ff7f0e"
SINGLE_COLOR = u"#d62728"


flags.DEFINE_integer("dpi", 100,
                     "DPI for saving")

flags.DEFINE_multi_string("qs", ["0.95", "0.90", "0.80", "0.70", "0.60", "0.50", "0.40", "0.30", "0.20", "0.10", "0.05"], "q values matching files to load from")
#flags.DEFINE_multi_string("qs", ["0.95", "0.90", "0.80", "0.70", "0.60", "0.50", "0.40"], "q values matching files to load from")

flags.DEFINE_string("path_chem", "/data/rsg/nlp/fisch/projects/few-shot-cp/ckpts/chembl/k=16/conformal/q=%s/val-trials=5000-results.jsonl",
                    "Path to chembl results (with %s placeholder for q value)")

flags.DEFINE_string("path_cv", "/data/rsg/nlp/fisch/projects/few-shot-cp/ckpts/mini_imagenet/16_shot_10_way/q=%s/conf_val-trials=5000-results.json",
                    "Path to mini imagenet results (with %s placeholder for q value)")

flags.DEFINE_string("path_nlp", "/data/rsg/nlp/fisch/projects/few-shot-cp/ckpts/fewrel/16_shot_10_way/q=%s/conf_val-trials=5000-results.json",
                    "Path to fewrel results (with %s placeholder for q value)")

flags.DEFINE_string("output_dir", "plots",
                    "Path to store output plots")

FLAGS = flags.FLAGS


def plot_task(qs, results_path, task_name, output_dir, max_size=10, dpi=100):
    # Each has row entries of the form:
    # {"epsilon": eps, "lambda": lmb, "results": {"meta": {"acc": [avg, std, p84, p16], "eff": [...]}, "exact": {...}}}
    # Here we only plot results for lambda = 0.
    exact = collections.defaultdict(dict)
    meta = collections.defaultdict(dict)
    for q in qs:
        fname = results_path % q
        with open(fname, "r") as f:
            for line in f:
                trial = json.loads(line)
                key = round(1 - trial["epsilon"], 2)
                res = trial["result"]
                if trial["delta"] != 0:
                    continue
                meta[q][key] = (res["meta"]["acc"], res["meta"]["eff"])
                if key not in exact:
                    exact[key] = (res["exact"]["acc"], res["exact"]["eff"])

    output_dir = os.path.join(output_dir, task_name)
    os.makedirs(output_dir, exist_ok=True)

    # Accuracy vs. tolerance
    out_path = os.path.join(output_dir, "acc.png")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.0, 1.0)
    ax.set_ylim(-0.0, 1.02)
    all_x = set()
    all_y = {}
    for q, values in meta.items():
        x = sorted(values.keys())
        all_x.update(x)
        for i in x:
            # There is some overlap in values of epsilon for quantile predictions,
            # we take the smaller of the overlaps.
            if i in all_y and all_y[i][0] < values[i][0][0]:
                continue
            else:
                all_y[i] = values[i][0]

    all_x = sorted(list(all_x))
    plt.plot(all_x, all_x, "--", color="k", label="diagonal")

    y = [all_y[i][0] for i in all_x]
    low = [all_y[i][2] for i in all_x]
    high = [all_y[i][3] for i in all_x]
    plt.fill_between(all_x, low, high, alpha=0.2)
    plt.plot(all_x, y, '-', label="Meta")

    x = sorted(exact.keys())
    y = [exact[i][0][0] for i in x]
    low = [exact[i][0][2] for i in all_x]
    high = [exact[i][0][3] for i in all_x]
    plt.fill_between(x, low, high, alpha=0.2)
    plt.plot(x, y, '-', label="Exact")

    plt.xlabel("1 - $\epsilon$")
    plt.ylabel("Accuracy")
    #plt.title("Accuracy vs $\epsilon$")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

    # Accuracy vs. size (efficiency)
    out_path = os.path.join(output_dir, "size.png")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.0, 1.0)
    ax.set_ylim(-0.0, 1.02 * max_size)
    all_x = set()
    all_y = {}
    for q, values in meta.items():
        x = sorted(values.keys())
        all_x.update(x)
        for i in x:
            # There is some overlap in values of epsilon for quantile predictions,
            # we take the smaller of the overlaps.
            if i in all_y and all_y[i][0] < values[i][1][0]:
                continue
            else:
                all_y[i] = values[i][1]

    all_x = sorted(list(all_x))

    y = [min(all_y[i][0], 2*max_size) for i in all_x]
    low = [all_y[i][2] for i in all_x]
    high = [all_y[i][3] for i in all_x]
    plt.fill_between(all_x, low, high, alpha=0.2)
    plt.plot(all_x, y, '-', label="Meta")

    x = sorted(exact.keys())
    y = [min(exact[i][1][0], 2*max_size) for i in x]
    low = [exact[i][1][2] for i in x]
    high = [exact[i][1][3] for i in x]
    plt.fill_between(x, low, high, alpha=0.2)
    plt.plot(x, y, '-', label="Exact")

    plt.xlabel("1 - $\epsilon$")
    plt.ylabel("Size")
    #plt.title("Accuracy vs $\epsilon$")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

    # Table.
    out_path = os.path.join(output_dir, "table_acc_size.txt")
    with open(out_path, 'w') as f:
        f.write("1-eps & meta acc & exact acc & meta size & exact size\\\\ \n")
        for q in meta.keys():
            qq = float(q)
            exact_values = exact[float(q)]
            f.write("%2.2f & %2.2f & %2.2f & %2.2f & %2.2f \\\\ \n" % (qq, meta[q][qq][0][0], exact_values[0][0], meta[q][qq][1][0], exact_values[1][0]))
    


def main(_):
    if FLAGS.path_chem:
        plot_task(FLAGS.qs, FLAGS.path_chem, "chem", FLAGS.output_dir, max_size=5, dpi=FLAGS.dpi)

    if FLAGS.path_cv:
        plot_task(FLAGS.qs, FLAGS.path_cv, "cv", FLAGS.output_dir, max_size=10, dpi=FLAGS.dpi)

    if FLAGS.path_nlp:
        plot_task(FLAGS.qs, FLAGS.path_nlp, "nlp", FLAGS.output_dir, max_size=10, dpi=FLAGS.dpi)



if __name__ == "__main__":
    app.run(main)