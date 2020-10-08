#!/usr/bin/env python3
"""Barebones job launcher/scheduler.

Reads commands from stdin and executes them via queueing on available GPUs.
"""

import argparse
import sys
import time
import uuid
import os
import pathlib
import multiprocessing
import subprocess

CUDA_VISIBLE_DEVICES = None

parser = argparse.ArgumentParser()
parser.add_argument('--devices', type=str, nargs='+', default=[0])


def name():
    jid = str(uuid.uuid4())[:8]
    date = time.strftime("%Y%m%d")
    base = '/tmp/chem-cp-job-logs'
    dirname = os.path.join(base, date, jid)
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    return dirname


def init(queue):
    global idx
    idx = queue.get()


def run(cmd):
    global CUDA_VISIBLE_DEVICES
    gpu_id = CUDA_VISIBLE_DEVICES[idx]
    cmd = 'CUDA_VISIBLE_DEVICES=%s %s' % (gpu_id, cmd)
    log_dir = name()
    print(log_dir)
    with open(os.path.join(log_dir, 'stdout.txt'), 'w') as stdout, \
         open(os.path.join(log_dir, 'stderr.txt'), 'w') as stderr:
        subprocess.call(cmd, shell=True, stdout=stdout, stderr=stderr)


def main(args):
    global CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES = args.devices

    jobs = []
    for line in sys.stdin:
        jobs.append(line.strip())

    manager = multiprocessing.Manager()
    idQueue = manager.Queue()
    for i in range(len(args.devices)):
        idQueue.put(i)
    workers = multiprocessing.Pool(len(args.devices), init, (idQueue,))
    workers.map(run, jobs)


if __name__ == "__main__":
    main(parser.parse_args())
