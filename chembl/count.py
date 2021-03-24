import sys
import csv
import tqdm
import subprocess
fname = sys.argv[1]
num_lines = int(subprocess.check_output(["wc", "-l", fname], encoding="utf8").split()[0])
with open(fname, "r") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    tasks = set()
    for row in tqdm.tqdm(reader, total=num_lines - 1, desc="reading dataset"):
        task = row[columns[1]]
        tasks.add(task)
print("Num tasks", len(tasks))
