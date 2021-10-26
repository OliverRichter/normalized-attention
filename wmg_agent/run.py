#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os, sys, random, time
import json
import torch
# torch.autograd.set_detect_anomaly(True)

# Add the wmg_agent dir to the system path.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# CODE_DIR = os.path.dirname(os.path.abspath(__file__))
spec_rand = random.Random(time.time())  # For 'truly' random hyperparameter selection.

# No need for argparse. All settings are contained in the spec file.
num_args = len(sys.argv) - 1
if num_args != 2:
    print('run.py accepts a one argument specifying the path to the runspec.')
    # print("Config-Id 0 represents original hyperparameters.")
    exit(1)

# Read the runspec.
# from utils.spec_reader import SpecReader
# SpecReader(sys.argv[1])
full_path = os.path.join(sys.argv[1])
with open(full_path, 'r') as f:
    spec = json.load(f)
if spec["ENV_RANDOM_SEED"] == "randint":
    spec["ENV_RANDOM_SEED"] = spec_rand.randint(0,999999999)

run_id = int(sys.argv[2])
row = (run_id // 8) % 10
column = run_id % 8
spec["ID"] = run_id
spec["LEARNING_RATE"] = 0.3 ** (row + 1)
model_dimension = 2 ** (column + 3)
spec["WMG_MEMO_SIZE"] = model_dimension
spec["WMG_ATTENTION_HEAD_SIZE"] = model_dimension // spec["WMG_NUM_ATTENTION_HEADS"]
spec["WMG_HIDDEN_SIZE"] = 4 * model_dimension
spec["AC_HIDDEN_LAYER_SIZE"] = model_dimension
spec["AGENT_RANDOM_SEED"] = (run_id // 80) + 1
spec["ENV_RANDOM_SEED"] = (run_id // 80) + 1

# Execute the runspec.
from utils.worker import Worker
worker = Worker(spec)
worker.execute()
