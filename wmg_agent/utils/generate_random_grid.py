import random
import json
import numpy as np
from copy import copy

from pprint import pprint

random.seed(42)

HP = {
    "A3C_T_MAX": [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 96, 120],
    "LEARNING_RATE": [4e-6, 6.3e-6, 1e-5, 1.6e-5, 2.5e-5, 4e-5, 6.3e-5, 1e-4, 1.6e-4, 2.5e-4, 4e-4],
    "DISCOUNT_FACTOR": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998],
    "GRADIENT_CLIP": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    "ENTROPY_TERM_STRENGTH": [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    "ADAM_EPS": [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12],
    "REWARD_SCALE": [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128],

    # WMG
    "WMG_MAX_MEMOS": [1, 2, 3, 4, 6, 8, 10, 12, 16, 20],
    "WMG_MEMO_SIZE": [32, 45, 64, 90, 128, 180, 256, 360, 512, 720, 1024, 1440, 2048],
    "WMG_NUM_LAYERS": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "WMG_NUM_ATTENTION_HEADS": [1,2,3,4,6,8,10,12,16,20],
    "WMG_ATTENTION_HEAD_SIZE": [8, 12, 16, 24, 32, 48, 64, 90, 128, 180, 256, 360, 512],
    "WMG_HIDDEN_SIZE": [6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512],
    "AC_HIDDEN_LAYER_SIZE": [64, 90, 128, 180, 256, 360, 512, 720, 1024, 1440, 2048, 2880, 4096, 5760]
}

def main():
    # load template config
    with open("./specs/train_wmg_on_factored_babyai_TEMPLATE.json","r") as f:
        template = json.load(f)

    grid = {'NAP':[],'Original':[]}
    # generate 150 random HP combinations, the same for NAP and Original
    for i in range(150):
        config = template.copy()
        config['ID'] = i
        for hp, vals in HP.items():
            config[hp] = random.choice(vals)
        for k in grid.keys():
            new_config = config.copy()
            new_config['WMG_TRANSFORMER_TYPE'] = k
            new_config['SAVE_MODELS_TO'] = new_config['SAVE_MODELS_TO'].replace("X",
                f"{new_config['WMG_TRANSFORMER_TYPE']}_{i}")
            grid[k].append(new_config)

    # to check for duplicate configurations, not the case for random.seed(42)
    for i in range(len(grid['Original'])):
        for j in range(i+1,len(grid['Original'])):
            if grid['Original'][i] == grid['Original'][j]:
                print(i, j)


    # pprint(configs)
    for k in grid.keys():
        for c in grid[k]:
            with open(f"./specs/random_grid/train_wmg_on_factored_babyai_{k}_{c['ID']}.json","w") as f:
                json.dump(c,f)

if __name__ == '__main__':
    main()
