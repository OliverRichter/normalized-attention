import random
import json
import numpy as np

from pprint import pprint

random.seed(7) # 13 for config 1 - 50
A = -0.7395
B = -5.3648

HP = {
    "learning_rate": [4e-6, 6.3e-6, 1e-5, 1.6e-5, 2.5e-5, 4e-5, 6.3e-5, 1e-4, 1.6e-4, 2.5e-4, 4e-4],
    "attention_head_size":[8, 12, 16, 24, 32, 48, 64, 90, 128, 180, 256, 360, 512],
    "attention_heads":[1,2,3] #,4,6,8,10,12,16,20]
}

def find_clostes(log_model_dim):
    optimal_lr = np.exp(A*log_model_dim + B)
    diff = np.array([abs(optimal_lr - lr) for lr in HP['learning_rate']])
    return HP['learning_rate'][diff.argmin()]

def generate_new_config(lr, head_size, num_heads):
    config = {
        "learning_rate":lr,
        "attention_head_size":head_size,
        "attention_heads":num_heads
    }
    return config

def main():
    # original_config = generate_new_config(lr=6.3e-5, head_size=128, num_heads=2)
    configs = {
        # 0:original_config
    }

    # 1 - 50
    # for i in range(1,50):
    #     head_size = random.choice(HP["attention_head_size"])
    #     num_heads = random.choice(HP["attention_heads"])
    #     lr = random.choice(HP["learning_rate"])
    #     configs[i] = generate_new_config(lr, head_size, num_heads)

    # 50 - 70
    choices = [(ahs, numh) for ahs in HP['attention_head_size'] for numh in HP['attention_heads']]
    for i, (head_size, num_heads) in enumerate(choices, start=50):
        lr = find_clostes(np.log(head_size*num_heads))
        configs[i] = generate_new_config(lr, head_size, num_heads)

    # to check for duplicate configurations, not the case for random.seed(13)
    for i in range(50,50+len(choices)):
        for j in range(50,i):
            if configs[i] == configs[j]:
                print(i, j)
    # pprint(configs)
    # with open("./specs/hyperparameter-combinations-NAP.json","w") as file:
    #     json.dump(configs,file)

if __name__ == '__main__':
    main()
