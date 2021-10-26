import os, sys
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import json
import pandas as pd
from tqdm import tqdm
import seaborn as sns

def extract_success_rate(lines):
    '''Read success rate from output file.'''
    success_rate = []
    lines = [l for l in lines if 'Success rate' in l]
    for l in lines:
        l = l.split(' ')
        idx = l.index('Success')
        success_rate.append(float(l[idx-1]))
    return success_rate

def extract_success_rate_and_time(lines):
    '''Read success rate and time from output file.'''
    success_rate = []
    time = []
    lines = [l for l in lines if 'Success rate' in l]
    for l in lines:
        l = l.split(' ')
        idx = l.index('Success')
        success_rate.append(float(l[idx-1]))
        idx = l.index('sec')
        time.append(float(l[idx-1]))
    return time, success_rate

def extract_time_steps(lines):
    ''''''
    time = float(lines[-2].split('sec')[0].lstrip(' ').rstrip(' '))
    steps = int(lines[-2].split('steps')[0].split('sec')[1].lstrip(' ').rstrip(' ').replace(',',''))
    return time, steps

def sec_to_hms(sec):
    sec = int(sec)
    return f"{sec//3600} h {(sec%3600)//60} m {(sec%3600)%60} s"

def plot_rezero():
    files = []
    files.append(os.path.abspath('./results/NAP_0_20-11-01_07-39-07.txt'))
    files.append(os.path.abspath('./results/201022_array_job_results/NAP_0_20-10-19_14-24-24.txt'))
    model = files[0].split('/')[-1].split('_')[0]
    for file_path, mode in zip(files,['ReZero','Normal']):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        success_rate = extract_success_rate(lines)
        steps = np.linspace(1e3,1e3*len(success_rate),len(success_rate))
        plt.plot(steps,success_rate,label=f"{mode}, SR: {max(success_rate)}")
        plt.ylim((0.,1.1))
        plt.xlim((0,2e5))
        plt.hlines(y = 0.99, xmin=0,xmax=2e5,linestyle='--',color='r')
        plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
        plt.xlabel('Steps')
        plt.ylabel('Success Rate')
        plt.title(f'{model}')
        plt.legend(loc='lower right')
    plt.savefig(f'./plots/Rezero-{model}.png',dpi=100)
    plt.show()

def plot_single(file_path):
    # files = glob.glob(os.path.join(os.path.abspath(folder_path),'_*.txt'))
    # files = glob.glob(file_path)
    # for file_path in files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        success_rate = extract_success_rate(lines)
        label = file_path.split('/')[-1][:-4]
        steps = np.linspace(1e3,1e3*len(success_rate),len(success_rate))
        plt.plot(steps,success_rate,label=label)
        plt.ylim((0.,1.1))
        plt.xlim((0,8e5))
        plt.hlines(y = 0.99, xmin=0,xmax=8e5,linestyle='--',color='r')
        plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
        plt.xlabel('Steps')
        plt.ylabel('Success Rate')
        # plt.title(f"Max. Success Rate: {max(success_rate)}")
        plt.legend(loc='upper left')
        # out_path = file_path[:-4] +'.png'
        # plt.savefig('./results/name.png',dpi=100)
        plt.show()

def plot_all_in_one(folder_path, model=None):
    if model == None:
        print("Please specify model {Original, NAP}.")
        exit(1)
    files = glob.glob(os.path.join(os.path.abspath(folder_path),f"{model}_*.txt"))
    max_success_rates = []
    max_len = 0
    success_rates = []
    for file_path in files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        success_rate = extract_success_rate(lines)
        max_len = max(max_len, len(success_rate))
        success_rates.append(success_rate)
        max_success_rates.append(max(success_rate))
    label = files[0].split('/')[-1].split('_')[0]

    mean_success_rate = [[],[]]
    for i in range(max_len):
        tmp = [sr[i] if len(sr) > i else sr[-1] for sr in success_rates]
        mean_success_rate[0].append(np.mean(tmp))
        mean_success_rate[1].append(np.std(tmp))

    mean_success_rate = np.array(mean_success_rate)
    steps = np.linspace(1e3,1e3*max_len,max_len)
    plt.plot(steps,mean_success_rate[0],label=label)
    plt.fill_between(steps, mean_success_rate[0]-mean_success_rate[1],
                            mean_success_rate[0]+mean_success_rate[1],
                            color='r',
                            alpha=0.1)
    plt.ylim((0.,1.1))
    plt.xlim((0,2e5))
    plt.hlines(y = 0.99, xmin=0,xmax=2e5,linestyle='--',color='r')
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
    plt.xlabel('Steps')
    plt.ylabel('Success Rate')
    plt.title(f"Mean-Max-Succes-Rate: {np.mean(max_success_rates):.4f} +/- {np.std(max_success_rates):.4f}\n"
                f"Min-Max-SR: {np.min(max_success_rates):.4f}, Max.-Max.-SR: {np.max(max_success_rates):.4f}")
    plt.legend()
    # plt.savefig(f"./{model}_array.png",dpi=100)
    plt.show()
    print(f"Mean-Max-Succes-Rate: {np.mean(max_success_rates):.3f} +/- {np.std(max_success_rates):.3f}")
    print(f"Median-Max-Success-Rate: {np.median(max_success_rates):.3f}")

def who_won(folder_path):
    files_org = glob.glob(os.path.join(os.path.abspath(folder_path),'201022_array_job_results/Original_*.txt'))
    files_nap = glob.glob(os.path.join(os.path.abspath(folder_path),'NormalizedOriginal_*.txt'))

    stats = {'Original':0,
            'NormalizedOriginal':0}
    max_org_sr, max_nap_sr = [], []
    for org, nap in zip(files_org, files_nap):
        with open(org, 'r') as f:
            lines = f.readlines()
        max_org_sr.append(max(extract_success_rate(lines)))

        with open(nap, 'r') as f:
            lines = f.readlines()
        max_nap_sr.append(max(extract_success_rate(lines)))
        if max_org_sr[-1] < 0.99 or max_nap_sr[-1] < 0.99: # else considered a tie
            if max_org_sr[-1] > max_nap_sr[-1]:
                stats['Original'] += 1
            else:
                stats['NormalizedOriginal'] += 1

    for name, score in stats.items():
        print(name, score)
    plt.figure()
    plt.hist(max_org_sr,bins=50,range=(0.,1.),alpha=0.5,color='b',label='Original')
    plt.hist(max_nap_sr,bins=50,range=(0.,1.),alpha=0.5,color='r',label='NormalizedOriginal')
    plt.legend(loc='upper left')
    plt.xlabel('Success Rate')
    plt.savefig('Success-rate-distribution-no-vs-orig.png',dpi=100)
    plt.show()

def correlation(path, model):
    with open('./specs/hyperparameter-combinations.json','r') as f:
        params = json.load(f)
    files = glob.glob(os.path.join(os.path.abspath(path),f"{model}_*.txt"))

    max_sr, lr, model_dim = list(), list(), list()
    for file_path in files:
        idx = file_path.split('/')[-1].split('_')[1]
        lr.append(params[idx]['learning_rate'])
        model_dim.append(params[idx]['attention_head_size']*params[idx]['attention_heads'])
        with open(file_path, 'r') as f:
            lines = f.readlines()
        max_sr.append(max(extract_success_rate(lines)))

    plt.set_cmap('coolwarm')
    fig, ax = plt.subplots(2)
    ax[0].set_title(f"{model}")
    ax0 = ax[0].scatter(np.log(lr), max_sr, c=np.log(model_dim))
    x = np.linspace(np.min(np.log(lr)), np.max(np.log(lr)))
    m, b, r_val, p_val, _ = stats.linregress(np.log(lr), max_sr)
    ax[0].plot(x,b+m*x,c='black', label=f"Corr.-Coef.: {r_val:.3f}")
    ax[0].set_xlabel('log Learning Rate')
    ax[0].set_ylabel('Success Rate')
    ax[0].legend(loc='lower left')
    plt.colorbar(ax0, ax=ax[0])

    ax1 = ax[1].scatter(np.log(model_dim), max_sr, c=np.log(lr))
    x = np.linspace(np.min(np.log(model_dim)), np.max(np.log(model_dim)))
    m, b, r_val, p_val, _ = stats.linregress(np.log(model_dim), max_sr)
    ax[1].plot(x,b+m*x,c='black', label=f"Corr.-Coef.: {r_val:.3f}")
    ax[1].set_xlabel('log Model Dimension')
    ax[1].legend(loc='lower left')
    plt.colorbar(ax1, ax=ax[1])

    plt.tight_layout()
    plt.savefig(f"{model}-correlation.png",dpi=100)
    plt.show()

def correlation2(path, model):
    with open('./specs/hyperparameter-combinations.json','r') as f:
        params = json.load(f)
    files = glob.glob(os.path.join(os.path.abspath(path),f"{model}_*.txt"))

    max_sr, lr, model_dim, markers = list(), list(), list(), list()
    for file_path in files:
        idx = file_path.split('/')[-1].split('_')[1]
        with open(file_path, 'r') as f:
            lines = f.readlines()
        if "overall reward per step" in lines[-1] and params[idx]['attention_heads'] < 4:
            lr.append(params[idx]['learning_rate'])
            max_sr.append(max(extract_success_rate(lines)))
            model_dim.append(params[idx]['attention_head_size']*params[idx]['attention_heads'])
            markers.append(f"${params[idx]['attention_heads']}$")
            # print(f"Idx: {idx}, No. heads: {params[idx]['attention_heads']}, SR: {max_sr[-1]}")
        else:
            print(idx)

    plt.set_cmap('coolwarm')
    fig, ax = plt.subplots()
    ax.set_title(f"{model} single-layer")
    ax0 = ax.scatter(np.log(model_dim), np.log(lr),c=max_sr,vmin=0.5,vmax=1.)
    ax.set_ylabel('log Learning Rate')
    ax.set_xlabel('log Model Dimension')
    plt.colorbar(ax0, ax=ax)

    plt.tight_layout()
    # plt.savefig(f"./plots/{model}-correlation-single-layer.png",dpi=100)
    # plt.show()

def plot_hp_IDs():
    with open('./specs/hyperparameter-combinations.json','r') as f:
        params = json.load(f)

    for idx in range(89):
        p = params[str(idx)]
        model_dim = np.log(p['attention_head_size']*p['attention_heads'])
        lr = np.log(p['learning_rate'])
        plt.scatter(model_dim, lr, s=150, marker=f"${idx}$",c='black')
    plt.title("Hyperparameter ID mapping")
    plt.xlabel("log model dimension")
    plt.ylabel("log learning rate")
    plt.grid()
    plt.show()

def correlation_size(path, model):
    with open('./specs/hyperparameter-combinations.json','r') as f:
        params = json.load(f)
    files = glob.glob(os.path.join(os.path.abspath(path),f"{model}_*.txt"))

    max_sr, lr, head_size, num_heads = list(), list(), list(), list()
    for file_path in files:
        idx = file_path.split('/')[-1].split('_')[1]
        with open(file_path, 'r') as f:
            lines = f.readlines()
        if "overall reward per step" in lines[-1]:
            lr.append(params[idx]['learning_rate'])
            max_sr.append(max(extract_success_rate(lines)))
            head_size.append(params[idx]['attention_head_size'])
            num_heads.append(params[idx]['attention_heads']*10)
        else:
            print(idx)

    plt.set_cmap('coolwarm')
    fig, ax = plt.subplots()
    ax.set_title(f"{model}")
    ax0 = ax.scatter(np.log(lr), np.log(head_size), c=max_sr,vmin=.5,vmax=1.,s=num_heads)
    ax.set_ylabel('log Head Size')
    ax.set_xlabel('log LR')
    plt.colorbar(ax0, ax=ax)

    plt.tight_layout()
    # plt.savefig(f"./plots/{model}-correlation2.png",dpi=100)
    plt.show()

def few_head_results(path):
    with open('./specs/hyperparameter-combinations.json','r') as f:
        params = json.load(f)

    print(f"Config, NAP, NormalizedOriginal, Original")
    df = {
            'ID':[],
            'num_heads':[],
            'head_size':[],
            'learning_rate':[],
            'NAP':[],
            'NormalizedOriginal':[],
            'Original':[]}
    for i in range(len(params)):
        idx = str(i)
        if params[idx]['attention_heads'] >= 1:
            files = glob.glob(os.path.join(os.path.abspath(path),f"*_{idx}_*.txt"))
            assert len(files) == 3, print(files)
            max_sr, num_heads, models = list(), list(), list()
            output = f"{idx,params[idx]['attention_heads'],params[idx]['attention_head_size']}"
            df['ID'].append(i)
            df['num_heads'].append(params[idx]['attention_heads'])
            df['head_size'].append(params[idx]['attention_head_size'])
            df['learning_rate'].append(params[idx]['learning_rate'])
            for file_path in sorted(files):
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                if "overall reward per step" in lines[-1]:
                    models.append(file_path.split('/')[-1].split('_')[0])
                    num_heads.append(params[idx]['attention_heads'])
                    max_sr.append(max(extract_success_rate(lines)))
                    output += f", {max_sr[-1]}"
                    df[models[-1]].append(max_sr[-1])
                else:
                    models.append(file_path.split('/')[-1].split('_')[0])
                    df[models[-1]].append(0)
            print(output)
    df = pd.DataFrame.from_dict(df)
    df.set_index('ID', inplace=True)
    df.sort_values(by=['num_heads', 'head_size','learning_rate'],inplace=True)
    print(df)

def different_levels_single_layer(path):
    with open('./specs/hyperparameter-combinations-NAP.json','r') as f:
        params = json.load(f)

        df = {
                'ID':[],
                'num_heads':[],
                'head_size':[],
                'learning_rate':[],
                'NAP':[],
                'Training Time': [],
                'Steps': []}
        for i in range(50,50+len(params)):
            idx = str(i)
            files = glob.glob(os.path.join(os.path.abspath(path),f"*_{idx}_*.txt"))
            assert len(files) == 1, print(files)
            output = f"{idx,params[idx]['attention_heads'],params[idx]['attention_head_size']}"
            df['ID'].append(i)
            df['num_heads'].append(params[idx]['attention_heads'])
            df['head_size'].append(params[idx]['attention_head_size'])
            df['learning_rate'].append(params[idx]['learning_rate'])
            with open(files[0], 'r') as f:
                lines = f.readlines()
            if "overall reward per step" in lines[-1]:
                model = files[0].split('/')[-1].split('_')[0]
                num_heads =  params[idx]['attention_heads']
                max_sr = max(extract_success_rate(lines))
                output += f", {max_sr}"
                df[model].append(max_sr)
                t, s = extract_time_steps(lines)
                df['Training Time'].append(t)
                df['Steps'].append(s)
            else:
                model = files.split('/')[-1].split('_')[0]
                df[model].append(0)
            print(output)
        df = pd.DataFrame.from_dict(df)
        df.set_index('ID', inplace=True)
        df.sort_values(by=['num_heads', 'head_size','learning_rate'],inplace=True)
        print(df)
        print(df.describe())

def get_termination(error_lines):
    if error_lines:
        last_line = error_lines[-1]
        if "RuntimeError" in last_line:
            return "RuntimeError"
        elif "oom-kill" in last_line:
            return "Out of memory"
        elif "ImportError" in last_line:
            return "ImportError"
        elif "DUE TO TIME LIMIT" in last_line:
            return "Out of time"
        else:
            return "Success"
    else:
        return "Success"

def get_hp_df():
    df = []
    paths = glob.glob('./specs/random_grid/train_wmg_on_factored_babyai_Original_*.json')
    for json_path in paths:
        with open(json_path,'r') as f:
            spec = json.load(f)
        df.append(spec)
    df = pd.DataFrame(df)
    return df

def update_df(model_name, df, id, out_lines, termination):
    if id not in df['ID']:
        df['ID'].append(id)
        time, sr = extract_success_rate_and_time(out_lines)
        if sr:
            df[f'{model_name}-SR'].append(max(sr[:4*50]))
            df[f'{model_name}-steps'].append(int(min(len(sr)*250,5e4)))
            df[f'{model_name}-time'].append(time[:4*50][-1])
        else:
            df[f'{model_name}-SR'].append(0.0)
            df[f'{model_name}-steps'].append(0)
            df[f'{model_name}-time'].append(0.0)
        df[f'{model_name}-Comment'].append(termination)

def process_meta_files():

    # read err and output files
    paths = ["./results/201126_results/","./results/201217_results/"]
    err_files, out_files = [], []
    for p in paths:
        err_files += glob.glob(os.path.join(os.path.abspath(p),"*.err"))
        out_files += glob.glob(os.path.join(os.path.abspath(p),"*.out"))
    err_files.sort()
    out_files.sort()

    df_Original = {'ID':[],'Original-SR':[],'Original-time':[],'Original-steps':[],'Original-Comment':[]}
    df_NAP = {'ID':[],'NAP-SR':[],'NAP-time':[],'NAP-steps':[],'NAP-Comment':[]}
    df_BIAS_NAP = {'ID':[],'BIAS-NAP-SR':[],'BIAS-NAP-time':[],'BIAS-NAP-steps':[],'BIAS-NAP-Comment':[]}

    for out_f, err_f in tqdm(zip(out_files, err_files), total=len(err_files)):
        try:
            assert out_f[-10:-4] == err_f[-10:-4]
        except Exception as e:
            print(e)
            print("Out and Error does not match")
            print(out_f[-10:], err_f[-10:])
            exit(0)

        with open(err_f, 'r') as f:
            err_lines = f.readlines()

        termination = get_termination(err_lines)
        if termination == "ImportError":
            # edge case, forgot to activate conda environment
            continue

        with open(out_f, 'r') as f:
            out_lines = f.readlines()

        model = out_lines[4].split('/')[-1].split('_')[-2]
        id = int(out_lines[4].split('/')[-1].split('_')[-1].split('.')[0])
        out_file = []
        for p in paths:
            out_file += glob.glob(os.path.join(os.path.abspath(p),f"*{model}_{id}_*.txt"))

        for of in out_file:
            with open(of,'r') as f :
                out_lines = f.readlines()
            if model == 'Original':
                update_df(model, df_Original, id, out_lines, termination)
            elif model == 'NAP':
                if '4e-1' in of:
                    update_df('BIAS-NAP', df_BIAS_NAP, id, out_lines, termination)
                else:
                    update_df(model, df_NAP, id, out_lines, termination)

    df_Original = pd.DataFrame.from_dict(df_Original).sort_values(by=['ID'])
    df_NAP = pd.DataFrame.from_dict(df_NAP).sort_values(by=['ID'])
    df_BIAS_NAP = pd.DataFrame.from_dict(df_BIAS_NAP).sort_values(by=['ID'])
    df = pd.merge(df_Original, df_NAP, on='ID')
    df = pd.merge(df, df_BIAS_NAP, on='ID')
    json_df = get_hp_df()
    with pd.ExcelWriter("201217_random_grid_search.xlsx") as writer:
        df.to_excel(writer,sheet_name="results")
        json_df.to_excel(writer, sheet_name="hyperparameters")
    return 0

def make_df():
    hp = pd.read_excel(open("201126_random_grid_search.xlsx",'rb'),sheet_name="hyperparameters")
    res = pd.read_excel(open("201126_random_grid_search.xlsx",'rb'),sheet_name="results")

    hp = hp.drop(["LOAD_MODEL_FROM","TYPE_OF_RUN","SAVE_MODELS_TO","ENV","ENV_RANDOM_SEED","AGENT_RANDOM_SEED",
                "REPORTING_INTERVAL","TOTAL_STEPS","ANNEAL_LR","AGENT_NET","V2",
                "BABYAI_ENV_LEVEL","USE_SUCCESS_RATE","SUCCESS_RATE_THRESHOLD","HELDOUT_TESTING",
                "NUM_TEST_EPISODES","OBS_ENCODER","BINARY_REWARD","WEIGHT_DECAY","WMG_MAX_OBS",
                "WMG_TRANSFORMER_TYPE","REZERO","ID","Unnamed: 0"], axis=1)
    log_cols = ["AC_HIDDEN_LAYER_SIZE","ADAM_EPS","DISCOUNT_FACTOR",
                "GRADIENT_CLIP","LEARNING_RATE","REWARD_SCALE","WMG_ATTENTION_HEAD_SIZE",
                "WMG_MEMO_SIZE","WMG_HIDDEN_SIZE"]
    hp = hp.apply(lambda x: np.log(x) if x.name in log_cols else x)
    hp["ENTROPY_TERM_STRENGTH"] = hp["ENTROPY_TERM_STRENGTH"].apply(lambda x: np.log(x+1e-6))

    res = res.drop(["ID","Original-time","Original-steps","Original-Comment",
                    "NAP-time","NAP-steps","NAP-Comment","Unnamed: 0"],axis=1)

    df = pd.concat([hp, res],axis=1)
    df.to_csv("201126_random_grid_search_clean.csv")
    return df

def hp_correlation():
    df = pd.read_csv('201126_random_grid_search_clean.csv',index_col=False)
    df['MODEL_DIMENSION'] = np.log(df['WMG_NUM_ATTENTION_HEADS']) + df['WMG_ATTENTION_HEAD_SIZE']

    best_nap = df[df['NAP-SR']>0.95]
    best_nap = best_nap.drop(['Original-SR','Unnamed: 0'],axis=1)
    best_org = df[df['Original-SR']>0.95]
    best_org = best_org.drop(['NAP-SR','Unnamed: 0'],axis=1)

    print(len(best_nap))
    print(len(best_org))
    corr_nap = best_nap.corr()
    corr_org = best_org.corr()
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    mask_nap = (corr_nap > 0.3) | (corr_nap < -0.3)
    mask_org = (corr_org > 0.3) | (corr_org < -0.3)
    fig, ax = plt.subplots(1,2, figsize=(11,9))
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_nap, mask=~mask_nap, cmap=cmap, vmax=1., vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .25},ax=ax[0])
    ax[0].set_title(r'NAP Correlation SR > 0.95 (n=38), |$\rho$| > 0.3')
    sns.heatmap(corr_org, mask=~mask_org, cmap=cmap, vmax=1., vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .25},ax=ax[1])
    ax[1].set_title(r'Original Correlation SR > 0.95 (n=39), |$\rho$| > 0.3')
    ax[1].set_yticks([], [])
    plt.tight_layout()
    # plt.show()
    # plt.savefig("201210_Correlation.png",dpi=100)

def convert_and_dump(folder_path):
    experiment_name = "rl/var_dim/"
    variations = [{'name': 'head_dimension', 'min': 3, 'range': 8, 'base': 2},
                  {'name': 'learning_rate', 'min': 1, 'range': 10, 'base': 0.3}]
    params = {}

    try:
        os.makedirs(experiment_name)
        print("Directory ", experiment_name, " created ")
    except FileExistsError:
        print("Directory ", experiment_name, " already exists, reusing this directory.")
    from os import listdir
    from os.path import isfile, join
    import pickle

    pickle.dump([variations, params], open(experiment_name + 'NAP_params.pickle', 'wb'))
    pickle.dump([variations, params], open(experiment_name + 'Original_params.pickle', 'wb'))
    # NAP_results = np.ones((80, 6, 5, 4, 1)) * 0.99
    # Ori_results = np.ones((80, 6, 5, 4, 1)) * 0.99
    # NAP_results = np.zeros((32, 10, 8, 4, 1))
    # Ori_results = np.zeros((32, 10, 8, 4, 1))
    # NAP_results = np.zeros((80, 4, 4, 4, 1))
    # Ori_results = np.zeros((80, 4, 4, 4, 1))
    for model in ['Original', 'NAP', 'BERT', 'MTE', 'NON', 'MAX', 'SUM']:
        # all_results = np.zeros((32, 10, 8, 4, 1))
        all_results = np.zeros((32, 10, 8, 4, 5))
        model_found = False
        for file in listdir(folder_path):
            if model not in file:
                continue
            model_found = True
            if isfile(join(folder_path, file)):
                file_path = join(folder_path, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                results = extract_success_rate(lines)
                if results[-1] >= 0.99:
                    length = len(results)
                    for _ in range(32 - length):
                        results.append(0.99)
                run_id = int(file.split('_')[1])
                row = (run_id // 8) % 10
                column = run_id % 8
                seed = run_id // 80
                # row = run_id // 4
                # column = run_id % 4
                all_results[:len(results), row, column, 0, seed] = results
                all_results[:len(results), row, column, 1, seed] = results
                # else:
                #     Ori_results[:len(results), row, column, 0, 0] = results
        if model_found:
            pickle.dump([variations, params], open(experiment_name + model + '_params.pickle', 'wb'))
            # print(all_results)
            pickle.dump((all_results, 400, 0), open(experiment_name + model + '.pickle', 'wb'))
    # pickle.dump((Ori_results, 30, 0), open(experiment_name + 'Original.pickle', 'wb'))

def main():
    num_args = len(sys.argv) - 1
    if num_args != 1:
        print('Specify path to output folder.')
        exit(1)

    folder_path = sys.argv[1] # e.g. '../results/201009_moving_layer_norm/'

    ### Plot single learning curve ###
    # plot_single(folder_path)

    convert_and_dump(folder_path)

    ### Plot Average over all Learning Curves ###
    # plot_all_in_one(folder_path, "Original")

    ### Check which model won ###
    # who_won(folder_path)

    ### correlation
    # correlation2(folder_path, "NormalizedOriginal")

    ### rezero
    # plot_rezero()

    ### results for few heads across models
    # few_head_results(folder_path)

    # plot_hp_IDs()

    # different_levels_single_layer(folder_path)

    # process_meta_files(folder_path)

if __name__ == '__main__':
    main()
