from network import *
import numpy as np
import cProfile
import pstats
import pandas as pd
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
import os.path
from ctypes import *
import subprocess
import shutil
import time
from scipy import stats
import matplotlib._color_data as mcd
import caffeine

SEED = 9

SNAME = 'Moderate_5'
NETWORK = 'SiouxFalls'
NETFILE = "SiouxFalls/SiouxFalls_net.tntp"
TRIPFILE = "SiouxFalls/SiouxFalls_trips.tntp"
SAVED_FOLDER_NAME = "saved"

PROJECT_ROOT_DIR = "."

SAVED_DIR = os.path.join(PROJECT_ROOT_DIR, SAVED_FOLDER_NAME)
os.makedirs(SAVED_DIR, exist_ok=True)

NETWORK_DIR = os.path.join(SAVED_DIR, NETWORK)
os.makedirs(NETWORK_DIR, exist_ok=True)


def save(fname, data, extension='pickle'):
    path = fname + "." + extension

    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load(fname, extension='pickle'):
    path = fname + "." + extension

    with open(path, 'rb') as f:
        item = pickle.load(f)

    return item


def save_fig(plt_path, algo, tight_layout=True, fig_extension="png", resolution=300):
    plt_path = os.path.join(plt_path, "figures")
    os.makedirs(plt_path, exist_ok=True)
    path = os.path.join(plt_path, algo + "." + fig_extension)
    print("Saving figure", algo)

    if tight_layout:
        plt.tight_layout(pad=1)
    plt.savefig(path, format=fig_extension, dpi=resolution)


def create_network(netfile=None, tripfile=None):
    return Network(netfile, tripfile)


def read_scenario(fname='ScenarioAnalysis.xlsx', sname='Moderate_1'):
    scenario_pd = pd.read_excel(fname, sname)
    dlinks = scenario_pd[scenario_pd['Link Condition'] == 1]['Link'].tolist()
    cdays = scenario_pd[scenario_pd['Link Condition'] == 1][
        'Closure day (day)'].tolist()

    damage_dict = {}
    for i in range(len(dlinks)):
        damage_dict[dlinks[i]] = cdays[i]
    return damage_dict


def net_update(net):

    f = "flows.txt"
    with open(f, "r") as flow_file:
        for line in flow_file.readlines():
            ij = str(line[:line.find(' ')])
            line = line[line.find(' '):].strip()
            flow = float(line[:line.find(' ')])
            line = line[line.find(' '):].strip()
            cost = float(line.strip())
            net.link[ij].flow = flow
            net.link[ij].cost = cost

    f = "full_log.txt"
    with open(f, "r") as log_file:
        last_line = log_file.readlines()[-1]
        obj = last_line[last_line.find('obj') + 3:].strip()
        try:
            tstt = float(obj[:obj.find(',')])
        except:
            pdb.set_trace()
            
        log_file.close()

    return tstt


def solve_UE(net=None):

    # modify the net.txt file to send to c code
    
    shutil.copy('SiouxFalls/SiouxFalls_net.tntp', 'current_net.tntp')
    networkFileName = "current_net.tntp"

    df = pd.read_csv(networkFileName, 'r+', delimiter='\t')

    for a_link in net.not_fixed:
        home = a_link[a_link.find("'(")+2:a_link.find(",")]
        to = a_link[a_link.find(",")+1:]
        to = to[:to.find(")")]

        ind = df[(df['Unnamed: 1'] == str(home)) & (
        df['Unnamed: 2'] == str(to))].index.tolist()[0]
        df.loc[ind, 'Unnamed: 5'] = 1e9
        # print(a_link, df.loc[ind])
    df.to_csv('current_net.tntp', index=False, sep="\t")
    # send it to c code
    args = ("tap-b/bin/tap current_net.tntp SiouxFalls/SiouxFalls_trips.tntp 8", "-c")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE, shell=True)
    popen.wait()
    output = popen.stdout.read()
    # print(output)

    # receive the output from c and modify the net
    tstt = net_update(net)

    return tstt





