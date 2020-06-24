import pickle 
import os
from copy import deepcopy
import shlex
import subprocess
import shutil
import pdb
from network import *
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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


def create_network(netfile=None, tripfile=None, odmultip=1):

    net = Network(netfile, tripfile)
    net.netfile = netfile
    net.tripfile = tripfile
    return net


def read_scenario(fname='ScenarioAnalysis.xlsx', sname='Moderate_1'):
    scenario_pd = pd.read_excel(fname, sname)
    dlinks = scenario_pd[scenario_pd['Link Condition'] == 1]['Link'].tolist()
    cdays = scenario_pd[scenario_pd['Link Condition'] == 1][
        'Closure day (day)'].tolist()

    damage_dict = {}
    for i in range(len(dlinks)):
        damage_dict[dlinks[i]] = cdays[i]
    return damage_dict


def net_update(net, args, flows=False):

    if flows:
        f = "flows.txt"
        file_created = False
        st = time.time()
        while not file_created:
            if os.path.exists(f):
                file_created = True

            if time.time()-st >10:
                popen = subprocess.call(args, stdout=subprocess.DEVNULL)

            if file_created:
                with open(f, "r") as flow_file:
                    for line in flow_file.readlines():
                        try:
                            ij = str(line[:line.find(' ')])
                            line = line[line.find(' '):].strip()
                            flow = float(line[:line.find(' ')])
                            line = line[line.find(' '):].strip()
                            cost = float(line.strip())
                            net.link[ij].flow = flow
                            net.link[ij].cost = cost
                        except:
                            break

                os.remove('flows.txt')





    try_again = False
    f = "full_log.txt"
    file_created = False
    while not file_created:
        if os.path.exists(f):
            file_created = True
        if file_created:
            with open(f, "r") as log_file:
                last_line = log_file.readlines()[-1]
                if last_line.find('obj') >= 0:

                    obj = last_line[last_line.find('obj') + 3:].strip()
                    try:
                        tstt = float(obj[:obj.find(',')])
                    except:
                        try_again = True
                else:
                    try_again = True


            idx_wanted = None
            if try_again:
                with open(f, "r") as log_file:
                    lines = log_file.readlines()
                    for idx, line in enumerate(lines):
                        if line[:4] == 'next':
                            idx_wanted = idx-1
                            break
                    last_line = lines[idx_wanted]        
                    obj = last_line[last_line.find('obj') + 3:].strip()
                    try:
                        tstt = float(obj[:obj.find(',')])
                    except:
                        try_again = True
            os.remove('full_log.txt')
    
    os.remove('current_net.tntp')

    return tstt


def solve_UE(net=None, relax=False, eval_seq=False, flows=False):

    # modify the net.txt file to send to c code
    shutil.copy(net.netfile, 'current_net.tntp')
    networkFileName = "current_net.tntp"



    df = pd.read_csv(networkFileName, 'r+', delimiter='\t')

    for a_link in net.not_fixed:
        home = a_link[a_link.find("'(") + 2:a_link.find(",")]
        to = a_link[a_link.find(",") + 1:]
        to = to[:to.find(")")]

        ind = df[(df['Unnamed: 1'] == str(home)) & (
            df['Unnamed: 2'] == str(to))].index.tolist()[0]
        df.loc[ind, 'Unnamed: 5'] = 1e8

    df.to_csv('current_net.tntp', index=False, sep="\t")

    f = 'current_net.tntp'
    file_created = False
    while not file_created:
        if os.path.exists(f):
            file_created = True

    if eval_seq:
        args = shlex.split("tap-b_real/bin/tap current_net.tntp " + net.tripfile)
    else:
        if relax:
            args = shlex.split("tap-b_relax/bin/tap current_net.tntp " + net.tripfile)
        else:
            args = shlex.split("tap-b/bin/tap current_net.tntp " + net.tripfile)

    popen = subprocess.run(args, stdout=subprocess.DEVNULL)

    tstt = net_update(net, args, flows)

    return tstt


def eval_sequence(net, order_list, after_eq_tstt, before_eq_tstt, if_list=None, importance=False, is_approx=False, damaged_dict=None):
    tap_solved = 0
    days_list = []
    tstt_list = []
    fp = None
    seq_list = []

    if importance:
        fp = []
        firstfp = 1
        for link_id in order_list:
            firstfp -= if_list[link_id]
        fp.append(firstfp * 100)
        curfp = firstfp

    # T = 0
    # for link_id in order_list:
    #     T += damaged_dict[link_id]

    level = 0
    prev_linkid = None
    tstt_before = after_eq_tstt

    to_visit = order_list
    added = []
    for link_id in order_list:
        level += 1
        days_list.append(damaged_dict[link_id])
        net.link[link_id].add_link_back()
        added.append(link_id)
        not_fixed = set(to_visit).difference(set(added))
        net.not_fixed = set(not_fixed)
        if is_approx:
            damaged_links = list(damaged_dict.keys())
            state = list(set(damaged_links).difference(net.not_fixed))
            state = [damaged_links.index(i) for i in state]
            pattern = np.zeros(len(damaged_links))
            pattern[[state]] = 1
            tstt_after = model.predict(pattern.reshape(1, -1)) * stdy + meany
        else:
            tap_solved += 1
            tstt_after = solve_UE(net=net, eval_seq=True)

        tstt_list.append(tstt_after)

        if importance:
            curfp += if_list[link_id]
            fp.append(curfp * 100)

    tot_area = 0
    for i in range(len(days_list)):
        if i == 0:
            tstt = after_eq_tstt
        else:
            tstt = tstt_list[i - 1]

        tot_area += (tstt - before_eq_tstt) * days_list[i]

    return tot_area, tap_solved, tstt_list




def get_marginal_tstts(net, path, after_eq_tstt, before_eq_tstt, damaged_dict):

    _, _, tstt_list = eval_sequence(
        deepcopy(net), path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict)

    # tstt_list.insert(0, after_eq_tstt)

    days_list = []
    for link in path:
        days_list.append(damaged_dict[link])

    return tstt_list, days_list






