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
# from tabulate import tabulate
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
    """
    This method should take given values of path flows (stored in the
    self.path[].flow attributes), and do the following:
       1. Set link flows to correspond to these values (self.link[].flow)
       2. Set link costs based on new flows (self.link[].cost), see link.py
       3. Set path costs based on new link costs (self.path[].cost), see path.py
    """
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
    #### net.userEquilibrium("FW", 1e4, 1e-3, net.averageExcessCost)

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
    args = ("../../../tap_c/tap-b/bin/tap current_net.tntp SiouxFalls/SiouxFalls_trips.tntp", "-c")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE, shell=True)
    popen.wait()
    output = popen.stdout.read()
    # print(output)

    # receive the output from c and modify the net
    tstt = net_update(net)

    return tstt


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






















##### LOCAL SEARCH ON TOP OF GREEDY ALG #####
# import random
# import copy
# mynet = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
# for link_id in links_to_remove:
#     mynet.link[link_id].remove()

# solve_UE(net=mynet)
# after_eq_tstt = find_tstt(mynet)

# def swap_order(order_list):
#     # print(order_list)
#     first = random.sample(order_list,1)[0]
#     found = False
#     while not found:
#         second = random.sample(order_list,1)[0]
#         if second != first:
#             found = True
#     sec_ind = order_list.index(second)
#     order_list[order_list.index(first)] = second
#     order_list[sec_ind] = first
#     print(order_list)
#     return order_list

# max_iter = 5
# pdb.set_trace()
# # plt.show()
# origtsttlist = tstt_list
# print(order_list)
# for j in range(max_iter):
#     new_order_list = copy.copy(order_list)
#     new_order_list = swap_order(new_order_list)
#     for link_id in links_to_remove:
#         mynet.link[link_id].remove()
#     days_list_decoy, tstt_list_decoy, tot_area_decoy = solve_w_order(mynet, new_order_list, after_eq_tstt, before_eq_tstt)
#     if tot_area_decoy < tot_area:
#         order_list = new_order_list
#         days_list = days_list_decoy
#         tstt_list = tstt_list_decoy
#         tot_area = tot_area_decoy

# tstt_g = []
# for i in range(len(days_list)):
#     if i==0:
#         for j in range(days_list[i]):
#             tstt_g.append(after_eq_tstt)
#     else:
#         for j in range(days_list[i]):
#             tstt_g.append(tstt_list[i-1])

# y = tstt_g
# x = range(sum(days_list))

# # plt.subplot(213)
# # pdb.set_trace()
# plt.subplot(212)
# plt.fill_between(x,y, step="pre", color='blue', alpha=0.4)
# plt.step(x, y, label='tstt')
# plt.xlabel('Week')
# plt.ylabel('TSTT')
# plt.title('Greedy + local search')
# plt.legend()
# plt.savefig('TSTT')
# pdb.set_trace()
# plt.show()


# # damage_dict = read_scenarios('scenario.csv')
# damage_dict = {'(10,15)': 10, '(17,10)': 20}
# links_to_remove = damage_dict.keys()

# orig = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
# orig.userEquilibrium("FW", 1e4, 1e-4, orig.averageExcessCost)
# tx = 0
# for ij in orig.link:
#    tx += orig.link[ij].cost * orig.link[ij].flow
# orig_TSTT = tx

# print('orig done')


# ########
# after = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
# for link_id in links_to_remove:
#   after.link[link_id].remove()
# after.userEquilibrium("FW", 1e4, 1e-4, after.averageExcessCost)
# tx = 0
# for ij in after.link:
#    tx += after.link[ij].cost * after.link[ij].flow
# after_TSTT = tx

# ########

# recovered = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")

# for link_id in links_to_remove:
#   recovered.link[link_id].remove()

# links_to_add_back = ['(10,15)']
# for link_id in links_to_add_back:
#   recovered.link[link_id].add_link_back()

# recovered.userEquilibrium("FW", 1e4, 1e-4, recovered.averageExcessCost)

# tx = 0
# for ij in recovered.link:
#    tx += recovered.link[ij].cost * recovered.link[ij].flow

# recovered_TSTT = tx

# print("original_TSTT %f: after_EQ_TSTT %f: recovered_TSTT %f" % (orig_TSTT, after_TSTT, recovered_TSTT))

##### EARLIEST RELIEF FIRST STRATEGY#######

# for link_id in links_to_remove:
#     mynet.link[link_id].remove()

# solve_UE(net=mynet)
# after_eq_tstt = find_tstt(mynet)

# order_list = []

# sorted_d = sorted(damage_dict.items(), key=lambda x: x[1])
# for key, value in sorted_d:
#     print("%s: %s" % (key, value))
#     order_list.append(key)

# days_list, tstt_list, tot_area = solve_w_order(mynet, order_list, after_eq_tstt, before_eq_tstt)

# tstt_g = []
# for i in range(len(days_list)):
#     if i==0:
#         for j in range(days_list[i]):
#             tstt_g.append(after_eq_tstt)
#     else:
#         for j in range(days_list[i]):
#             tstt_g.append(tstt_list[i-1])

# y = tstt_g
# x = range(sum(days_list))

# plt.figure(1)
# plt.subplot(211)
# plt.fill_between(x,y, step="pre", color='red', alpha=0.4)
# plt.step(x, y, label='tstt')
# plt.xlabel('Week')
# plt.ylabel('TSTT')
# plt.title('Earliest relief first')
# plt.legend()
# # plt.show()

# print('early relief done...')


####### ANALYSIS FOR INSIGHTS #########
# snames = ['Strong_2']
# for sname in snames:
#     mydict = {}
#     damage_dict = read_scenario(sname=sname)

#     # BEFORE EQ
#     before = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
#     solve_UE(net=before)
#     before_eq_tstt = find_tstt(before)

#     tot_flow = 0
#     for ij in before.link:
#         tot_flow += before.link[ij].flow

#     ffp = 1
#     if_list = {}
#     for link_id in damage_dict.keys():
#         link_flow = before.link[link_id].flow
#         if_list[link_id] = link_flow / tot_flow

#     caps = []
#     fps = []
#     repair_days = []
#     costs = []
#     flows = []
#     ids = []
#     fc = []

#     for k, v in damage_dict.items():
#         caps.append(before.link[k].capacity)
#         repair_days.append(v)
#         flows.append(before.link[k].flow)
#         costs.append(before.link[k].cost)
#         ids.append(k)
#         fps.append(if_list[k])
#         fc.append(before.link[k].flow*before.link[k].cost)

#     headers = ['Link', 'Importance', 'Flow', 'Cost', 'F*C', 'Cap', 'Repair_Time' ]
#     table = zip(ids, fps, flows, costs, fc, caps, repair_days)
#     print(tabulate(table, headers=headers))

    # seq_dict = load('sequence_dict.pickle')

    # import operator
    # sorted_x = sorted(seq_dict.items(), key=operator.itemgetter(1))

    # print(tabulate(sorted_x, headers=['Nodeuence','Total_Cost']))
    # pdb.set_trace()


# net = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
# G = nx.DiGraph()

# node_pos = pd.read_csv('SiouxFalls_positions.csv')
# labels = {}

# for idx,row in node_pos.iterrows():
#   G.add_node(row['Node'], pos=(row['X'], row['Y']))
#   labels[row['Node']] = str(row['Node'])

# edge_list = []
# for alink in net.link:
#   edge_list.append((int(net.link[alink].tail), int(net.link[alink].head)))

# G.add_edges_from(edge_list)
# pos=nx.get_node_attributes(G,'pos')

# nx.draw(G,pos)
# nx.draw_networkx_labels(G, pos, labels, font_size=10)
# plt.show()


# df_places = gpd.read_file('SiouxFallsCoordinates.geojson')

# ax = df_places.plot(color='blue')

# for idx, row in df_places.iterrows():

#     coordinates = row['geometry'].coords.xy
#     x, y = coordinates[0][0], coordinates[1][0]
#     ax.annotate(row['id'], xy=(x, y), xytext=(x, y))

# for e in edge_list:
#   t = node_pos[node_pos['Node']==e[0]]
#   h = node_pos[node_pos['Node']==e[1]]

# plt.arrow(t['X'].values[0], t['Y'].values[0],
# h['X'].values[0]-t['X'].values[0], h['Y'].values[0]-t['Y'].values[0])

# solve_UE(net=net)

# for e in edge_list:
#   t = node_pos[node_pos['Node']==e[0]]
#   h = node_pos[node_pos['Node']==e[1]]
#   lnk = '(' + str(e[0]) +',' + str(e[1]) +')'
#   ax.annotate(str(net.link[lnk].flow), xy=(t['X'].values[0], t['Y'].values[0]), xytext=(h['X'].values[0]-t['X'].values[0]/2, h['Y'].values[0]-t['Y'].values[0]/2))
# for u,v,e in G.edges(data=True):
#   lnk = '(' + str(u) +',' + str(v) +')'
#   e['cost'] = round(net.link[lnk].cost,2)
#   e['flow'] = round(net.link[lnk].flow,2)
# cost_labels = nx.get_edge_attributes(G,'cost')
# flow_labels = nx.get_edge_attributes(G,'flow')
# nx.draw_networkx_edge_labels(G,pos,edge_labels = flow_labels)
# # nx.draw_networkx_edge_labels(G,pos,flow_labels)
# plt.show()
# pdb.set_trace()


# snames = ['Moderate_1', 'Strong_1']
# for sname in snames:
#     mydict = {}
#     damage_dict = read_scenario(sname=sname)
#     damaged_links = damage_dict.keys()
#     links_to_remove = damaged_links
#     print(links_to_remove)

#     mynet = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
#     solve_UE(net=mynet)
#     before_eq_tstt = find_tstt(mynet)

#     tot_flow = 0
#     for ij in mynet.link:
#         tot_flow += mynet.link[ij].flow

#     ffp = 1
#     if_list = {}
#     for link_id in damaged_links:
#         link_flow = mynet.link[link_id].flow
#         if_list[link_id] = link_flow/tot_flow
#         ffp -= if_list[link_id]

#     ffp=ffp*100

#     mynet = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
#     for link_id in links_to_remove:
#         mynet.link[link_id].remove()
#     solve_UE(net=mynet)
#     after_eq_tstt = find_tstt(mynet)
#     pdb.set_trace()
#     mydict['beforeTSTT'] = before_eq_tstt
#     mydict['afterTSTT'] = after_eq_tstt
#     mydict['beforeFP'] = 100
#     mydict['afterFP'] = ffp
#     print('ffp: ', ffp)
# print('before TSTT, after TSTT, diff TSTT, diffFP ',
# before_eq_tstt,after_eq_tstt,after_eq_tstt-before_eq_tstt,100-ffp)

#     df = pd.DataFrame({'b)-after': after_eq_tstt, 'a)-before': before_eq_tstt}, index=[''])
#     ax = df.plot.bar(rot=0)
#     ax.set_ylabel('TSTT')
#     # ax.set_xlabel('')
#     ax.set_title('TSTT (Before - After)')
#     fig = ax.get_figure()
#     fig.savefig(sname + "_tstt_before_after")

#     df = pd.DataFrame({'b)-after': ffp, 'a)-before': 100}, index=[''])
#     ax = df.plot.bar(rot=0)
#     ax.set_ylabel('Functionality')
#     ax.set_xlabel('')
#     ax.set_ylim(60)
#     ax.set_title('Network Functionality (Before - After)')
#     fig = ax.get_figure()
#     fig.savefig(sname + "_functionality_before_after")

#     save(mydict, sname + '_saved_variables')


# snames = ['Moderate_1']
# for sname in snames:
#     mydict = {}
#     damage_dict = read_scenario(sname=sname)

#     damaged_links = damage_dict.keys()
#     links_to_remove = ['(10,15)', '(6,2)', '(20,18)', '(18,20)']

#     # BEFORE EQ
#     before = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
#     solve_UE(net=before)
#     before_eq_tstt = find_tstt(before)

#     test = [['(10,15)', '(6,2)', '(20,18)'], ['(6,2)', '(10,15)', '(20,18)']]
#     T = 0
#     for link_id in test[0]:
#         T += damage_dict[link_id]

#     after = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
#     for link_id in test[0]:
#         after.link[link_id].remove()
#     solve_UE(net=after)
#     after_eq_tstt = find_tstt(after)

#     i = 0
#     cost1, seq_list1 = eval_sequence(
#         after, test[0], after_eq_tstt, before_eq_tstt)
#     for link_id in test[0]:
#         after.link[link_id].remove()
#     cost2, seq_list2 = eval_sequence(
#         after, test[1], after_eq_tstt, before_eq_tstt)

#     # only 1 fixed
#     marginal_1 = seq_list1[0].tstt_before - seq_list1[0].tstt_after
#     marginal_2 = seq_list1[1].tstt_before - seq_list1[1].tstt_after

#     link1 = seq_list1[0].link_id
#     link2 = seq_list1[1].link_id
#     effect = 0
#     # 1 and 2 fixed
#     net1 = seq_list1[0].net
#     net2 = seq_list1[1].net
#     combo = 0
#     for od in net2.ODpair:
#         origin = net2.ODpair[od].origin
#         backlink, cost = net2.shortestPath(
#             origin, wo=False, not_elig=None)

#         path = []
#         current = net2.ODpair[od].destination

#         while current != origin:
#             path.append(backlink[current])
#             current = before.link[backlink[current]].tail

#         if (link1 in path) and (link2 in path):
#             pdb.set_trace()
#             backlink, cost1 = net1.shortestPath(net1.ODpair[od].origin)
#             backlink, cost2 = net2.shortestPath(net2.ODpair[od].origin)
#             cost1 = cost1[net1.ODpair[od].destination]
#             cost2 = cost2[net2.ODpair[od].destination]

#             combo += cost1 - cost2

#     est = after_eq_tstt * damage_dict[link2] + (after_eq_tstt - (marginal_2 - combo)) * damage_dict[link1] + (
#         (after_eq_tstt - (marginal_2 - combo)) - (combo + marginal_1)) * T - (damage_dict[link1] + damage_dict[link2])

#     print('est', est)
#     print('real', seq_list2[1].ub)
#     pdb.set_trace()
    # for seq in test:

    # print(seq)

    # if i==0:
    #     after = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
    # for link_id in links_to_remove:
    #     after.link[link_id].remove()
    # if i==0:
    #     solve_UE(net=after)
    #     after_eq_tstt = find_tstt(after)
    #     print('hoppala')

    # cost = eval_sequence(after, seq, after_eq_tstt, before_eq_tstt)
    # seq_dict[seq] = cost

    # print(i)
    # print(cost)
    # print(seq)


# snames = ['Moderate_1']
# for sname in snames:
#     mydict = {}
#     damage_dict = read_scenario(sname=sname)
#     j = 0
#     dell = []
#     for key in damage_dict.keys():
#         dell.append(key)
#         if j == 3:
#             break
#         j += 1
#     for key in dell:
#         del damage_dict[key]

#     damaged_links = damage_dict.keys()
#     links_to_remove = damaged_links

#     # BEFORE EQ
#     before = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
#     solve_UE(net=before)
#     before_eq_tstt = find_tstt(before)

#     # for l in links_to_remove:

#     #     for i in range(len(links_to_remove)):
#     #     if len(fixed) == len(links_to_remove):
#     #         break
#     #     # diff links to remove and fixed
#     #     links_eligible_addback = list(set(links_to_remove) - set(fixed))
#     #     link_id, tstt, days = find_max_relief(
#     #         links_eligible_addback, fixed, cur_tstt)
#     #     cur_tstt = tstt
#     #     days_list.append(days)
#     #     tstt_list.append(tstt)
#     #     print(i, link_id)
#     #     mynet.link[link_id].add_link_back()
#     #     fixed.append(link_id)

#     print('hop')

#     import itertools
#     all_sequences = list(itertools.permutations(links_to_remove))

#     seq_dict = {}
#     i = 0
#     min_cost = 1000000000000000
#     min_seq = None
#     for seq in all_sequences:
#         if i == 0:
#             after = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
#         for link_id in links_to_remove:
#             after.link[link_id].remove()
#         if i == 0:
#             solve_UE(net=after)
#             after_eq_tstt = find_tstt(after)
#             print('hoppala')

#         cost = eval_sequence(after, seq, after_eq_tstt, before_eq_tstt)
#         seq_dict[seq] = cost

#         if cost < min_cost:
#             min_cost = cost
#             min_seq = seq

#         i += 1

#         print(i)
#         print(cost)
#         print(seq)
#     save(seq_dict, 'sequence_dict')
#     pdb.set_trace()
#     import operator
#     sorted_x = sorted(seq_dict.items(), key=operator.itemgetter(1))

#     # TEST
#     after = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
#     # i = 0
#     for link_id in links_to_remove:
#         after.link[link_id].remove()

#     solve_UE(net=after)

#     bridge_effect_c = {}
#     bridge_effect_f = {}
