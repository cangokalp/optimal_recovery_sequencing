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

SCENARIO_DIR = os.path.join(NETWORK_DIR, SNAME)
os.makedirs(SCENARIO_DIR, exist_ok=True)


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
        tstt = float(obj[:obj.find(',')])
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
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)



def graph_current(tstt_state, days_state, before_eq_tstt, after_eq_tstt, path, plt_path, algo, together, place):

    N = sum(days_state)

    cur_tstt = after_eq_tstt

    tstt_g = []
    tot_area = 0
    days = 0

    for i in range(len(days_state)):
        days += days_state[i]
        if i == 0:
            tot_area += days_state[i] * (after_eq_tstt - before_eq_tstt)
            tstt_g.append(after_eq_tstt)
        else:
            tot_area += days_state[i] * (tstt_state[i - 1] - before_eq_tstt)
            tstt_g.append(tstt_state[i - 1])

    
    tstt_g.append(before_eq_tstt)
    y = tstt_g
    x = np.zeros(len(y))

    for j in range(len(y)):
        x[j] = sum(days_state[:j])



    if together:
        plt.subplot(place)
    else:
        plt.figure(figsize=(16,8))

    plt.fill_between(x, y, before_eq_tstt, step="post",
                     color='green', alpha=0.2, label='TOTAL AREA:' + '{0:1.5e}'.format(tot_area))
    # plt.step(x, y, label='tstt', where="post")
    plt.xlabel('DAYS')
    plt.ylabel('TSTT')

    plt.ylim((before_eq_tstt, after_eq_tstt + after_eq_tstt*0.07))


    for i in range(len(tstt_state)):

        start = sum(days_state[:i])

        bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="b", alpha=0.8, lw=1)
        t = plt.text(start + N*0.01, y[i] + before_eq_tstt*0.01, "link: " + path[i] + '\n' + "time: " + str(round(days_state[i],2)) , ha='left', size=8,
            bbox=bbox_props)
        bb = t.get_bbox_patch()

        # start + days_state[i]/2
        # plt.annotate("link: " + path[i], (start, y[i]), textcoords='offset points', xytext=(5,15), ha='left', size='smaller') 
        # plt.annotate("time: " + str(round(days_state[i],2)), (start, y[i]), textcoords='offset points', xytext=(5,5), ha='left', size='smaller') #, arrowprops=dict(width= days_state[i])) 
        
        # plt.annotate("", xy=(start + days_state[i], y[i]), xytext=(start, y[i]) , textcoords='offset points', ha='center', arrowprops=dict(arrowstyle='<->', connectionstyle="arc3"))
        # plt.arrow(0.85, 0.5, dx = -0.70, dy = 0, head_width=0.05, head_length=0.03, linewidth=4, color='g', length_includes_head=True)
        
        plt.annotate(s='', xy=(start + days_state[i], y[i]), xytext=(start, y[i]), arrowprops=dict(arrowstyle='<->', color='blue'))
        plt.annotate(s='', xy=(start + days_state[i], y[i]), xytext=(start + days_state[i], y[i+1]), arrowprops=dict(arrowstyle='<->', color='indigo'))

        # plt.annotate("" + str(round(y[i] - y[i+1], 2)), xy=(start + days_state[i], (y[i] - y[i+1])/2), xytext=(10,10), textcoords='offset points', ha='left') #, arrowprops=dict(width= days_state[i])) 

        #animation example:
        # import numpy as np
        # import matplotlib.pyplot as plt
        # import matplotlib.animation as animation

        # fig, ax = plt.subplots()
        # ax.axis([-2,2,-2,2])

        # arrowprops=dict(arrowstyle='<-', color='blue', linewidth=10, mutation_scale=150)
        # an = ax.annotate('Blah', xy=(1, 1), xytext=(-1.5, -1.5), xycoords='data', 
        #                  textcoords='data', arrowprops=arrowprops)

        # colors=["crimson", "limegreen", "gold", "indigo"]
        # def update(i):
        #     c = colors[i%len(colors)]
        #     an.arrow_patch.set_color(c)

        # ani = animation.FuncAnimation(fig, update, 10, interval=1000, repeat=True)
        # plt.show()


        # plt.text(0, 0.1, r'$\delta$',
        #  {'color': 'black', 'fontsize': 24, 'ha': 'center', 'va': 'center',
        #   'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
        # xticks(np.arange(5), ('Tom', 'Dick', 'Harry', 'Sally', 'Sue'))
        
        # using tex in labels
        # Use tex in labels
        # plt.xticks((-1, 0, 1), ('$-1$', r'$\pm 0$', '$+1$'), color='k', size=20)

        # Left Y-axis labels, combine math mode and text mode
        # plt.ylabel(r'\bf{phase field} $\phi$', {'color': 'C0', 'fontsize': 20})
        # plt.yticks((0, 0.5, 1), (r'\bf{0}', r'\bf{.5}', r'\bf{1}'), color='k', size=20)
    
    plt.title(algo, fontsize=16)
    plt.legend(loc='upper right', fontsize=14)
    
    if not together:    
        save_fig(plt_path, algo)
        plt.clf()





def mean_std_lists(metric_values, sample_size):
    t_critical = stats.t.ppf(q = 0.9, df=sample_size)
    
    mean = np.mean(metric_values)
    stdev = np.std(metric_values)    # Get the sample standard deviation
    sigma = stdev/np.sqrt(sample_size)  # Standard deviation estimate
    error = t_critical * sigma

    return mean, error    

def prep_dictionaries(method_dict):
    method_dict['_obj'] = []
    method_dict['_num_tap'] = []
    method_dict['_elapsed'] = []

def result_table(reps, file_path):

    filenames = ['algo_solution', 'min_seq', 'greedy_solution', 'importance_factor_bound'] 

    heuristic = {}
    greedy = {}
    brute_force = {}
    importance_factor = {}

    dict_list = [heuristic, brute_force, greedy, importance_factor]
    key_list = ['_obj', '_num_tap', '_elapsed']

    for method_dict in dict_list:
        prep_dictionaries(method_dict)

    for rep in range(reps):
        for method_dict in dict_list:
            for key in key_list:
                method_dict[key].append(load(os.path.join(file_path, str(rep)) + '/' + filenames[dict_list.index(method_dict)] + key))

    sample_size = reps


    obj_means = []
    tap_means = []
    elapsed_means = []

    obj_err = []
    tap_err = []
    elapsed_err = []


    optimal = np.array(brute_force['_obj'])

    for method_dict in dict_list:
        if method_dict == brute_force:
            continue

        objs = np.array(method_dict['_obj'])
        objs = objs - optimal
        objs = np.maximum(0, objs)
        mean, error = mean_std_lists(objs, sample_size)
        obj_means.append(mean)
        obj_err.append(error)

        taps = method_dict['_num_tap']
        mean, error = mean_std_lists(taps, sample_size)
        tap_means.append(mean)
        tap_err.append(error)

        elapsed_values = method_dict['_elapsed']
        mean, error = mean_std_lists(elapsed_values, sample_size)
        elapsed_means.append(mean)
        elapsed_err.append(error)

    obj_means_scaled = obj_means/max(obj_means)
    obj_err_scaled = obj_err/max(obj_means)

    tap_means_scaled = tap_means/max(tap_means)
    tap_err_scaled = tap_err/max(tap_means)

    elapsed_means_scaled = elapsed_means/max(elapsed_means)
    elapsed_err_scaled = elapsed_err/max(elapsed_means)

    plt.figure()

    barWidth = 0.2
    r_obj = np.arange(len(obj_means))
    r_tap = [x + barWidth for x in r_obj]
    r_elapsed = [x + 2*barWidth for x in r_obj]

    # hatch='///', hatch='\\\\\\', hatch='xxx'
    plt.bar(r_obj, obj_means_scaled, width = barWidth, edgecolor = 'black', color='#087efe', ecolor='#c6ccce', alpha=0.8, yerr=obj_err_scaled, capsize=7, label='Avg (Method - Optimal)')
    plt.bar(r_tap, tap_means_scaled, width = barWidth, edgecolor = 'black', color='#b7fe00', ecolor='#c6ccce', alpha=0.8, yerr=tap_err_scaled, capsize=7, label='Avg Tap Solved')
    plt.bar(r_elapsed, elapsed_means_scaled, width = barWidth, edgecolor = 'black', color='#ff9700', ecolor='#c6ccce', alpha=0.8, yerr=elapsed_err_scaled, capsize=7, label='Avg Time Elapsed (s)')
    

    # tap_means_scaled[i]/2
    dict_list.remove(brute_force)
    for i in range(len(dict_list)):
        plt.annotate('{0:1.1e}'.format(obj_means[i]), (i , 0), textcoords='offset points', xytext=(0,20), ha='center', va='bottom', rotation=70)
        plt.annotate('{0:1.1f}'.format(tap_means[i]), (i + barWidth, 0), textcoords='offset points', xytext=(0,20), ha='center', va='bottom', rotation=70)  
        plt.annotate('{0:1.1f}'.format(elapsed_means[i]), (i + 2*barWidth, 0), textcoords='offset points', xytext=(0,20), ha='center', va='bottom', rotation=70) 



    plt.ylabel('Normalized Metric Value')
    plt.xticks([(r + barWidth) for r in range(len(obj_means))], ['Heuristic', 'Greedy', 'Importance_Factor'])
    plt.xlabel('Method')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig('plots/' + 'averaged_results')






    # import numpy as np
    # import matplotlib.pyplot as plt


    # data = [[ 66386, 174296,  75131, 577908,  32015],
    #         [ 58230, 381139,  78045,  99308, 160454],
    #         [ 89135,  80552, 152558, 497981, 603535],
    #         [ 78415,  81858, 150656, 193263,  69638],
    #         [139361, 331509, 343164, 781380,  52269]]

    # columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
    # rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

    # values = np.arange(0, 2500, 500)
    # value_increment = 1000

    # # Get some pastel shades for the colors
    # colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    # n_rows = len(data)

    # index = np.arange(len(columns)) + 0.3
    # bar_width = 0.4

    # # Initialize the vertical-offset for the stacked bar chart.
    # y_offset = np.zeros(len(columns))

    # # Plot bars and create text labels for the table
    # cell_text = []
    # for row in range(n_rows):
    #     plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    #     y_offset = y_offset + data[row]
    #     cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
    # # Reverse colors and text labels to display the last value at the top.
    # colors = colors[::-1]
    # cell_text.reverse()

    # # Add a table at the bottom of the axes
    # the_table = plt.table(cellText=cell_text,
    #                       rowLabels=rows,
    #                       rowColours=colors,
    #                       colLabels=columns,
    #                       loc='bottom')

    # # Adjust layout to make room for the table:
    # plt.subplots_adjust(left=0.2, bottom=0.2)

    # plt.ylabel("Loss in ${0}'s".format(value_increment))
    # plt.yticks(values * value_increment, ['%d' % val for val in values])
    # plt.xticks([])
    # plt.title('Loss by Disaster')

    # plt.show()

def get_tables():

    broken_bridges = ['7']
    repetitions = [10]

    for broken in broken_bridges:

        ULT_SCENARIO_DIR = os.path.join(SCENARIO_DIR, broken)
        reps = repetitions[broken_bridges.index(broken)]

        result_table(reps, ULT_SCENARIO_DIR)


    pdb.set_trace()

# sname = 'Moderate_5'
# # for sname in snames:
# mydict = {}
# damage_dict = read_scenario(sname=sname)

# damaged_links = damage_dict.keys()
# alldays = damage_dict.values()

# N = 0
# for i in alldays:
# 	N += i
# links_to_remove = damaged_links

# # Find before earthquake equilibrium
# # before = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
# # solve_UE(net=before)
# # before_eq_tstt = find_tstt(before)

# # Find after earthquake equilibrium
# # after = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
# # for link in links_to_remove:
# # 	after.link[link].remove()
# # solve_UE(net=after)
# # after_eq_tstt = find_tstt(after)

# ###### WORSE BENEFIT ANALYSIS #######

# # if analysis haven't been done yet
# if not os.path.exists('saved_dictionaries/' + 'worst_benefit_dict' +
# sname + '.pickle'):

# 	# for each bridge, find the effect on TSTT when that bridge is removed
# 	# while keeping others
# 	wb = {}
# 	for link in links_to_remove:
# 		test_net = deepcopy(before)
# 		test_net.link[link].remove()
# 		solve_UE(net=test_net)
# 		wb[link] = find_tstt(test_net) - before_eq_tstt
# 	# save dictionary
# 	save(wb, 'worst_benefit_dict' + sname)

# else:
# 	wb = load('worst_benefit_dict' + sname)

# ###### BEST BENEFIT ANALYSIS #######
# seq_list = []
# # if analysis haven't been done yet:
# if not os.path.exists('saved_dictionaries/' + 'best_benefit_dict' +
# sname + '.pickle'):

# 	# for each bridge, find the effect on TSTT when that bridge is removed
# 	# while keeping others
# 	bb = {}

# 	for link in links_to_remove:
# 		test_net = deepcopy(after)
# 		test_net.link[link].add_link_back()
# 		solve_UE(net=test_net)
# 		tstt_after = find_tstt(test_net)

# 		seq_list.append(Node(link_id=link, parent=None, net=test_net, tstt_after=tstt_after, tstt_before=after_eq_tstt, level=1, damaged_dict=damage_dict))
# 		bb[link] = after_eq_tstt - tstt_after
# 	save(bb, 'best_benefit_dict' + sname)
# 	save(seq_list, 'seq_list' + sname)


# else:
# 	bb = load('best_benefit_dict' + sname)
# 	# seq_list = load('seq_list' + sname)
# ###### FIND PRECEDENCE RELATIONSHIPS ######
# if not os.path.exists('saved_dictionaries/' + 'precedence_dict' + sname
# + '.pickle'):

# 	precedence = {} #if 1: 3,4 means 1 has to come before 3 and also 4
# 	following = {} #if 3: 1,2 means 3 has to come after 1 and also 2

# 	for a_link in links_to_remove:
# 		for other in links_to_remove:
# 			if a_link != other:
# 				if wb[a_link] * damage_dict[other] - bb[other] * damage_dict[a_link] > 0:
# 					if a_link in precedence.keys():
# 						precedence[a_link].append(other)
# 					else:
# 						precedence[a_link] = [other]
# 					if other in following.keys():
# 						following[other].append(a_link)
# 					else:
# 						following[other] = [a_link]

# 	save(precedence, 'precedence_dict'+ sname)
# 	save(following, 'following_dict'+ sname)
# else:
# 	precedence = load('precedence_dict'+ sname)
# 	following = load('following_dict'+ sname)

# # start sequences
# ## first pruning by precedence, if 2 has to follow something you cannot start with 2
# fathomed_seqlist = []
# for seq in seq_list:
# 	if seq.link_id in following.keys():
# 		fathomed_seqlist.append(seq)

# #TODO
# # implemennt 2 pair elimination 1-2 vs 2-1 without solving
# # implement that if 3-2-1 solved 3-1-2 you can get the solution from the other

# alive_list = list(set(seq_list) - set(fathomed_seqlist))
# ### expand sequences here

# level = 1
# print('max_length: ', len(damaged_links))
# tot_solved = 2*len(damaged_links) + 2
# while level < len(damaged_links):
# 	level += 1
# 	new_seqs = []
# 	for aseq in alive_list:
# 		possible_additions = list(set(damaged_links)-set(aseq.path))
# 		following_keys = [i for i in following.keys()]
# 		for j in possible_additions:
# 			if j in following_keys:
# 				follows = following[j]
# 				if len(set(follows).difference(set(aseq.path))) == 0:
# 					seq = expand_seq(seq=aseq, lad=j, level=level)
# 					new_seqs.append(seq)
# 			else:
# 				seq = expand_seq(seq=aseq, lad=j, level=level)
# 				new_seqs.append(seq)

# 	seq_list = new_seqs
# 	print('-------')
# 	print(level)
# 	counting = 0
# 	for i in seq_list:
# 		counting +=1
# 		print('level ' + str(level) + ', seq ' + str(counting) + ':' + '\n')
# 		print(i.path)

# 	tot_solved += counting
# 	print('length of sequence list before pruning: ', len(seq_list))
# 	alive_list = prune(seq_list)
# 	print('pruning completed')

# 	counting = 0
# 	for i in alive_list:
# 		counting +=1
# 		print('alive in level ' + str(level) + ', seq ' + str(counting) + ':' + '\n')
# 		print(i.path)
# 	print('length of alive sequences: ', len(alive_list))


# pdb.set_trace()
# print('# TAP solved: ', tot_solved)
# print(alive_list[0].path)
# save(alive_list[0], 'dp_soln' + sname)


# Compare it to importance factor

    # tot_flow = 0
    # for ij in before.link:
    #     tot_flow += before.link[ij].flow
    # ffp = 1
    # if_list = {}

    # for link_id in damaged_links:
    #     link_flow = before.link[link_id].flow
    #     if_list[link_id] = (link_flow / tot_flow)/(N-damage_dict[link_id])
    #     ffp -= if_list[link_id]

    # import itertools
    # all_sequences = list(itertools.permutations(links_to_remove))

    # fixed_state = []
    # tstt_state = []
    # days_state = []

    # to_fix = '(10,15)'
    # updated_net, fixed_state, tstt_state, days_state = fix_one(
    # 	after, fixed_state, to_fix, tstt_state, days_state)
    # print('fixed, graphing now..')
    # # graph_current(updated_net, fixed_state, tstt_state, days_state, before_eq_tstt, after_eq_tstt, N)

    # pdb.set_trace()

    # to_fix = '(18,20)'
    # updated_net, fixed_state, tstt_state, days_state = fix_one(
    # 	updated_net, fixed_state, to_fix, tstt_state, days_state)
    # graph_current(updated_net, fixed_state, tstt_state,
    # 			  days_state, before_eq_tstt, after_eq_tstt, N)

    # pdb.set_trace()
    # to_fix = '(11,14)'
    # updated_net, fixed_state, tstt_state, days_state = fix_one(
    # 	updated_net, fixed_state, to_fix, tstt_state, days_state)
    # graph_current(updated_net, fixed_state, tstt_state,
    # 			  days_state, before_eq_tstt, after_eq_tstt, N)

    # fixed_state = []
    # tstt_state = []
    # days_state = []

    # to_fix = '(18,20)'
    # updated_net, fixed_state, tstt_state, days_state = fix_one(
    # 	after, fixed_state, to_fix, tstt_state, days_state)
    # graph_current(updated_net, fixed_state, tstt_state,
    # 			  days_state, before_eq_tstt, after_eq_tstt, N)

    # pdb.set_trace()

    # to_fix = '(10,15)'
    # updated_net, fixed_state, tstt_state, days_state = fix_one(
    # 	updated_net, fixed_state, to_fix, tstt_state, days_state)
    # graph_current(updated_net, fixed_state, tstt_state,
    # 			  days_state, before_eq_tstt, after_eq_tstt, N)

    # pdb.set_trace()
    # to_fix = '(11,14)'
    # updated_net, fixed_state, tstt_state, days_state = fix_one(
    # 	updated_net, fixed_state, to_fix, tstt_state, days_state)
    # graph_current(updated_net, fixed_state, tstt_state,
    # 			  days_state, before_eq_tstt, after_eq_tstt, N)
    # pdb.set_trace()

    # seq_dict = {}
    # i = 0
    # min_cost = 1000000000000000
    # min_seq = None

    # cost, _ = eval_sequence(after, seq, after_eq_tstt, before_eq_tstt)
    # seq_dict[seq] = cost

    # if cost < min_cost:
    # 	min_cost = cost
    # 	min_seq = seq

    # i += 1

    # print(i)
    # print(cost)
    # print(seq)
    # save(seq_dict, 'sequence_dict')
    # import operator
    # sorted_x = sorted(seq_dict.items(), key=operator.itemgetter(1))
    # pdb.set_trace()

    # # TEST
    # after = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
    # # i = 0
    # for link_id in links_to_remove:
    # 	after.link[link_id].remove()

    # solve_UE(net=after)

    # bridge_effect_c = {}
    # bridge_effect_f = {}

    # for link_id in links_to_remove:

    # 	for od in before.ODpair:
    # 		origin = before.ODpair[od].origin
    # 		backlink, cost = before.shortestPath(
    # 			origin, wo=True, not_elig=link_id)
    # 		current = before.ODpair[od].destination

    # 		while current != origin:
    # 			# denomcost += decoy.link[backlink[current]].calculateCost()
    # 			# na_sum += decoy.link[backlink[current]].monetary
    # 			# if link_id in bridge_effect_ol.keys()
    # 			#         bridge_effect_ol[link_id] += 1
    # 			#     else:
    # 			#         bridge_effect_ol[link_id] = 1
    # 			if backlink[current] in links_to_remove:
    # 				if link_id in bridge_effect_c.keys():
    # 					bridge_effect_c[link_id] += 1
    # 				else:
    # 					bridge_effect_c[link_id] = 1
    # 				if link_id in bridge_effect_f.keys():
    # 					bridge_effect_f[link_id] += before.ODpair[od].demand
    # 				else:
    # 					bridge_effect_f[link_id] = before.ODpair[od].demand

    # 			current = before.link[backlink[current]].tail

    # for k, v in bridge_effect_f.items():
    # 	bridge_effect_f[k] = damage_dict[k] * v

    # tot_flow = 0
    # for ij in before.link:
    # 	tot_flow += before.link[ij].flow

    # ffp = 1
    # if_list = {}
    # for link_id in damaged_links:
    # 	link_flow = before.link[link_id].flow
    # 	if_list[link_id] = link_flow / tot_flow
    # 	ffp -= if_list[link_id]

    # ffp = ffp * 100

    # sorted_d = sorted(if_list.items(), key=lambda x: x[1])
    # if_order, if_importance = zip(*sorted_d)
    # if_order = if_order[::-1]
    # if_importance = if_importance[::-1]
    # print('if_order: ', if_order)
    # print('if_importance: ', if_importance)

    # pdb.set_trace()

    # b = []
    # a = []
    # iL = []
    # for ij in before.link:
    # 	if ij in links_to_remove:
    # 		b.append(before.link[ij].flow)
    # 		a.append(after.link[ij].flow)
    # 		iL.append(ij)

    # b = np.array(b)
    # a = np.array(a)
    # diff = abs(b - a)

    # b = list(b)
    # a = list(a)
    # diff = list(diff)

    # diff, b, a, iL = (list(t) for t in zip(*sorted(zip(diff, b, a, iL))))

    # diff = diff[::-1]
    # b = b[::-1]
    # a = a[::-1]
    # iL = iL[::-1]

    # cut = -1
    # df = pd.DataFrame({'a)-before': b[:cut],
    # 				   'b)-after': a[:cut]}, index=iL[:cut])
    # ax = df.plot.bar(rot=0)
    # ax.set_ylabel('flow')
    # ax.set_xlabel('i-j Links')
    # ax.set_title('Bridge link flow changes (Before - After)')
    # fig = ax.get_figure()
    # fig.savefig("flow changes")
    # pdb.set_trace()

    # # AFTER EQ
    # after = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
    # for link_id in links_to_remove:
    # 	after.link[link_id].remove()

    # solve_UE(net=after)
    # after_eq_tstt = find_tstt(after)

    # tot_flow = 0
    # for ij in mynet.link:
    #     tot_flow += mynet.link[ij].flow

    # ffp = 1
    # if_list = {}
    # for link_id in damaged_links:
    #     link_flow = mynet.link[link_id].flow
    #     if_list[link_id] = link_flow/tot_flow
    #     ffp -= if_list[link_id]

    # ffp=ffp*100

    # sorted_d = sorted(if_list.items(), key=lambda x: x[1])
    # if_order, if_importance = zip(*sorted_d)
    # if_order = if_order[::-1]
    # if_importance = if_importance[::-1]
    # print('if_order: ', if_order)
    # print('if_importance: ', if_importance)

    ##### MAX RELIEF FIRST STRATEGY - GREEDY ######

    # mynet = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
    # for link_id in links_to_remove:
    # 	mynet.link[link_id].remove()

    # solve_UE(net=mynet)
    # after_eq_tstt = find_tstt(mynet)
    # mydict['beforeTSTT'] = before_eq_tstt
    # mydict['afterTSTT'] = after_eq_tstt
    # mydict['afterFP'] = ffp
    # days_list = []
    # tstt_list = []
    # fixed = []
    # cur_tstt = after_eq_tstt

    # for i in range(len(links_to_remove)):
    # 	if len(fixed) == len(links_to_remove):
    # 		break
    # 	# diff links to remove and fixed
    # 	links_eligible_addback = list(set(links_to_remove) - set(fixed))
    # 	link_id, tstt, days = find_max_relief(
    # 		links_eligible_addback, fixed, cur_tstt)
    # 	cur_tstt = tstt
    # 	days_list.append(days)
    # 	tstt_list.append(tstt)
    # 	print(i, link_id)
    # 	mynet.link[link_id].add_link_back()
    # 	fixed.append(link_id)

    # order_list = fixed
    # tstt_g = []
    # tot_area = 0
    # for i in range(len(days_list)):
    # 	if i == 0:
    # 		tot_area += days_list[i] * (after_eq_tstt - before_eq_tstt)
    # 		# for j in range(int(days_list[i])):
    # 		tstt_g.append(after_eq_tstt)
    # 	else:
    # 		tot_area += days_list[i] * (tstt_list[i - 1] - before_eq_tstt)
    # 		# for j in range(int(days_list[i])):
    # 		tstt_g.append(tstt_list[i - 1])

    # # pdb.set_trace()
    # y = [after_eq_tstt] + tstt_g + [tstt_g[-1]]
    # x = [0] + days_list

    # for j in range(len(days_list)):
    # 	x[j] = days_list[j]
    # 	if j != 0:
    # 		x[j] = sum(days_list[:j])

    # x[0] = 0
    # x[j + 1] = sum(days_list[:j + 1])
    # x.append(x[-1])

    # # x = days_list
    # # plt.subplot(211)
    # plt.figure()
    # plt.fill_between(x, y, before_eq_tstt, step="pre",
    # 				 color='green', alpha=0.4)
    # plt.step(x, y, label='tstt')
    # plt.xlabel('Days')
    # plt.ylabel('TSTT')
    # plt.title('Maximum relief first')
    # tt = 'Total Area: ' + str(tot_area)
    # xy = (0.2, 0.2)
    # plt.annotate(tt, xy, xycoords='figure fraction')
    # plt.legend()
    # plt.savefig(sname + '_MRF_TSTT')
    # print('max relief area: ', tot_area)

    # pdb.set_trace()
    # ####### Network Functionality ######

    # mynet = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
    # for link_id in links_to_remove:
    # 	mynet.link[link_id].remove()
    # solve_UE(net=mynet)
    # after_eq_tstt = find_tstt(mynet)

    # days_list_functionality, tstt_list_functionality, _, fp = eval_sequence(
    # 	mynet, if_order, after_eq_tstt, before_eq_tstt, if_list)
    # # pdb.set_trace()

    # y = [ffp] + fp
    # x = [0] + days_list_functionality

    # for j in range(len(days_list_functionality)):
    # 	x[j] = days_list_functionality[j]
    # 	if j != 0:
    # 		x[j] = sum(days_list_functionality[:j])

    # x[0] = 0
    # x[j + 1] = sum(days_list_functionality[:j + 1])
    # x.append(x[-1])

    # plt.figure()
    # # plt.fill_between(x,y, step="pre", color='blue', alpha=0.4)
    # plt.step(x, y, label='f_p')
    # plt.xlabel('Days')
    # plt.ylabel('Functionlity')
    # plt.title('Network Functionality')
    # plt.legend()
    # plt.savefig(sname + '_Func_over_phases')

    # tot_area = 0
    # tstt_g = []
    # for i in range(len(days_list_functionality)):
    # 	if i == 0:
    # 		tot_area += days_list_functionality[i] * \
    # 			(after_eq_tstt - before_eq_tstt)
    # 		# for j in range(int(days_list[i])):
    # 		tstt_g.append(after_eq_tstt)
    # 	else:
    # 		tot_area += days_list_functionality[i] * \
    # 			(tstt_list_functionality[i - 1] - before_eq_tstt)
    # 		# for j in range(int(days_list[i])):
    # 		tstt_g.append(tstt_list_functionality[i - 1])

    # y = [after_eq_tstt] + tstt_g + [tstt_g[-1]]
    # x = [0] + days_list_functionality

    # for j in range(len(days_list_functionality)):
    # 	x[j] = days_list_functionality[j]
    # 	if j != 0:
    # 		x[j] = sum(days_list_functionality[:j])

    # x[0] = 0
    # x[j + 1] = sum(days_list_functionality[:j + 1])
    # x.append(x[-1])

    # print('functionality area: ', tot_area)

    # # plt.subplot(213)
    # pdb.set_trace()
    # # plt.subplot(212)
    # plt.figure()
    # plt.fill_between(x, y, before_eq_tstt, step="pre", color='blue', alpha=0.4)
    # plt.step(x, y, label='tstt')
    # plt.xlabel('Days')
    # plt.ylabel('TSTT')
    # plt.title('Network Functionality')
    # tt = 'Total Area: ' + str(tot_area)
    # xy = (0.2, 0.2)
    # plt.annotate(tt, xy, xycoords='figure fraction')
    # plt.legend()
    # plt.savefig(sname + '_NF_TSTT')
    # # plt.show()

    # df = pd.DataFrame({'b)-after': after_eq_tstt,
    # 				   'a)-before': before_eq_tstt}, index=[''])
    # ax = df.plot.bar(rot=0)
    # ax.set_ylabel('TSTT')
    # # ax.set_xlabel('')
    # ax.set_title('TSTT (Before - After)')
    # fig = ax.get_figure()
    # fig.savefig(sname + "_tstt_before_after")

    # df = pd.DataFrame({'b)-after': ffp, 'a)-before': 100}, index=[''])
    # ax = df.plot.bar(rot=0)
    # ax.set_ylabel('Functionality')
    # ax.set_xlabel('')
    # ax.set_title('Network Functionality (Before - After)')
    # fig = ax.get_figure()
    # fig.savefig(sname + "_functionality_before_after")

    # save(mydict, sname + '_saved_variables')

# pdb.set_trace()
# plt.show()
# plt.savefig('TSTT')

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
