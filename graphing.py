from sequence_utils import *
from sequence_helpers import *
from matplotlib import cm

def graph_current(tstt_state, days_state, before_eq_tstt, after_eq_tstt, path, plt_path, algo, together, place, color_dict, sname):

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
        plt.figure(figsize=(16, 8))

    plt.fill_between(x, y, before_eq_tstt, step="post",
                     color='#087efe', alpha=0.2, label='TOTAL AREA:' + '{0:1.5e}'.format(tot_area))
    # plt.step(x, y, label='tstt', where="post")
    plt.xlabel('DAYS')
    plt.ylabel('TSTT')

    plt.ylim((before_eq_tstt, after_eq_tstt + after_eq_tstt * 0.07))

    for i in range(len(tstt_state)):

        color = color_dict[path[i]]

        start = sum(days_state[:i])

        bbox_props = dict(boxstyle="round,pad=0.3",
                          fc="w", ec=color, alpha=0.8, lw=1)
        
        ha = 'center'
        # if i >= len(tstt_state) - 3:
        #     ha = 'right'


        t = plt.text(start + N * 0.01, y[i] + before_eq_tstt * 0.02, "link: " + path[i] + '\n' + "time: " + str(round(days_state[i], 2)), ha=ha, size=8,
                 bbox=bbox_props)

        bb = t.get_bbox_patch()

        # start + days_state[i]/2
        # plt.annotate("link: " + path[i], (start, y[i]), textcoords='offset points', xytext=(5,15), ha='left', size='smaller')
        # plt.annotate("time: " + str(round(days_state[i],2)), (start, y[i]),
        # textcoords='offset points', xytext=(5,5), ha='left', size='smaller')
        # #, arrowprops=dict(width= days_state[i]))

        # plt.annotate("", xy=(start + days_state[i], y[i]), xytext=(start, y[i]) , textcoords='offset points', ha='center', arrowprops=dict(arrowstyle='<->', connectionstyle="arc3"))
        # plt.arrow(0.85, 0.5, dx = -0.70, dy = 0, head_width=0.05, head_length=0.03, linewidth=4, color='g', length_includes_head=True)


        plt.annotate(s='', xy=(start + days_state[i], y[i]), xytext=(
            start, y[i]), arrowprops=dict(arrowstyle='<->', color=color))
        

        # plt.annotate(s='', xy=(start + days_state[i], y[i]), xytext=(start + days_state[
        #              i], y[i + 1]), arrowprops=dict(arrowstyle='<->', color='indigo'))

        # plt.annotate("" + str(round(y[i] - y[i+1], 2)), xy=(start +
        # days_state[i], (y[i] - y[i+1])/2), xytext=(10,10), textcoords='offset
        # points', ha='left') #, arrowprops=dict(width= days_state[i]))

        # animation example:
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
    t_critical = stats.t.ppf(q=0.9, df=sample_size)

    mean = np.mean(metric_values)
    stdev = np.std(metric_values)    # Get the sample standard deviation
    sigma = stdev / np.sqrt(sample_size)  # Standard deviation estimate
    error = t_critical * sigma

    return mean, error


def prep_dictionaries(method_dict):
    method_dict['_obj'] = []
    method_dict['_num_tap'] = []
    method_dict['_elapsed'] = []


def result_table(reps, file_path, broken, sname):

    filenames = ['algo_solution', 'min_seq',
                 'greedy_solution', 'importance_factor_bound']

    heuristic = {}
    greedy = {}
    brute_force = {}
    importance_factor = {}




    dict_list = [heuristic, brute_force, greedy, importance_factor]
    key_list = ['_obj', '_num_tap', '_elapsed']

    for method_dict in dict_list:
        prep_dictionaries(method_dict)

    if int(broken) >= 8:
        filenames[1] = 'algo_solution'

    for rep in range(reps+1):
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

    dict_list.remove(brute_force)

    optimal = deepcopy(np.array(brute_force['_obj']))

    data = np.zeros((len(dict_list), 6))

    r = 0

    if int(broken) == 10:
        pdb.set_trace()

    for method_dict in dict_list:
        objs = np.array(method_dict['_obj'])
        objs = ((objs - optimal) / optimal) * 100
        objs = np.maximum(0, objs)

        mean, error = mean_std_lists(objs, sample_size)
        obj_means.append(mean)
        obj_err.append(error)
        data[r, 0] = mean
        data[r, 1] = error

        taps = method_dict['_num_tap']
        mean, error = mean_std_lists(taps, sample_size)
        tap_means.append(mean)
        tap_err.append(error)
        data[r, 2] = mean
        data[r, 3] = error

        elapsed_values = method_dict['_elapsed']
        mean, error = mean_std_lists(elapsed_values, sample_size)
        elapsed_means.append(mean)
        elapsed_err.append(error)
        data[r, 4] = mean
        data[r, 5] = error

        r += 1

    obj_means_scaled = obj_means / max(obj_means)
    obj_err_scaled = obj_err / max(obj_means)

    tap_means_scaled = tap_means / max(tap_means)
    tap_err_scaled = tap_err / max(tap_means)

    elapsed_means_scaled = elapsed_means / max(elapsed_means)
    elapsed_err_scaled = elapsed_err / max(elapsed_means)

    plt.figure(figsize=(10, 6))

    barWidth = 0.2
    r_obj = np.arange(len(obj_means))
    r_tap = [x + barWidth for x in r_obj]
    r_elapsed = [x + 2 * barWidth for x in r_obj]

    plt.rcParams["font.size"] = 8
    # hatch='///', hatch='\\\\\\', hatch='xxx'
    if len(objs) <= 2:
        plt.bar(r_obj, obj_means_scaled, width=barWidth, edgecolor='#087efe', color='#087efe',
                ecolor='#c6ccce', alpha=0.8, capsize=5, label='Avg Rel Gap')
        plt.bar(r_tap, tap_means_scaled, width=barWidth, edgecolor='#b7fe00', color='#b7fe00',
                ecolor='#c6ccce', alpha=0.8, capsize=5, label='Avg Tap Solved')
        plt.bar(r_elapsed, elapsed_means_scaled, width=barWidth, edgecolor='#ff9700', color='#ff9700',
                ecolor='#c6ccce', alpha=0.8,  capsize=5, label='Avg Time Elapsed (s)')

    else:
        plt.bar(r_obj, obj_means_scaled, width=barWidth, edgecolor='#087efe', color='#087efe',
                ecolor='#c6ccce', alpha=0.8, yerr=obj_err_scaled, capsize=5, label='Avg Rel Gap')
        plt.bar(r_tap, tap_means_scaled, width=barWidth, edgecolor='#b7fe00', color='#b7fe00',
                ecolor='#c6ccce', alpha=0.8, yerr=tap_err_scaled, capsize=5, label='Avg Tap Solved')
        plt.bar(r_elapsed, elapsed_means_scaled, width=barWidth, edgecolor='#ff9700', color='#ff9700',
                ecolor='#c6ccce', alpha=0.8, yerr=elapsed_err_scaled, capsize=5, label='Avg Time Elapsed (s)')

    # tap_means_scaled[i]/2
    for i in range(len(dict_list)):
        plt.annotate('{0:1.1f}'.format(obj_means[i]) + '%', (i, 0), textcoords='offset points', xytext=(
            0, 20), ha='center', va='bottom', rotation=70, size=8)
        plt.annotate('{0:1.1f}'.format(tap_means[i]), (i + barWidth, 0), textcoords='offset points', xytext=(
            0, 20), ha='center', va='bottom', rotation=70,  size='smaller')
        plt.annotate('{0:1.1f}'.format(elapsed_means[i]), (i + 2 * barWidth, 0), textcoords='offset points', xytext=(
            0, 20), ha='center', va='bottom', rotation=70,  size='smaller')

    plt.ylabel('Normalized Metric Value')
    plt.xticks([(r + barWidth) for r in range(len(obj_means))],
               ['ProposedMethod', 'GreedyMethod', 'ImportanceFactor'])
    plt.title('Performance Comparison - ' + sname, fontsize=7)
    if broken != 10:
        txt = "# Broken Links: " + \
            str(broken) + ".\n Averaged over: " + \
            str(reps+1) + ' different instances.'
    else:
        txt = "# Broken Links: " + str(broken) + ".\n Averaged over: " + str(
            reps+1) + ' different instances.\n Heuristic solution was taken as best - since not possible to solve to optimality'
    plt.figtext(0.5, 0.01, txt, wrap=True,
                ha='center', va="bottom", fontsize=7)
    plt.legend(fontsize=8)
    save_fig(file_path, 'performance_graph_' +
             'w_bridge_' + str(broken), tight_layout=False)

    columns = ('Avg Rel Gap', 'Delta', 'Avg Tap Solved',
               'Delta', 'Avg Elapsed(s)', 'Delta')
    rows = ['PM', 'GM', 'IF']

    plt.close()

    fig, ax = plt.subplots()

    cell_text = []
    for row in range(len(data)):
        cell_text.append(['%1.2f' % x + '%' if (i == 0 or i == 1)
                          else '%1.2f' % x for i, x in enumerate(data[row])])

    ax.axis('off')
    ax.axis('tight')

    # # Add a table at the bottom of the axes
    the_table = ax.table(cellText=cell_text,
                         rowLabels=rows,
                         # rowColours=colors,
                         colLabels=columns,
                         loc='center')

    if broken != 10:
        txt = "# Broken Links: " + \
            str(broken) + ". \n Averaged over: " + \
            str(reps+1) + ' different instances.'
    else:
        txt = "# Broken Links: " + str(broken) + ". \n Averaged over: " + str(
            reps+1) + ' different instances.\n Heuristic solution was taken as best - since not possible to solve to optimality'

    plt.figtext(0.5, 0.01, txt, wrap=True,
                horizontalalignment='center', fontsize=7)

    save_fig(file_path, 'performance_table_' + 'w_bridge_' + str(broken))


def result_sequence_graphs(rep, save_dir, sname):

    path_pre = os.path.join(save_dir, str(rep)) + '/'

    net_after = load(path_pre + 'net_after')
    net_after.damaged_dict = load(path_pre + 'damaged_dict')
    
    if len(net_after.damaged_dict) >= 8:
        opt = False
    else:
        opt = True

    if opt:
        opt_soln = load(path_pre + 'min_seq_path')

    algo_path = load(path_pre + 'algo_solution_path')
    greedy_soln = load(path_pre + 'greedy_solution_path')
    importance_soln = load(path_pre + 'importance_factor_bound_path')
    after_eq_tstt = load(path_pre + 'net_after_tstt')
    before_eq_tstt = load(path_pre + 'net_before_tstt')

    # GRAPH THE RESULTS
    if opt:
        paths = [algo_path, opt_soln, greedy_soln, importance_soln]
        names = ['ProposedMethod', 'Optimal',
                 'GreedyMethod', 'ImportanceFactor']
        places = [221, 222, 223, 224]
    else:
        paths = [algo_path, greedy_soln, importance_soln]
        names = ['ProposedMethod', 'GreedyMethod', 'ImportanceFactor']
        places = [221, 222, 223]


    colors = plt.cm.ocean(np.linspace(0, 0.7, len(algo_path)))
    color_dict = {}
    color_i = 0
    for alink in algo_path:
        color_dict[alink] = colors[color_i]
        color_i += 1

    # colors = ['y', 'r', 'b', 'g']
    for path, name, place in zip(paths, names, places):
        tstt_list, days_list = get_marginal_tstts(
            net_after, path, after_eq_tstt, before_eq_tstt)
        graph_current(tstt_list, days_list, before_eq_tstt,
                      after_eq_tstt, path, path_pre, name, False, place, color_dict, sname)

    plt.close()
    # plt.figure(figsize=(16, 8))

    for path, name, place in zip(paths, names, places):
        tstt_list, days_list = get_marginal_tstts(
            net_after, path, after_eq_tstt, before_eq_tstt)
        graph_current(tstt_list, days_list, before_eq_tstt,
                      after_eq_tstt, path, path_pre, name, True, place, color_dict, sname)

    save_fig(path_pre, 'Comparison')


def get_folders(path):
    folders = os.listdir(path)

    try:
        folders.remove('.DS_Store')
    except:
        pass

    try:
        folders.remove('figures')
    except:
        pass

    return folders


def get_tables(snames):

    for sname in snames:

        SCENARIO_DIR = os.path.join(NETWORK_DIR, sname)

        try:
            broken_bridges = get_folders(SCENARIO_DIR)
        except:
            return

        for broken in broken_bridges:

            ULT_SCENARIO_DIR = os.path.join(SCENARIO_DIR, broken)
            repetitions = get_folders(ULT_SCENARIO_DIR)

            if len(repetitions) == 0:
                return

            reps = [int(i) for i in repetitions]
            max_reps = max(reps)
            result_table(max_reps, ULT_SCENARIO_DIR, broken, sname)


def get_sequence_graphs(snames):

    for sname in snames:

        SCENARIO_DIR = os.path.join(NETWORK_DIR, sname)
        
        try:
            broken_bridges = get_folders(SCENARIO_DIR)
        except:
            return

        if len(broken_bridges) == 0:
            return

        for broken in broken_bridges:

            ULT_SCENARIO_DIR = os.path.join(SCENARIO_DIR, broken)
            repetitions = get_folders(ULT_SCENARIO_DIR)
            
            if len(repetitions) == 0:
                return

            reps = [int(i) for i in repetitions]

            for rep in reps:
                result_sequence_graphs(rep, ULT_SCENARIO_DIR, sname)
