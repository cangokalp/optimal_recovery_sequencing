from sequence_utils import *

def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def eval_sequence(net, order_list, after_eq_tstt, before_eq_tstt, if_list=None, importance=False):
    tap_solved = 0
    damaged_dict = net.damaged_dict
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
        tap_solved += 1
        tstt_after = solve_UE(net=net)

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


def get_marginal_tstts(net, path, after_eq_tstt, before_eq_tstt):
    damaged_dict = net.damaged_dict

    _, _, tstt_list = eval_sequence(deepcopy(net), path, after_eq_tstt, before_eq_tstt)

    # tstt_list.insert(0, after_eq_tstt)
    
    days_list = []
    for link in path:
        days_list.append(damaged_dict[link])
    
    return tstt_list, days_list
