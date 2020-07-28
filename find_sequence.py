from sequence_utils import *
import random
import operator as op
from functools import reduce
import itertools
from prettytable import PrettyTable
import operator
import multiprocessing as mp
from graphing import *
from scipy.special import comb
import math
import argparse
import numpy as np
import cProfile
import pstats
import networkx as nx
import os.path
from ctypes import *
import time
from scipy import stats
from correspondence import *
import progressbar
from network import *

extension = '.pickle'

parser = argparse.ArgumentParser(
    description='find an order for repairing bridges')
parser.add_argument('-n', '--net_name', type=str, help='network name')
parser.add_argument('-b', '--num_broken', type=int,
                    help='number of broken bridges')
parser.add_argument('-a', '--approx', type=bool,
                    help='approximation enabled for speed', default=False)
parser.add_argument('-s', '--beamsearch', type=bool,
                    help='beam search enabled for speed', default=False)
parser.add_argument('-k', '--beamk', type=int,
                    help='beam search parameter', default=20)
parser.add_argument('-r', '--reps', type=int,
                    help='number of scenarios with the given parameters', default=1)
parser.add_argument('-g', '--graphing', type=bool, 
                    help='graphing mode', default=False)
parser.add_argument('-o', '--scenario', type=str, 
                    help='scenario file', default='scenario.csv')
parser.add_argument('-t', '--strength', type=str, 
                    help='strength of the earthquake')
parser.add_argument('-y', '--onlybeforeafter', type=bool,
                    help='to get before and after tstt', default=False)
parser.add_argument('-f', '--full', type=bool,
                    help='to use full algo not beam search', default=False)
parser.add_argument('-j', '--random', type=bool,
                    help='generate scenarios randomly', default=False)

args = parser.parse_args()

SEED = 9
FOLDER = "TransportationNetworks"
MAX_DAYS = 180
MIN_DAYS = 21


class Node():
    """A node class for bi-directional search for pathfinding"""

    def __init__(self, visited=None, link_id=None, parent=None, tstt_after=None, tstt_before=None, level=None, forward=True, relax=False, not_fixed=None):

        self.relax = relax
        self.forward = forward
        self.parent = parent
        self.level = level
        self.link_id = link_id
        self.path = []
        self.tstt_before = tstt_before
        self.tstt_after = tstt_after
        self.g = 0
        self.h = 0
        self.f = 0
        if relax:
            self.err_rate = 0.01
        else:
            self.err_rate = 0

        self.assign_char()

    def assign_char(self):

        if self.parent is not None:
            self.benefit = self.tstt_after - self.tstt_before
            self.days = damaged_dict[self.link_id]
            prev_path = deepcopy(self.parent.path)
            prev_path.append(self.link_id)
            self.path = prev_path
            self.visited = set(self.path)
            self.days_past = self.parent.days_past + self.days
            self.before_eq_tstt = self.parent.before_eq_tstt
            self.after_eq_tstt = self.parent.after_eq_tstt

            self.realized = self.parent.realized + \
                (self.tstt_before - self.before_eq_tstt) * self.days

            self.realized_u = self.parent.realized_u + \
                (self.tstt_before - self.before_eq_tstt) * \
                self.days * (1 + self.err_rate)

            self.realized_l = self.parent.realized_l + \
                (self.tstt_before - self.before_eq_tstt) * \
                self.days * (1 - self.err_rate)

            self.not_visited = set(damaged_dict.keys()
                                   ).difference(self.visited)

            self.forward = self.parent.forward
            if self.forward:
                self.not_fixed = self.not_visited
                self.fixed = self.visited

            else:
                self.not_fixed = self.visited
                self.fixed = self.not_visited

        else:
            if self.link_id is not None:
                self.path = [self.link_id]
                self.realized = (self.tstt_before -
                                 self.before_eq_tstt) * self.days
                self.realized_u = (self.tstt_before -
                                   self.before_eq_tstt) * self.days * (1 + self.err_rate)
                self.realized_l = (self.tstt_before -
                                   self.before_eq_tstt) * self.days * (1 - self.err_rate)
                self.days_past = self.days

            else:
                self.realized = 0
                self.realized_u = 0
                self.realized_l = 0
                self.days = 0
                self.days_past = self.days

    def __eq__(self, other):
        return self.fixed == other.fixed


def get_successors_f(node):
    """given a state, returns list of bridges that has not been fixed yet"""
    not_visited = node.not_visited
    successors = []

    if node.level != 0:
        tail = node.path[-1]
        for a_link in not_visited:
            if wb[a_link] * damaged_dict[tail] - bb[tail] * damaged_dict[a_link] > 0:
                continue
            successors.append(a_link)
    else:
        successors = not_visited

    return successors


def get_successors_b(node):
    """given a state, returns list of bridges that has not been removed yet"""
    not_visited = node.not_visited
    successors = []

    # if node.path == ['(13,12)']:
    #     pdb.set_trace()
    if node.level != len(damaged_dict.keys()):
        tail = node.path[-1]
        for a_link in not_visited:

            # if wb[tail] * node.damaged_dict[a_link] - bb[a_link] *
            # node.damaged_dict[tail] > 0:
            if -wb[tail] * damaged_dict[a_link] + bb[a_link] * damaged_dict[tail] < 0:
                continue
            successors.append(a_link)
    else:
        successors = not_visited

    return successors


def expand_sequence_f(node, a_link, level):
    """given a link and a node, it expands the sequence"""
    solved = 0
    tstt_before = node.tstt_after
    
    net = create_network(NETFILE, TRIPFILE)

    net.not_fixed = set(node.not_fixed).difference(set([a_link]))

    if frozenset(net.not_fixed) in memory.keys():
        tstt_after = memory[frozenset(net.not_fixed)]

    else:
        tstt_after = solve_UE(net=net, relax=node.relax)
        memory[frozenset(net.not_fixed)] = (tstt_after)
        solved = 1


    node = Node(link_id=a_link, parent=node, tstt_after=tstt_after,
                tstt_before=tstt_before, level=level, relax=node.relax)


    del net
    return node, solved


def expand_sequence_b(node, a_link, level):
    """given a link and a node, it expands the sequence"""
    solved = 0
    tstt_after = node.tstt_before

    net = create_network(NETFILE, TRIPFILE)

    net.not_fixed = node.not_fixed.union(set([a_link]))

    if frozenset(net.not_fixed) in memory.keys():
        tstt_before = memory[frozenset(net.not_fixed)]
    else:
        tstt_before = solve_UE(net=net, relax=node.relax)
        memory[frozenset(net.not_fixed)] = tstt_before
        solved = 1

    node = Node(link_id=a_link, parent=node, tstt_after=tstt_after,
                tstt_before=tstt_before, level=level, relax=node.relax)

    del net
    return node, solved


def orderlists(benefits, days, slack=0, rem_keys=None, reverse=True):
    #
    # if sum(benefits) > slack:
    #     benefits = np.array(benefits)
    #     benefits = [min(slack, x) for x in benefits]

    bang4buck = np.array(benefits) / np.array(days)

    days = [x for _, x in sorted(
        zip(bang4buck, days), reverse=reverse)]
    benefits = [x for _, x in sorted(
        zip(bang4buck, benefits), reverse=reverse)]

    if rem_keys is not None:
        rem_keys = [x for _, x in sorted(
            zip(bang4buck, rem_keys), reverse=reverse)]
        return _,_, rem_keys

    return benefits, days

def check(fwd_node, bwd_node, relax):
    fwdnet = create_network(NETFILE, TRIPFILE)

    fwdnet.not_fixed = fwd_node.not_fixed

    fwd_tstt = solve_UE(net=fwdnet, relax=relax)

    bwdnet = create_network(NETFILE, TRIPFILE)

    bwdnet.not_fixed = bwd_node.not_fixed

    bwd_tstt = solve_UE(net=bwdnet, relax=relax)

    return fwd_tstt, bwd_tstt, fwdnet, bwdnet

def get_minlb(node, fwd_node, bwd_node, orderedb_benefits, orderedw_benefits, ordered_days, forward_tstt, backward_tstt, backwards=False, best_feasible_soln=None, uncommon_number=0, common_number=0):

    slack = forward_tstt - backward_tstt
    if slack < 0:
        backward_tstt_orig = backward_tstt
        backward_tstt = forward_tstt
        forward_tstt = backward_tstt_orig
        slack = forward_tstt - backward_tstt

    if len(ordered_days) == 0:
        node.ub = fwd_node.realized_u + bwd_node.realized_u
        node.lb = fwd_node.realized_l + bwd_node.realized_l
        cur_obj = fwd_node.realized + bwd_node.realized
        new_feasible_path = fwd_node.path + bwd_node.path[::-1]

        if cur_obj < best_feasible_soln.g:
            best_feasible_soln.g = cur_obj
            best_feasible_soln.path = new_feasible_path

        return uncommon_number, common_number

    elif len(ordered_days) == 1:
        node.ub = fwd_node.realized_u + bwd_node.realized_u + \
            (fwd_node.tstt_after - node.before_eq_tstt) * ordered_days[0]
        node.lb = fwd_node.realized_l + bwd_node.realized_l + \
            (fwd_node.tstt_after - node.before_eq_tstt) * ordered_days[0]
        cur_obj = fwd_node.realized + bwd_node.realized + \
            (fwd_node.tstt_after - node.before_eq_tstt) * ordered_days[0]

        lo_link = list(set(damaged_dict.keys()).difference(
            set(fwd_node.path).union(set(bwd_node.path))))
        new_feasible_path = fwd_node.path + \
            [str(lo_link[0])] + bwd_node.path[::-1]

        if cur_obj < best_feasible_soln.g:
            best_feasible_soln.g = cur_obj
            best_feasible_soln.path = new_feasible_path

        return uncommon_number, common_number

    b, days_b = orderlists(orderedb_benefits, ordered_days, slack)
    w, days_w = orderlists(orderedw_benefits, ordered_days, slack)
    sumw = sum(w)
    sumb = sum(b)

    orig_lb = node.lb
    orig_ub = node.ub

    ###### FIND LB FROM FORWARDS #####
    if sumw < slack:

        for i in range(len(days_w)):

            if i == 0:
                fwd_w = deepcopy(w)
                fwd_days_w = deepcopy(days_w)
                b_tstt = sumw

            else:
                b_tstt = b_tstt - fwd_w[0]
                fwd_w = fwd_w[1:]
                fwd_days_w = fwd_days_w[1:]


            node.lb += b_tstt * fwd_days_w[0] + (bwd_node.tstt_before - node.before_eq_tstt) * fwd_days_w[0]
        common_number+=1
    else:
        node.lb += (fwd_node.tstt_after-node.before_eq_tstt)*min(days_w) + (bwd_node.tstt_before - node.before_eq_tstt)*(sum(days_w) - min(days_w))
        uncommon_number+=1

    lb2 = (fwd_node.tstt_after-node.before_eq_tstt)*min(days_w) + (bwd_node.tstt_before - node.before_eq_tstt)*(sum(days_w) - min(days_w))
    node.lb = max(node.lb, lb2 + orig_lb)

    ###### FIND UB FROM FORWARDS #####
    if sum(b) > slack:
        for i in range(len(days_b)):
            if i == 0:
                fwd_b = deepcopy(b)
                fwd_days_b = deepcopy(days_b)
                fwd_b, fwd_days_b = orderlists(fwd_b, fwd_days_b, reverse=False)
                b_tstt = sumb

            else:
                b_tstt = b_tstt - fwd_b[0]
                fwd_b = fwd_b[1:]
                fwd_days_b = fwd_days_b[1:]

            node.ub += b_tstt * fwd_days_b[0] + (bwd_node.tstt_before - node.before_eq_tstt) * fwd_days_b[0]

        common_number += 1
    else:
        node.ub += sum(days_b) * (forward_tstt - node.before_eq_tstt)
        uncommon_number += 1

    node.ub = min(node.ub, sum(days_b) * (forward_tstt - node.before_eq_tstt) + orig_ub)

    #
    # if backward_ub < node.ub:
    #     node.ub = backward_ub
    #
    # if backward_lb > node.lb:
    #     node.lb = backward_lb

    if node.lb > node.ub:

        if node.relax == False:
            pdb.set_trace()

        if abs(node.lb - node.ub) < 5:
            tempub = node.ub
            node.ub = node.lb
            node.lb = tempub

        else:
            print('problem')
            pdb.set_trace()

    return uncommon_number, common_number

def set_bounds_bif(node, open_list_b, end_node, front_to_end=True, debug=False, best_feasible_soln=None, uncommon_number=0, common_number=0, get_feas=False):

    sorted_d = sorted(damaged_dict.items(), key=lambda x: x[1])

    remaining = []
    eligible_backward_connects = []

    if front_to_end:
        eligible_backward_connects = [end_node]
    else:
        for other_end in open_list_b:
            if len(set(node.visited).intersection(other_end.visited)) == 0:
                eligible_backward_connects.append(other_end)

    minlb = np.inf
    maxub = np.inf

    if debug:
        pdb.set_trace()
    if len(eligible_backward_connects) == 0:
        eligible_backward_connects = [end_node]

    minother_end = None
    for other_end in eligible_backward_connects:
        ordered_days = []
        orderedw_benefits = []
        orderedb_benefits = []
        # debug right here
        node.ub = node.realized_u + other_end.realized_u
        node.lb = node.realized + other_end.realized

        union = node.visited.union(other_end.visited)
        remaining = set(damaged_dict.keys()).difference(union)

        for key, value in sorted_d:
            # print("%s: %s" % (key, value))
            if key in remaining:
                ordered_days.append(value)
                orderedw_benefits.append(wb[key])
                orderedb_benefits.append(bb[key])

        forward_tstt = node.tstt_after
        backward_tstt = other_end.tstt_before


        uncommon_number, common_number = get_minlb(node, node, other_end, orderedb_benefits,
                  orderedw_benefits, ordered_days, forward_tstt, backward_tstt, best_feasible_soln=best_feasible_soln, uncommon_number=uncommon_number, common_number=common_number)

        if node.lb < minlb:
            minlb = node.lb
            minother_end = other_end

        if node.ub < maxub:
            maxub = node.ub

    node.lb = minlb
    node.ub = maxub

    if get_feas:
        accrued = node.realized + minother_end.realized
        union = node.visited.union(minother_end.visited)
        remaining = set(damaged_dict.keys()).difference(union)
        for key, value in sorted_d:
            # print("%s: %s" % (key, value))
            if key in remaining:
                ordered_days.append(value)
                orderedb_benefits.append(bb[key])

        _, _, rem_ord = orderlists(orderedb_benefits, ordered_days, rem_keys=remaining)

        net_a = create_network(NETFILE, TRIPFILE)
        net_a.not_fixed = remaining
        tstt_after = node.tstt_after

        for added in remaining:
            net_a.not_fixed = set(remaining).union(minother_end.visited).difference(set(added))
            accrued += (tstt_after - node.before_eq_tstt) * damaged_dict[added]
            tstt_after = solve_UE(net=net_a)

        new_feasible_path = node.path + rem_ord + minother_end.path
        if accrued < best_feasible_soln.g:
            best_feasible_soln.g = accrued
            best_feasible_soln.path = new_feasible_path

    if node.lb > node.ub:
        pdb.set_trace()

    return uncommon_number, common_number

def set_bounds_bib(node, open_list_f, start_node, front_to_end=True, best_feasible_soln=None, uncommon_number=0, common_number=0, get_feas=False):

    sorted_d = sorted(damaged_dict.items(), key=lambda x: x[1])

    remaining = []
    eligible_backward_connects = []

    if front_to_end:
        eligible_backward_connects = [start_node]
    else:
        for other_end in open_list_f:
            if len(set(node.visited).intersection(other_end.visited)) == 0:
                eligible_backward_connects.append(other_end)

    minlb = np.inf
    maxub = np.inf

    if len(eligible_backward_connects) == 0:
        eligible_backward_connects = [start_node]

    minother_end = None
    for other_end in eligible_backward_connects:
        ordered_days = []
        orderedw_benefits = []
        orderedb_benefits = []

        node.ub = node.realized_u + other_end.realized_u
        node.lb = node.realized + other_end.realized

        union = node.visited.union(other_end.visited)
        remaining = set(damaged_dict.keys()).difference(union)

        for key, value in sorted_d:
            # print("%s: %s" % (key, value))
            if key in remaining:
                ordered_days.append(value)
                orderedw_benefits.append(wb[key])
                orderedb_benefits.append(bb[key])
        forward_tstt = other_end.tstt_after
        backward_tstt = node.tstt_before


        uncommon_number, common_number = get_minlb(node, other_end, node, orderedb_benefits, orderedw_benefits,
                  ordered_days, forward_tstt, backward_tstt, backwards=True, best_feasible_soln=best_feasible_soln, uncommon_number=uncommon_number, common_number=common_number)

        if node.lb < minlb:
            minlb = node.lb
            minother_end = other_end
        if node.ub < maxub:
            maxub = node.ub

    node.lb = minlb
    node.ub = maxub

    if get_feas:
        accrued = node.realized + minother_end.realized
        union = node.visited.union(minother_end.visited)
        remaining = set(damaged_dict.keys()).difference(union)
        for key, value in sorted_d:
            # print("%s: %s" % (key, value))
            if key in remaining:
                ordered_days.append(value)
                orderedb_benefits.append(bb[key])

        _, _, rem_ord = orderlists(orderedb_benefits, ordered_days, rem_keys=remaining)

        net_a = create_network(NETFILE, TRIPFILE)
        net_a.not_fixed = remaining
        tstt_before = minother_end.tstt_after
        for added in remaining:
            net_a.not_fixed = set(remaining).union(node.path).difference(set(added))
            accrued += (tstt_before - node.before_eq_tstt) * damaged_dict[added]
            tstt_before = solve_UE(net=net_a)

        new_feasible_path = minother_end.path + rem_ord + node.path
        if accrued < best_feasible_soln.g:
            best_feasible_soln.g = accrued
            best_feasible_soln.path = new_feasible_path
            print(accrued)
    if node.lb > node.ub:
        pdb.set_trace()
    return uncommon_number, common_number


def expand_forward(start_node, end_node, open_list_b, open_list_f, closed_list_b, closed_list_f, best_ub, best_feasible_soln, num_tap_solved, max_level_b, front_to_end=False, uncommon_number=0, tot_child=0, common_number=0):
    debug = False
    # print('in forward search')

    # Get the most promising node from the open list
    current_node = open_list_f[0]
    current_index = 0
    for index, item in enumerate(open_list_f):
        if item.f < current_node.f:
            current_node = item
            current_index = index

    # Pop current off open list, add to closed list
    open_list_f.pop(current_index)
    closed_list_f.append(current_node)

    if len(damaged_dict) - max_level_b + len(current_node.visited) + 1 >= len(damaged_dict):

        cur_visited = current_node.visited
        for other_end in open_list_b:
        # for other_end in open_list_b:

            if len(set(other_end.visited).intersection(set(cur_visited))) == 0 and len(damaged_dict) - len(set(other_end.visited).union(set(cur_visited))) == 1:
                lo_link = set(damaged_dict.keys()).difference(
                    set(other_end.visited).union(set(cur_visited)))
                lo_link = lo_link.pop()
                cur_soln = current_node.g + other_end.g + \
                    (current_node.tstt_after - current_node.before_eq_tstt) * \
                    damaged_dict[lo_link]
                cur_path = current_node.path + \
                    [str(lo_link)] + other_end.path[::-1]

                if cur_soln <= best_feasible_soln.g:
                    best_feasible_soln.g = cur_soln
                    best_feasible_soln.path = cur_path
                    # print(current_node.visited)
                    # print('-------BEST_SOLN-----: ', best_feasible_soln.g)
                    print(best_feasible_soln.path)


            if set(other_end.not_visited) == set(cur_visited):
                cur_soln = current_node.g + other_end.g
                cur_path = current_node.path + other_end.path[::-1]
                if cur_soln <= best_feasible_soln.g:
                    # print(current_node.visited)

                    best_feasible_soln.g = cur_soln
                    best_feasible_soln.path = cur_path
                    # print('-------BEST_SOLN-----: ', best_feasible_soln.g)
                    print(best_feasible_soln.path)

    if best_feasible_soln.g < best_ub:
        best_ub = best_feasible_soln.g

    # Found the goal
    if current_node == end_node:
        # print('at the end_node')
        return open_list_f, closed_list_f, best_ub, best_feasible_soln, num_tap_solved, current_node.level, tot_child, uncommon_number, common_number

    if current_node.f > best_feasible_soln.g or current_node.f > best_ub:
        # print('current node is pruned')
        return open_list_f, closed_list_f, best_ub, best_feasible_soln, num_tap_solved, current_node.level, tot_child, uncommon_number, common_number

    # Generate children
    eligible_expansions = get_successors_f(current_node)
    children = []

    current_level = current_node.level
    for a_link in eligible_expansions:

        # Create new node
        current_level = current_node.level + 1
        new_node, solved = expand_sequence_f(
            current_node, a_link, level=current_level)
        num_tap_solved += solved
        # Append
        children.append(new_node)

    del current_node

    # Loop through children
    for child in children:

        # set upper and lower bounds
        tot_child += 1
        uncommon_number, common_number = set_bounds_bif(child, open_list_b,
                       front_to_end=front_to_end, end_node=end_node, debug=debug, best_feasible_soln=best_feasible_soln, uncommon_number=uncommon_number, common_number=common_number)

        if best_feasible_soln.g < best_ub:
            best_ub = best_feasible_soln.g

        if child.ub <= best_ub:
            best_ub = child.ub
            if len(child.path) == len(damaged_dict):
                if child.g < best_feasible_soln.g:
                    best_feasible_soln.g = child.g
                    best_feasible_soln.path = child.path

        if child.lb == child.ub:
            continue
            # pdb.set_trace()

            #     and len(child.path) != len(child.damaged_dict):
            #
            # lo_link = set(damaged_dict.keys()).difference(set(child.visited))
            # lo_link = lo_link.pop()
            # best_path = child.path + [str(lo_link)]

        if child.lb > child.ub:
            pdb.set_trace()

        # Create the f, g, and h values
        child.g = child.realized
        # child.h = child.lb - child.g
        # child.f = child.g + child.h
        child.f = child.lb

        if child.f > best_feasible_soln.g or child.f > best_ub:
            continue

        # Child is already in the open list
        add = True
        removal = []

        for open_node in open_list_f:
            if child == open_node:
                if child.g > open_node.g:
                    add = False
                else:
                    removal.append(open_node)

        for open_node in removal:
            open_list_f.remove(open_node)
            closed_list_f.append(open_node)

        # Child is on the closed list
        if add:
            for closed_node in closed_list_f:
                if child.fixed == closed_node.fixed:
                    if child.g >= closed_node.g:
                        add = False
                        break



        if add:
            # Add the child to the open list
            open_list_f.append(child)
        else:
            del child

    return open_list_f, closed_list_f, best_ub, best_feasible_soln, num_tap_solved, current_level, tot_child, uncommon_number, common_number


def expand_backward(start_node, end_node, open_list_b, open_list_f, closed_list_b, closed_list_f, best_ub, best_feasible_soln, num_tap_solved, max_level_f, front_to_end=False, uncommon_number=0, tot_child=0, common_number=0):

    # Loop until you find the end
    current_node = open_list_b[0]
    current_index = 0

    for index, item in enumerate(open_list_b):
        if item.f < current_node.f:
            current_node = item
            current_index = index

    # Pop current off open list, add to closed list
    open_list_b.pop(current_index)
    closed_list_b.append(current_node)


    if max_level_f + len(current_node.visited) + 1 >= len(damaged_dict):
        cur_visited = current_node.visited
        for other_end in open_list_f:
        # for other_end in open_list_f:

            if len(set(other_end.visited).intersection(set(cur_visited))) == 0 and len(damaged_dict) - len(set(other_end.visited).union(set(cur_visited))) == 1:
                lo_link = set(damaged_dict.keys()).difference(
                    set(other_end.visited).union(set(cur_visited)))
                lo_link = lo_link.pop()
                cur_soln = current_node.g + other_end.g + \
                    (other_end.tstt_after - other_end.before_eq_tstt) * \
                    damaged_dict[lo_link]

                if cur_soln <= best_feasible_soln.g:
                    best_feasible_soln.g = cur_soln
                    best_feasible_soln.path = other_end.path + \
                        [str(lo_link)] + current_node.path[::-1]
                    # print(current_node.visited)
                    #
                    # print('-------BEST_SOLN-----: ', best_feasible_soln.g)
                    print(best_feasible_soln.path)

            elif set(other_end.not_visited) == set(cur_visited):
                cur_soln = current_node.g + other_end.g
                # print('current_soln: {}, best_soln: {}'.format(cur_soln, best_soln))
                if cur_soln <= best_feasible_soln.g:
                    best_feasible_soln.g = cur_soln
                    best_feasible_soln.path = other_end.path + \
                        current_node.path[::-1]
                    # print(current_node.visited)
                    #
                    # print('-------BEST_SOLN-----: ', best_feasible_soln.g)
                    print(best_feasible_soln.path)

    if best_feasible_soln.g < best_ub:
        best_ub = best_feasible_soln.g

    # Found the goal
    if current_node == start_node:
        # print('at the start node')
        return open_list_b, closed_list_b, best_ub, best_feasible_soln, num_tap_solved, current_node.level, tot_child, uncommon_number, common_number

    if current_node.f > best_ub:
        # print('current node is pruned')
        return open_list_b, closed_list_b, best_ub, best_feasible_soln, num_tap_solved, current_node.level, tot_child, uncommon_number, common_number

    # Generate children
    eligible_expansions = get_successors_b(current_node)

    children = []

    current_level = current_node.level

    for a_link in eligible_expansions:

        # Create new node
        current_level = current_node.level - 1
        new_node, solved = expand_sequence_b(
            current_node, a_link, level=current_level)
        num_tap_solved += solved
        children.append(new_node)

    del current_node
    # Loop through children

    for child in children:

        # set upper and lower bounds
        tot_child += 1
        uncommon_number, common_number = set_bounds_bib(child, open_list_f,
                       front_to_end=front_to_end, start_node=start_node, best_feasible_soln=best_feasible_soln, uncommon_number=uncommon_number, common_number=common_number)

        if best_feasible_soln.g < best_ub:
            best_ub = best_feasible_soln.g

        if child.ub <= best_ub:
            best_ub = child.ub
            if len(child.path) == len(damaged_dict):
                if child.g < best_feasible_soln.g:
                    best_feasible_soln.g = child.g
                    best_feasible_soln.path = child.path[::-1]

        if child.lb == child.ub:
            # pdb.set_trace()
            continue
            # pdb.set_trace()
            # and len(child.path) != len(child.damaged_dict):
            # lo_link = set(damaged_dict.keys()).difference(set(child.visited))
            # lo_link = lo_link.pop()
            # best_path = [str(lo_link)] + child.path[::-1]
            # if len(best_path)!=len(child.damaged_dict):
            #     pdb.set_trace()
            #

        if child.lb > child.ub:
            pdb.set_trace()

        child.g = child.realized
        child.f = child.lb

        if child.f > best_feasible_soln.g or child.f > best_ub:
            continue

        # Child is already in the open list
        add = True
        removal = []
        for open_node in open_list_b:
            if child == open_node:
                if child.g > open_node.g:
                    add = False
                else:
                    removal.append(open_node)

        for open_node in removal:
            open_list_b.remove(open_node)
            closed_list_b.append(open_node)

        # Child is on the closed list
        if add:
            for closed_node in closed_list_b:
                if child.fixed == closed_node.fixed:
                    if child.g > closed_node.g:
                        add = False
                        break
            # for idx, closed_node_fixed in enumerate(closed_list_b):
            #     if child.fixed == closed_node_fixed:
            #         if child.g > closed_list_b_g[idx]:
            #             add = False
            #             break

        # Add the child to the open list
        if add:
            open_list_b.append(child)
        else:
            del child

    return open_list_b, closed_list_b, best_ub, best_feasible_soln, num_tap_solved, current_level, tot_child, uncommon_number, common_number

def purge(open_list_b, open_list_f, closed_list_b, closed_list_f, max_level_f, max_level_b, beam_k, num_purged):
    starting = int(len(damaged_dict)/4)
    keep_f =[]
    not_kept = []
    if max_level_f > starting:
        values_ofn = np.ones((beam_k, max_level_f - 1)) * np.inf
        indices_ofn = np.ones((beam_k, max_level_f - 1)) * np.inf

        for idx, ofn in enumerate(open_list_f):
            cur_lev = ofn.level
            if cur_lev > starting:
                try:
                    cur_max = np.max(values_ofn[:, cur_lev-(starting+1)])
                    max_idx = np.argmax(values_ofn[:, cur_lev-(starting+1)])
                except:
                    pdb.set_trace()

                if ofn.f < cur_max:
                    indices_ofn[max_idx, cur_lev-(starting+1)] = idx
                    values_ofn[max_idx, cur_lev-(starting+1)] = ofn.f
            else:
                keep_f.append(idx)
        indices_ofn = indices_ofn.ravel()
        indices_ofn = indices_ofn[indices_ofn < 1000]

        keepinds = np.concatenate((np.array(keep_f), indices_ofn), axis=None).astype(int)
        
        not_kept = np.delete(np.arange(len(open_list_f)), keepinds)
        for i in not_kept:
            not_kept_node = open_list_f[i]
            closed_list_f.append(not_kept_node)

        open_list_f = list(np.array(open_list_f)[keepinds])
    num_purged += len(not_kept)

    keep_b =[]
    not_kept = []
    max_level_b = len(damaged_dict) - max_level_b

    if max_level_b > starting:
        values_ofn = np.ones((beam_k, max_level_b - 1)) * np.inf
        indices_ofn = np.ones((beam_k, max_level_b - 1)) * np.inf

        for idx, ofn in enumerate(open_list_b):

            cur_lev = len(damaged_dict) - ofn.level
            if cur_lev > starting:
                try:
                    cur_max = np.max(values_ofn[:, cur_lev-(starting+1)])
                    max_idx = np.argmax(values_ofn[:, cur_lev-(starting+1)])
                except:
                    pdb.set_trace()

                if ofn.f < cur_max:
                    indices_ofn[max_idx, cur_lev-(starting+1)] = idx
                    values_ofn[max_idx, cur_lev-(starting+1)] = ofn.f
            else:
                keep_b.append(idx)

        indices_ofn = indices_ofn.ravel()
        indices_ofn = indices_ofn[indices_ofn < 1000]
        keepinds = np.concatenate((np.array(keep_b), indices_ofn), axis=None).astype(int)

        not_kept = np.delete(np.arange(len(open_list_b)), keepinds)
        for i in not_kept:
            not_kept_node = open_list_b[i]
            closed_list_b.append(not_kept_node)
        open_list_b = list(np.array(open_list_b)[keepinds])

    num_purged += len(not_kept)

    return open_list_b, open_list_f, closed_list_b, closed_list_f, num_purged


def search(start_node, end_node, best_ub, greedy_soln, beam_search=False, beam_k=None):
    """Returns the best order to visit the set of nodes"""

    # ideas in Holte: (search that meets in the middle)
    # another idea is to expand the min priority, considering both backward and forward open list
    # pr(n) = max(f(n), 2g(n)) - min priority expands
    # U is best solution found so far - stops when - U <=max(C,fminF,fminB,gminF +gminB + eps)
    # this is for front to end
    print('best_ub: ', best_ub)
    max_level_b = 0
    max_level_f = 0

    iter_count = 0
    kb, kf = 0, 0

    # solution comes from the greedy heuristic
    best_feasible_soln = Node()
    best_feasible_soln.g = best_ub
    best_feasible_soln.path = greedy_soln

    damaged_links = damaged_dict.keys()

    # Initialize both open and closed list for forward and backward directions
    open_list_f = []
    closed_list_f = []
    open_list_f.append(start_node)

    open_list_b = []
    closed_list_b = []
    open_list_b.append(end_node)

    num_tap_solved = 0
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

    tot_child = 0
    uncommon_number = 0
    common_number = 0
    num_purged = 0

    while len(open_list_f) > 0 or len(open_list_b) > 0:
        iter_count += 1
        search_direction = 'Forward'

        if len(open_list_f) <= len(open_list_b) and len(open_list_f) != 0:
            search_direction = 'Forward'
        else:
            if len(open_list_b) != 0:
                search_direction = 'Backward'
            else:
                search_direction = 'Forward'

        if search_direction == 'Forward':
            open_list_f, closed_list_f, best_ub, best_feasible_soln, num_tap_solved, level_f, tot_child, uncommon_number, common_number = expand_forward(
                start_node, end_node, open_list_b, open_list_f, closed_list_b, closed_list_f, best_ub, best_feasible_soln, num_tap_solved, max_level_b, front_to_end=False, uncommon_number=uncommon_number, tot_child=tot_child, common_number=common_number)
            max_level_f = max(max_level_f, level_f)

        else:
            open_list_b, closed_list_b, best_ub, best_feasible_soln, num_tap_solved, level_b, tot_child, uncommon_number, common_number = expand_backward(
                start_node, end_node, open_list_b, open_list_f, closed_list_b, closed_list_f, best_ub, best_feasible_soln, num_tap_solved, max_level_f, front_to_end=False, uncommon_number=uncommon_number, tot_child=tot_child, common_number=common_number)
            max_level_b = min(max_level_b, level_b)

        # check termination
        if len(open_list_b) > 0:
            minbnode = open_list_b[0]
            for index, item in enumerate(open_list_b):
                if item.f <= minbnode.f:
                    minbnode = item
                    kb = minbnode.f

        if len(open_list_f) > 0:
            minfnode = open_list_f[0]
            for index, item in enumerate(open_list_f):
                if item.f <= minfnode.f:
                    minfnode = item
                    kf = minfnode.f

        if max(kf, kb) >= best_ub and best_feasible_soln.path is not None:
            return best_feasible_soln.path, best_feasible_soln.g, num_tap_solved, tot_child, uncommon_number, common_number, num_purged

        if (len(open_list_f) == 0 or len(open_list_b) == 0) and best_feasible_soln.path is not None:
            return best_feasible_soln.path, best_feasible_soln.g, num_tap_solved, tot_child, uncommon_number, common_number, num_purged

        if max_level_b > int(len(damaged_dict)/4) or max_level_f > int(len(damaged_dict)/4):
            if iter_count % 10 == 0:

                if beam_search:
                    open_list_b, open_list_f, closed_list_b, closed_list_f, num_purged = purge(
                        open_list_b, open_list_f, closed_list_b, closed_list_f,
                        max_level_f, max_level_b, beam_k, num_purged)
        #


        # if iter_count % 25 == 0:
        #
        #     if kf < kb:
        #         uncommon_number, common_number = set_bounds_bif(minfnode, open_list_b, end_node, front_to_end=True, best_feasible_soln=best_feasible_soln, uncommon_number=uncommon_number, common_number=common_number, get_feas=True)
        #     else:
        #         uncommon_number, common_number = set_bounds_bib(minbnode, open_list_f, start_node, front_to_end=True, best_feasible_soln=best_feasible_soln, uncommon_number=uncommon_number, common_number=common_number, get_feas=True)
        # #
        # if best_feasible_soln.g < best_ub:
        #    best_ub = best_feasible_soln.g
        #    print('new_best:', best_ub)

        bar.update(iter_count)
    if best_feasible_soln.path is None:
        print('{} is {}'.format('bestpath', 'none'))
        pdb.set_trace()

    if len(best_feasible_soln.path) != len(damaged_links):
        print('{} is not {}'.format('bestpath', 'full length'))
        pdb.set_trace()

    return best_feasible_soln.path, best_feasible_soln.g, num_tap_solved, tot_child, uncommon_number, common_number, num_purged



def worst_benefit(before, links_to_remove, before_eq_tstt, relax=False, bsearch=False, ext_name=''):

    if relax:
        ext_name = '_relax'
    elif bsearch:
        ext_name = ext_name
    fname = before.save_dir + '/worst_benefit_dict' + ext_name
    if not os.path.exists(fname + extension):
        # print('worst benefit analysis is running ...')
        # for each bridge, find the effect on TSTT when that bridge is removed
        # while keeping others

        wb = {}
        not_fixed = []
        for link in links_to_remove:
            test_net = deepcopy(before)
            not_fixed = [link]
            test_net.not_fixed = set(not_fixed)

            tstt = solve_UE(net=test_net, eval_seq=True)

            memory[frozenset(test_net.not_fixed)] = tstt
            # if tstt - before_eq_tstt <0 :
            #     pdb.set_trace()
            wb[link] = max(tstt - before_eq_tstt, 0)
            # print(tstt)
            # print(link, wb[link])

        save(fname, wb)
    else:
        wb = load(fname)

    return wb


def best_benefit(after, links_to_remove, after_eq_tstt, relax=False, bsearch=False, ext_name=''):

    if relax:
        ext_name = '_relax'
    elif bsearch:
        ext_name = ext_name
    fname = after.save_dir + '/best_benefit_dict' + ext_name

    if not os.path.exists(fname + extension):
        # print('best benefit analysis is running ...')

        # for each bridge, find the effect on TSTT when that bridge is removed
        # while keeping others
        bb = {}
        to_visit = links_to_remove
        added = []
        for link in links_to_remove:
            test_net = deepcopy(after)
            added = [link]
            not_fixed = set(to_visit).difference(set(added))
            test_net.not_fixed = set(not_fixed)
            tstt_after = solve_UE(net=test_net, eval_seq=True)

            memory[frozenset(test_net.not_fixed)] = tstt_after

            # seq_list.append(Node(link_id=link, parent=None, net=test_net, tstt_after=tstt_after,
            # tstt_before=after_eq_tstt, level=1, damaged_dict=damaged_dict))
            # if after_eq_tstt - tstt_after < 0 :
            #     pdb.set_trace()
            bb[link] = max(after_eq_tstt - tstt_after, 0)

            # print(tstt_after)
            # print(link, bb[link])

        save(fname, bb)
        # save(seq_list, 'seq_list' + SNAME)
    else:
        bb = load(fname)
        # seq_list = load('seq_list' + SNAME)

    return bb


def state_after(damaged_links, save_dir, relax=False, real=False, bsearch=False, ext_name=''):
    if relax:
        ext_name = '_relax'
    elif real:
        ext_name = '_real'

    elif bsearch:
        ext_name = ext_name
    fname = save_dir + '/net_after' + ext_name
    if not os.path.exists(fname + extension):
        net_after = create_network(NETFILE, TRIPFILE)
        net_after.not_fixed = set(damaged_links)

        after_eq_tstt = solve_UE(net=net_after, eval_seq=True)

        memory[frozenset(net_after.not_fixed)] = after_eq_tstt

        save(fname, net_after)
        save(fname + '_tstt', after_eq_tstt)

    else:
        net_after = load(fname)
        after_eq_tstt = load(fname + '_tstt')

    return net_after, after_eq_tstt


def state_before(damaged_links, save_dir, relax=False, real=False, bsearch=False, ext_name=''):
    if relax:
        ext_name = '_relax'
    elif real:
        ext_name = '_real'

    elif bsearch:
        ext_name = ext_name

    fname = save_dir + '/net_before' + ext_name
    if not os.path.exists(fname + extension):

        net_before = create_network(NETFILE, TRIPFILE)
        net_before.not_fixed = set([])

        before_eq_tstt = solve_UE(net=net_before, eval_seq=True, flows=True)
        memory[frozenset(net_before.not_fixed)] = before_eq_tstt

        save(fname, net_before)
        save(fname + '_tstt', before_eq_tstt)

    else:
        net_before = load(fname)
        before_eq_tstt = load(fname + '_tstt')

    return net_before, before_eq_tstt

def safety(wb, bb):
    for a_link in wb:
        if bb[a_link] < wb[a_link]:
            worst = bb[a_link]
            bb[a_link] = wb[a_link]
            wb[a_link] = worst
    return wb, bb


def eval_state(state, after, damaged_links):

    test_net = deepcopy(after)
    added = []

    for link in state:
        added.append(link)

    not_fixed = set(damaged_links).difference(set(added))
    test_net.not_fixed = set(not_fixed)

    tstt_after = solve_UE(net=test_net)
    memory[frozenset(test_net.not_fixed)] = tstt_after

    return tstt_after


def preprocessing(damaged_links, net_after):

    samples = []
    X_train = []
    y_train = []

    damaged_links = [i for i in damaged_links]

    for k, v in memory.items():
        pattern = np.ones(len(damaged_links))
        state = [damaged_links.index(i) for i in k]
        pattern[[state]] = 0
        X_train.append(pattern)
        y_train.append(v[1])

    ns = 1
    card_P = len(damaged_links)
    denom = 2 ^ card_P

    for i in range(card_P):
        # nom = ns * ncr(card_P, i)
        nom = ns * comb(card_P, i)
        num_to_sample = math.ceil(nom / denom) // 4
        print(num_to_sample)
        for j in range(num_to_sample):
            pattern = np.zeros(len(damaged_links))
            state = random.sample(damaged_links, i)
            TSTT = eval_state(state, net_after, damaged_links)

            state = [damaged_links.index(i) for i in state]
            pattern[[state]] = 1

            X_train.append(pattern)
            y_train.append(TSTT)

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import SGDRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from tensorflow import keras
    # poly_features = PolynomialFeatures(degree=2, include_bias=False)
    # X_train = poly_features.fit_transform(X_train)
    # reg = RandomForestRegressor(n_estimators=50)
    # reg = MLPRegressor((100,75,50),early_stopping=True, verbose=True)

    c = list(zip(X_train, y_train))

    random.shuffle(c)

    X_train, y_train = zip(*c)

    X_train_full = np.array(X_train)
    y_train_full = np.array(y_train)

    meany = np.mean(y_train_full)
    stdy = np.std(y_train_full)
    y_train_full = (y_train_full - meany) / stdy

    cutt = int(X_train_full.shape[0] * 0.1)
    X_train = X_train_full[cutt:]
    y_train = y_train_full[cutt:]
    X_valid = X_train_full[:cutt]
    y_valid = y_train_full[:cutt]

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
        100, activation='relu', input_shape=X_train.shape[1:]))
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(
        learning_rate=0.001))
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=30)
    history = model.fit(X_train, y_train, validation_data=(
        X_valid, y_valid), epochs=1000, callbacks=[early_stopping_cb])

    ##Test##
    state = random.sample(damaged_links, 1)
    TSTT = eval_state(state, net_after, damaged_links)
    state = [damaged_links.index(i) for i in state]
    pattern = np.zeros(len(damaged_links))
    pattern[[state]] = 1
    # pattern = poly_features.transform(pattern.reshape(1,-1))
    predicted_TSTT = model.predict(pattern.reshape(1, -1)) * stdy + meany
    print('predicted tstt vs real tstt:', predicted_TSTT, TSTT)

    state = random.sample(damaged_links, 7)
    TSTT = eval_state(state, net_after, damaged_links)
    state = [damaged_links.index(i) for i in state]
    pattern = np.zeros(len(damaged_links))
    pattern[[state]] = 1
    predicted_TSTT = model.predict(pattern.reshape(1, -1)) * stdy + meany
    print('predicted tstt vs real tstt:', predicted_TSTT, TSTT)

    state = random.sample(damaged_links, 11)
    TSTT = eval_state(state, net_after, damaged_links)
    state = [damaged_links.index(i) for i in state]
    pattern = np.zeros(len(damaged_links))
    pattern[[state]] = 1
    predicted_TSTT = model.predict(pattern.reshape(1, -1)) * stdy + meany
    print('predicted tstt vs real tstt:', predicted_TSTT, TSTT)

    # state = random.sample(damaged_links, 18)
    # TSTT = eval_state(state, net_after, damaged_links)
    # state = [damaged_links.index(i) for i in state]
    # pattern = np.zeros(len(damaged_links))
    # pattern[[state]] = 1
    # predicted_TSTT = model.predict(pattern.reshape(1, -1))*1e5
    # print('predicted tstt vs real tstt:', predicted_TSTT, TSTT)
    # pdb.set_trace()
    return model, meany, stdy


def importance_factor_solution(net_before, after_eq_tstt, before_eq_tstt):
    start = time.time()
    tap_solved = 0

    fname = net_before.save_dir + '/importance_factor_bound'
    if not os.path.exists(fname + extension):
        print('Finding the importance factor solution ...')

        tot_flow = 0
        if_net = deepcopy(net_before)

        for ij in if_net.link:
            tot_flow += if_net.link[ij]['flow']

        damaged_links = damaged_dict.keys()
        if_dict = {}
        for link_id in damaged_links:
            link_flow = if_net.link[link_id]['flow']
            if_dict[link_id] = link_flow / tot_flow

        sorted_d = sorted(if_dict.items(), key=lambda x: x[1])
        path, if_importance = zip(*sorted_d)
        path = path[::-1]
        if_importance = if_importance[::-1]
        print('importance factor path: ', path)
        print('importances: ', if_importance)


        elapsed = time.time() - start
        bound, eval_taps, _ = eval_sequence(
            if_net, path, after_eq_tstt, before_eq_tstt, if_dict, importance=True, damaged_dict=damaged_dict)
        tap_solved += eval_taps

        save(fname + '_obj', bound)
        save(fname + '_path', path)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', tap_solved)
    else:

        bound = load(fname + '_obj')
        path = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    return bound, path, elapsed, tap_solved


def brute_force(net_after, after_eq_tstt, before_eq_tstt, is_approx=False):
    start = time.time()
    tap_solved = 0
    damaged_links = damaged_dict.keys()
    approx_ext = ''
    verbose = False
    if is_approx:
        approx_ext = '_approx'
        verbose = True

    fname = net_after.save_dir + '/min_seq' + approx_ext

    if not os.path.exists(fname + extension):

        print('Finding the optimal sequence...')

        all_sequences = itertools.permutations(damaged_links)

        # seq_dict = {}
        i = 0
        min_cost = 1000000000000000
        min_seq = None

        for sequence in all_sequences:

            seq_net = deepcopy(net_after)
            cost, eval_taps, _ = eval_sequence(
                seq_net, sequence, after_eq_tstt, before_eq_tstt, is_approx=is_approx, damaged_dict=damaged_dict)
            tap_solved += eval_taps
            # seq_dict[sequence] = cost

            if cost < min_cost:
                min_cost = cost
                min_seq = sequence

            i += 1
            if verbose:
                print(i)
                print('min cost found so far: ', min_cost)
                print('min seq found so far: ', min_seq)

        # def compute(*args):
        #     seq_net = args[1]
        #     sequence = args[0]
        #     cost, eval_taps, _ = eval_sequence(
        #         seq_net, sequence, after_eq_tstt, before_eq_tstt)
        #     # print('sequence: {}, cost: {}'.format(sequence, cost))
        #     return cost, sequence, eval_taps

        # def all_seq_generator():
        #     for seq in itertools.permutations(damaged_links):
        #         yield seq, deepcopy(net_after)

        # with mp.Pool(processes=4) as p:
        #     data = p.map(compute, all_seq_generator())

        # min_cost, min_seq, tap_solved = min(data)
        # pdb.set_trace()

        elapsed = time.time() - start
        bound, _, _ = eval_sequence(net, min_seq, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict)
        save(fname + '_obj', bound)
        save(fname + '_path', min_seq)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', tap_solved)
    else:
        min_cost = load(fname + '_obj')
        min_seq = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')
    print('brute_force solution: ', min_seq)
    return min_cost, min_seq, elapsed, tap_solved


def greedy_heuristic(net_after, after_eq_tstt, before_eq_tstt):
    start = time.time()

    tap_solved = 0

    fname = net_after.save_dir + '/greedy_solution'

    if not os.path.exists(fname + extension):
        print('Finding the greedy_solution ...')
        tap_solved = 0

        tot_days = sum(damaged_dict.values())
        damaged_links = [link for link in damaged_dict.keys()]

        eligible_to_add = damaged_links

        test_net = deepcopy(net_after)
        after_ = after_eq_tstt

        path = []
        for i in range(len(damaged_links)):
            improvements = []
            new_tstts = []
            for link in eligible_to_add:

                added = [link]

                not_fixed = set(eligible_to_add).difference(set(added))
                test_net.not_fixed = set(not_fixed)

                after_fix_tstt = solve_UE(net=test_net)
                tap_solved += 1

                diff = after_ - after_fix_tstt
                if wb[link] > diff:
                    wb[link] = diff
                if bb[link] < diff:
                    bb[link] = diff

                remaining = diff * \
                    (tot_days - damaged_dict[link])

                improvements.append(remaining)
                new_tstts.append(after_fix_tstt)

            min_index = np.argmax(improvements)
            link_to_add = eligible_to_add[min_index]
            path.append(link_to_add)
            after_ = new_tstts[min_index]
            eligible_to_add.remove(link_to_add)
            tot_days -= damaged_dict[link_to_add]

        net = deepcopy(net_after)

        elapsed = time.time() - start

        bound, _, _ = eval_sequence(net, path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict)

        save(fname + '_obj', bound)
        save(fname + '_path', path)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', tap_solved)
    else:
        bound = load(fname + '_obj')
        path = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    return bound, path, elapsed, tap_solved


def local_search(greedy_path):
    pass

def get_wb(damaged_links, save_dir, approx, relax=False, bsearch=False, ext_name=''):
    net_after, after_eq_tstt = state_after(damaged_links, save_dir, relax=relax,  bsearch=bsearch, ext_name=ext_name)
    net_before, before_eq_tstt = state_before(damaged_links, save_dir, relax=relax, bsearch=bsearch, ext_name=ext_name)

    save(save_dir + '/damaged_dict', damaged_dict)

    net_before.save_dir = save_dir
    net_after.save_dir = save_dir

    wb = worst_benefit(net_before, damaged_links, before_eq_tstt, relax=relax, bsearch=bsearch, ext_name=ext_name)
    bb = best_benefit(net_after, damaged_links, after_eq_tstt, relax=relax,  bsearch=bsearch, ext_name=ext_name)
    wb, bb = safety(wb, bb)

    # approx solution
    if approx:
        model, meany, stdy = preprocessing(damaged_links, net_after)
        approx_obj, approx_soln, approx_elapsed, approx_num_tap = brute_force(
            net_after, after_eq_tstt, before_eq_tstt, is_approx=True)

        print('approx obj: {}, approx path: {}'.format(
            approx_obj, approx_soln))

    start_node = Node(tstt_after=after_eq_tstt)
    start_node.before_eq_tstt = before_eq_tstt
    start_node.after_eq_tstt = after_eq_tstt
    start_node.realized = 0
    start_node.realized_u = 0
    start_node.level = 0
    start_node.visited, start_node.not_visited = set(
        []), set(damaged_links)
    start_node.fixed, start_node.not_fixed = set([]), set(damaged_links)


    end_node = Node(tstt_before=before_eq_tstt, forward=False)
    end_node.before_eq_tstt = before_eq_tstt
    end_node.after_eq_tstt = after_eq_tstt
    end_node.level = len(damaged_links)
    
    end_node.visited, end_node.not_visited = set([]), set(damaged_links)
    
    end_node.fixed, end_node.not_fixed = set(
        damaged_links), set([])

    start_node.relax = relax
    end_node.relax = relax

    return wb, bb, start_node, end_node, net_after, net_before, after_eq_tstt, before_eq_tstt, None


if __name__ == '__main__':

    net_name = args.net_name
    num_broken = args.num_broken
    approx = args.approx
    reps = args.reps
    beam_search = args.beamsearch
    # beam_k = int(args.beamk)
    graphing = args.graphing
    scenario_file = args.scenario
    before_after = args.onlybeforeafter
    strength = args.strength
    full = args.full
    rand_gen = args.random
    opt = True

    NETWORK = os.path.join(FOLDER, net_name)
    JSONFILE = os.path.join(NETWORK, net_name.lower() + '.geojson')
    NETFILE = os.path.join(NETWORK, net_name + "_net.tntp")
    TRIPFILE = os.path.join(NETWORK, net_name + "_trips.tntp")

    SAVED_FOLDER_NAME = "saved"
    PROJECT_ROOT_DIR = "."
    SAVED_DIR = os.path.join(PROJECT_ROOT_DIR, SAVED_FOLDER_NAME)
    os.makedirs(SAVED_DIR, exist_ok=True)

    NETWORK_DIR = os.path.join(SAVED_DIR, NETWORK)
    os.makedirs(NETWORK_DIR, exist_ok=True)

    if graphing:
        get_common_numbers()
        # get_tables(NETWORK_DIR)
    else:
        if rand_gen:

            for rep in range(reps):
                memory = {}

                net = create_network(NETFILE, TRIPFILE)
                net.not_fixed = set([])
                solve_UE(net=net)

                f = "flows.txt"
                file_created = False
                st = time.time()
                while not file_created:
                    if os.path.exists(f):
                        file_created = True

                    if time.time() - st > 10:
                        popen = subprocess.call(args, stdout=subprocess.DEVNULL)

                    net.link = {}
                    if file_created:
                        with open(f, "r") as flow_file:
                            for line in flow_file.readlines():
                                try:
                                    ij = str(line[:line.find(' ')])
                                    line = line[line.find(' '):].strip()
                                    flow = float(line[:line.find(' ')])
                                    line = line[line.find(' '):].strip()
                                    cost = float(line.strip())


                                    if flow != 0.0 and cost != 9999.0:
                                        net.link[ij] = {}
                                        net.link[ij] = flow
                                except:
                                    break
                        os.remove('flows.txt')

                import operator
                sorted_d = sorted(net.link.items(), key=operator.itemgetter(1))[::-1]

                if num_broken > 75:
                    cutind = 200
                else:
                    cutind = 75

                try:
                    sorted_d = sorted_d[:cutind]
                except:
                    sorted_d = sorted_d


                all_links = [lnk[0] for lnk in sorted_d]
                all_links = sorted(all_links)
                np.random.seed((rep+1)*42+int(num_broken))

                netg = Network(NETFILE, TRIPFILE)
                G = nx.DiGraph()


                G.add_nodes_from(np.arange(len(netg.node)) + 1)
                edge_list = []
                for alink in netg.link:
                    edge_list.append((int(netg.link[alink].tail), int(netg.link[alink].head)))
                G.add_edges_from(edge_list)

                damaged_links = []
                decoy = deepcopy(all_links)
                i = 0

                dG = deepcopy(G)
                prev_dG = deepcopy(G)
                del G

                while i < int(num_broken):
                    ij = np.random.choice(decoy, 1, replace=False)[0]
                    u = netg.link[ij].tail
                    v = netg.link[ij].head
                    dG.remove_edge(u, v)

                    all_reachable = True
                    for od in netg.ODpair.values():
                        if not nx.has_path(dG, od.origin, od.destination):
                            all_reachable = False
                            dG = deepcopy(prev_dG)
                            break

                    if all_reachable:
                        i += 1
                        damaged_links.append(ij)
                        prev_dG = deepcopy(dG)

                del netg
                del dG
                del prev_dG

                print('damaged_links are created')

                # damaged_links = np.random.choice(all_links, num_broken, replace=False)
                damaged_dict = {}

                for a_link in damaged_links:
                    damaged_dict[a_link] = np.random.uniform(30, 90, 1)[0]

                print(rep)

                if len(damaged_links) > 6:
                    opt=False

                SCENARIO_DIR = NETWORK_DIR
                ULT_SCENARIO_DIR = os.path.join(SCENARIO_DIR, str(num_broken))
                os.makedirs(ULT_SCENARIO_DIR, exist_ok=True)

                repetitions = get_folders(ULT_SCENARIO_DIR)

                if len(repetitions) == 0:
                    max_rep = -1
                else:
                    num_scenario = [int(i) for i in repetitions]
                    max_rep = max(num_scenario)
                cur_scnario_num = max_rep + 1

                ULT_SCENARIO_REP_DIR = os.path.join(
                    ULT_SCENARIO_DIR, str(cur_scnario_num))

                os.makedirs(ULT_SCENARIO_REP_DIR, exist_ok=True)

                damaged_links = damaged_dict.keys()
                num_damaged = len(damaged_links)

                save_dir = ULT_SCENARIO_REP_DIR

                if before_after:
                    damaged_links = damaged_dict.keys()
                    num_damaged = len(damaged_links)
                    net_after, after_eq_tstt = state_after(damaged_links, save_dir)
                    net_before, before_eq_tstt = state_before(damaged_links, save_dir)
                    t = PrettyTable()
                    t.title = net_name + ' with ' + str(num_broken) + ' broken bridges'
                    t.field_names = ['Before EQ TSTT', 'After EQ TSTT']
                    t.add_row([before_eq_tstt, after_eq_tstt])
                    print(t)
                else:
                    benefit_analysis_st = time.time()
                    wb, bb, start_node, end_node, net_after, net_before, after_eq_tstt, before_eq_tstt, _ = get_wb(
                        damaged_links, save_dir, approx)
                    benefit_analysis_elapsed = time.time() - benefit_analysis_st

                    # approx solution
                    if approx:
                        model, meany, stdy = preprocessing(damaged_links, net_after)
                        approx_obj, approx_soln, approx_elapsed, approx_num_tap = brute_force(
                            net_after, after_eq_tstt, before_eq_tstt, is_approx=True)

                        print('approx obj: {}, approx path: {}'.format(
                            approx_obj, approx_soln))

                    ### Get greedy solution ###
                    greedy_obj, greedy_soln, greedy_elapsed, greedy_num_tap = greedy_heuristic(
                        net_after, after_eq_tstt, before_eq_tstt)

                    ### Get feasible solution using importance factor ###
                    importance_obj, importance_soln, importance_elapsed, importance_num_tap = importance_factor_solution(
                        net_before, after_eq_tstt, before_eq_tstt)

                    best_benefit_taps = num_damaged
                    worst_benefit_taps = num_damaged

                    ## Get optimal solution via brute force ###
                    if opt:
                        opt_obj, opt_soln, opt_elapsed, opt_num_tap = brute_force(
                            net_after, after_eq_tstt, before_eq_tstt)

                        print('optimal obj: {}, optimal path: {}'.format(opt_obj, opt_soln))

                    memory1 = deepcopy(memory)
                    # feasible_soln_taps = num_damaged * (num_damaged + 1) / 2.0
                    if full:
                        print('full')
                        algo_num_tap = best_benefit_taps + worst_benefit_taps
                        best_ub = greedy_obj

                        fname = save_dir + '/algo_solution'

                        if not os.path.exists(fname + extension):
                            search_start = time.time()
                            algo_path, algo_obj, search_tap_solved, tot_child, uncommon_number, common_number, _ = search(
                                start_node, end_node, best_ub, greedy_soln)
                            search_elapsed = time.time() - search_start + greedy_elapsed

                            net_after, after_eq_tstt = state_after(damaged_links, save_dir, real=True)
                            net_before, before_eq_tstt = state_before(damaged_links, save_dir, real=True)

                            first_net = deepcopy(net_after)
                            first_net.relax = False
                            algo_obj, _, _ = eval_sequence(
                                first_net, algo_path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict)

                            algo_num_tap += search_tap_solved + greedy_num_tap
                            algo_elapsed = search_elapsed + benefit_analysis_elapsed

                            save(fname + '_obj', algo_obj)
                            save(fname + '_path', algo_path)
                            save(fname + '_num_tap', algo_num_tap)
                            save(fname + '_elapsed', algo_elapsed)
                            save(fname + '_totchild', tot_child)
                            save(fname + '_uncommon', uncommon_number)
                            save(fname + '_common', common_number)
                        else:
                            algo_obj = load(fname + '_obj')
                            algo_path = load(fname + '_path')
                            algo_num_tap = load(fname + '_num_tap')
                            algo_elapsed = load(fname + '_elapsed')

                        print('---------------FULLOBJ')
                        print('full obj: ', algo_obj)

                    if beam_search:

                        ks = [2, int(len(damaged_links)/2)]

                        for k in ks:

                            fname = save_dir + '/beamsearch_solution' + '_k' + str(k)
                            ext_name = '_k' + str(k)
                            del memory
                            memory = {}
                            memory = deepcopy(memory1)

                            # wb, bb, start_node, end_node, net_after, net_before, after_eq_tstt, before_eq_tstt, benefit_analysis_st = get_wb(
                            #     damaged_links, save_dir, approx, relax=False, bsearch=True, ext_name=ext_name)
                            # benefit_analysis_elapsed = time.time() - benefit_analysis_st

                            beamsearch_num_tap = best_benefit_taps + worst_benefit_taps
                            best_bound = greedy_obj

                            start_node.relax = False
                            end_node.relax = False

                            if not os.path.exists(fname + extension):
                                search_start = time.time()
                                beamsearch_path, beamsearch_obj, beamsearch_tap_solved, tot_childbs, uncommon_numberbs, common_numberbs, num_purgebs = search(
                                    start_node, end_node, best_bound, greedy_soln, beam_search=beam_search, beam_k=k)
                                search_elapsed = time.time() - search_start + greedy_elapsed

                                net_after, after_eq_tstt = state_after(damaged_links, save_dir, real=True)
                                net_before, before_eq_tstt = state_before(damaged_links, save_dir, real=True)

                                first_net = deepcopy(net_after)
                                first_net.relax = False
                                beamsearch_obj, _, _ = eval_sequence(
                                    first_net, beamsearch_path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict)

                                beamsearch_num_tap += beamsearch_tap_solved + greedy_num_tap
                                beamsearch_elapsed = search_elapsed + benefit_analysis_elapsed

                                save(fname + '_obj', beamsearch_obj)
                                save(fname + '_path', beamsearch_path)
                                save(fname + '_num_tap', beamsearch_num_tap)
                                save(fname + '_elapsed', beamsearch_elapsed)
                                save(fname + '_totchild', tot_childbs)
                                save(fname + '_uncommon', uncommon_numberbs)
                                save(fname + '_common', common_numberbs)
                                save(fname + '_num_purge', num_purgebs)

                            else:
                                beamsearch_obj = load(fname + '_obj')
                                beamsearch_path = load(fname + '_path')
                                beamsearch_num_tap = load(fname + '_num_tap')
                                beamsearch_elapsed = load(fname + '_elapsed')

                            # beamsearch with gap 1e-4

                            fname = save_dir + '/r_algo_solution' + '_k' + str(k)
                            ext_name = 'r_k' + str(k)
                            print('reg bsearch finished')
                            del memory
                            memory = deepcopy(memory1)

                            # wb, bb, start_node, end_node, net_after, net_before, after_eq_tstt, before_eq_tstt, benefit_analysis_st = get_wb(
                            #     damaged_links, save_dir, approx, relax=True, bsearch=True, ext_name=ext_name)
                            # benefit_analysis_elapsed = time.time() - benefit_analysis_st

                            r_algo_num_tap = best_benefit_taps + worst_benefit_taps
                            best_bound = greedy_obj


                            start_node.relax = True
                            end_node.relax = True

                            if not os.path.exists(fname + extension):
                                search_start = time.time()
                                r_algo_path, r_algo_obj, r_search_tap_solved, tot_childr, uncommon_numberr, common_numberr, num_purger = search(
                                    start_node, end_node, best_bound, greedy_soln, beam_search=beam_search, beam_k=k)
                                search_elapsed = time.time() - search_start + greedy_elapsed

                                net_after, after_eq_tstt = state_after(damaged_links, save_dir, real=True)
                                net_before, before_eq_tstt = state_before(damaged_links, save_dir, real=True)

                                first_net = deepcopy(net_after)
                                first_net.relax = False
                                r_algo_obj, _, _ = eval_sequence(
                                    first_net, r_algo_path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict)

                                r_algo_num_tap += r_search_tap_solved + greedy_num_tap
                                r_algo_elapsed = search_elapsed + benefit_analysis_elapsed

                                save(fname + '_obj', r_algo_obj)
                                save(fname + '_path', r_algo_path)
                                save(fname + '_num_tap', r_algo_num_tap)
                                save(fname + '_elapsed', r_algo_elapsed)
                                save(fname + '_totchild', tot_childr)
                                save(fname + '_uncommon', uncommon_numberr)
                                save(fname + '_common', common_numberr)
                                save(fname + '_num_purge', num_purger)

                            else:
                                r_algo_obj = load(fname + '_obj')
                                r_algo_path = load(fname + '_path')
                                r_algo_num_tap = load(fname + '_num_tap')
                                r_algo_elapsed = load(fname + '_elapsed')
                            print('------')
                            print('k: ', k)
                            print('obj: ', beamsearch_obj)
                            print('purged#: ', num_purgebs)
                    t = PrettyTable()
                    t.title = net_name + ' with ' + str(num_broken) + ' broken bridges'
                    t.field_names = ['Method', 'Objective', 'Run Time', '# TAP']
                    if opt:
                        t.add_row(['OPTIMAL', opt_obj, opt_elapsed, opt_num_tap])
                    if approx:
                        t.add_row(['APPROX', approx_obj, approx_elapsed, approx_num_tap])
                    if beam_search:
                        t.add_row(['BeamSearch', beamsearch_obj, beamsearch_elapsed, beamsearch_num_tap])

                    if full:
                        t.add_row(['ALGORITHM', algo_obj, algo_elapsed, algo_num_tap])
                    if rand_gen:
                        t.add_row(['BeamSearchLG', r_algo_obj,
                                   r_algo_elapsed, r_algo_num_tap])
                    t.add_row(['GREEDY', greedy_obj, greedy_elapsed, greedy_num_tap])
                    t.add_row(['IMPORTANCE', importance_obj,
                               importance_elapsed, importance_num_tap])
                    print(t)

                    print('bs totchild vs uncommon vs common vs purged')
                    print(tot_childbs, uncommon_numberbs, common_numberbs, num_purgebs)
                    print('r totchild vs uncommon vs common vs purged')
                    print(tot_childr, uncommon_numberr, common_numberr, num_purger)

                    print('PATHS')
                    if full:
                        print('algorithm: ', algo_path)
                        print('---------------------------')
                    if beam_search:
                        print('beamsearch: ', beamsearch_path)
                        print('---------------------------')
                    print('greedy: ', greedy_soln)
                    print('---------------------------')
                    print('importance factors: ', importance_soln)

        else:
            damaged_dict_ = get_broken_links(JSONFILE, scenario_file)
            # num_broken = len(damaged_dict)

            opt = True
            if num_broken > 8:
                opt = False


            memory = {}
            net = create_network(NETFILE, TRIPFILE)

            all_links = damaged_dict_.keys()
            damaged_links = np.random.choice(list(all_links), num_broken, replace=False)
            # repair_days = [net.link[a_link].length for a_link in damaged_links]
            removals = []

            # net.not_fixed = set([])
            # solve_UE(net=net)
            # file_created = False
            # f = 'flows.txt'
            # st = time.time()
            # while not file_created:
            #     if os.path.exists(f):
            #         file_created = True

            #     if time.time()-st >10:
            #         popen = subprocess.call(args, stdout=subprocess.DEVNULL)

            #     if file_created:
            #         with open(f, "r") as flow_file:
            #             for line in flow_file.readlines():
            #                 try:
            #                     ij = str(line[:line.find(' ')])
            #                     line = line[line.find(' '):].strip()
            #                     flow = float(line[:line.find(' ')])
            #                     line = line[line.find(' '):].strip()
            #                     cost = float(line.strip())
            #                     net.link[ij].flow = flow
            #                     net.link[ij].cost = cost
            #                 except:
            #                     break

            #         os.remove('flows.txt')

            # for ij in damaged_links:
            #     if net.link[ij].flow == 0:
            #         removals.append(ij)




            # min_rd = min(repair_days)
            # max_rd = max(repair_days)
            # med_rd = np.median(repair_days)
            damaged_dict = {}

            for a_link in damaged_links:
                if a_link in removals:
                    continue
                damaged_dict[a_link] = damaged_dict_[a_link]

                # repair_days = net.link[a_link].length
                # if repair_days > med_rd:
                #     repair_days += (max_rd - repair_days) * 0.3
                # y = ((repair_days - min_rd) / (min_rd - max_rd)
                #      * (MIN_DAYS - MAX_DAYS) + MIN_DAYS)
                # mu = y
                # std = y * 0.3
                # damaged_dict[a_link] = np.random.normal(mu, std, 1)[0]

            del damaged_dict_
            num_broken = len(damaged_dict)

            SCENARIO_DIR = os.path.join(NETWORK_DIR, strength)
            ULT_SCENARIO_DIR = os.path.join(SCENARIO_DIR, str(num_broken))
            os.makedirs(ULT_SCENARIO_DIR, exist_ok=True)

            repetitions = get_folders(ULT_SCENARIO_DIR)

            if len(repetitions) == 0:
                max_rep = -1
            else:
                num_scenario = [int(i) for i in repetitions]
                max_rep = max(num_scenario)
            cur_scnario_num = max_rep + 1

            ULT_SCENARIO_REP_DIR = os.path.join(
                ULT_SCENARIO_DIR, str(cur_scnario_num))

            os.makedirs(ULT_SCENARIO_REP_DIR, exist_ok=True)

            damaged_links = damaged_dict.keys()
            num_damaged = len(damaged_links)

            save_dir = ULT_SCENARIO_REP_DIR



            if before_after:
                damaged_links = damaged_dict.keys()
                num_damaged = len(damaged_links)
                net_after, after_eq_tstt = state_after(damaged_links, save_dir)
                net_before, before_eq_tstt = state_before(damaged_links, save_dir)
                t = PrettyTable()
                t.title = net_name + ' with ' + str(num_broken) + ' broken bridges'
                t.field_names = ['Before EQ TSTT', 'After EQ TSTT']
                t.add_row([before_eq_tstt, after_eq_tstt])
                print(t)
            else:

                wb, bb, start_node, end_node, net_after, net_before, after_eq_tstt, before_eq_tstt, benefit_analysis_st = get_wb(damaged_links, save_dir, approx, relax=False)
                benefit_analysis_elapsed = time.time() - benefit_analysis_st



                # approx solution
                if approx:
                    model, meany, stdy = preprocessing(damaged_links, net_after)
                    approx_obj, approx_soln, approx_elapsed, approx_num_tap = brute_force(
                        net_after, after_eq_tstt, before_eq_tstt, is_approx=True)

                    print('approx obj: {}, approx path: {}'.format(
                        approx_obj, approx_soln))



                ### Get greedy solution ###
                greedy_obj, greedy_soln, greedy_elapsed, greedy_num_tap = greedy_heuristic(
                    net_after, after_eq_tstt, before_eq_tstt)

                ### Get feasible solution using importance factor ###
                importance_obj, importance_soln, importance_elapsed, importance_num_tap = importance_factor_solution(
                    net_before, after_eq_tstt, before_eq_tstt)

                best_benefit_taps = num_damaged
                worst_benefit_taps = num_damaged

                ## Get optimal solution via brute force ###
                if opt:
                    opt_obj, opt_soln, opt_elapsed, opt_num_tap = brute_force(
                        net_after, after_eq_tstt, before_eq_tstt)

                    print('optimal obj: {}, optimal path: {}'.format(opt_obj, opt_soln))

                # feasible_soln_taps = num_damaged * (num_damaged + 1) / 2.0
                if full:
                    print('full')
                    algo_num_tap = best_benefit_taps + worst_benefit_taps
                    best_ub = importance_obj

                    fname = save_dir + '/algo_solution'

                    if not os.path.exists(fname + extension):
                        search_start = time.time()
                        algo_path, algo_obj, search_tap_solved, _, _, _, _ = search(
                            start_node, end_node, best_ub)
                        search_elapsed = time.time() - search_start + importance_elapsed

                        net_after, after_eq_tstt = state_after(damaged_links, save_dir, real=True)
                        net_before, before_eq_tstt = state_before(damaged_links, save_dir, real=True)


                        first_net = deepcopy(net_after)
                        first_net.relax = False
                        algo_obj, _, _ = eval_sequence(
                            first_net, algo_path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict)

                        algo_num_tap += search_tap_solved + importance_num_tap
                        algo_elapsed = search_elapsed + benefit_analysis_elapsed


                        save(fname + '_obj', algo_obj)
                        save(fname + '_path', algo_path)
                        save(fname + '_num_tap', algo_num_tap)
                        save(fname + '_elapsed', algo_elapsed)
                    else:
                        algo_obj = load(fname + '_obj')
                        algo_path = load(fname + '_path')
                        algo_num_tap = load(fname + '_num_tap')
                        algo_elapsed = load(fname + '_elapsed')

                # print('Sequence found: {}, cost: {}, number of TAPs solved: {}, time elapsed: {}'.format(
                #     algo_path, algo_obj, algo_num_tap, algo_elapsed))

                # if not only_beam:
                #     # algo with gap 1e-4
                #     del memory
                #     memory = {}

                #     r_algo_num_tap = best_benefit_taps + worst_benefit_taps
                #     best_bound = np.inf

                #     fname = save_dir + '/r_algo_solution'

                #     start_node.relax = True
                #     end_node.relax = True

                #     if not os.path.exists(fname + extension):
                #         search_start = time.time()
                #         r_algo_path, r_algo_obj, r_search_tap_solved = search(
                #             start_node, end_node, best_bound)
                #         search_elapsed = time.time() - search_start

                #         net_after, after_eq_tstt = state_after(damaged_links, save_dir, real=True)
                #         net_before, before_eq_tstt = state_before(damaged_links, save_dir, real=True)

                #         first_net = deepcopy(net_after)
                #         first_net.relax = False
                #         r_algo_obj, _, _ = eval_sequence(
                #             first_net, r_algo_path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict)


                #         r_algo_num_tap += r_search_tap_solved
                #         r_algo_elapsed = search_elapsed + benefit_analysis_elapsed

                #         save(fname + '_obj', r_algo_obj)
                #         save(fname + '_path', r_algo_path)
                #         save(fname + '_num_tap', r_algo_num_tap)
                #         save(fname + '_elapsed', r_algo_elapsed)
                #     else:
                #         r_algo_obj = load(fname + '_obj')
                #         r_algo_path = load(fname + '_path')
                #         r_algo_num_tap = load(fname + '_num_tap')
                #         r_algo_elapsed = load(fname + '_elapsed')


                if beam_search:
                    del memory
                    memory = {}

                    wb, bb, start_node, end_node, net_after, net_before, after_eq_tstt, before_eq_tstt, benefit_analysis_st = get_wb(damaged_links, save_dir, approx, relax=False, bsearch=True)
                    benefit_analysis_elapsed = time.time() - benefit_analysis_st

                    beamsearch_num_tap = best_benefit_taps + worst_benefit_taps
                    best_bound = importance_obj

                    fname = save_dir + '/beamsearch_solution'

                    start_node.relax = False
                    end_node.relax = False

                    if not os.path.exists(fname + extension):
                        search_start = time.time()
                        beamsearch_path, beamsearch_obj, beamsearch_tap_solved, _, _, _, _ = search(
                            start_node, end_node, best_bound, beam_search=beam_search, beam_k=beam_k)
                        search_elapsed = time.time() - search_start + importance_elapsed

                        net_after, after_eq_tstt = state_after(damaged_links, save_dir, real=True)
                        net_before, before_eq_tstt = state_before(damaged_links, save_dir, real=True)

                        first_net = deepcopy(net_after)
                        first_net.relax = False
                        beamsearch_obj, _, _ = eval_sequence(
                            first_net, beamsearch_path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict)


                        beamsearch_num_tap += beamsearch_tap_solved + importance_num_tap
                        beamsearch_elapsed = search_elapsed + benefit_analysis_elapsed

                        save(fname + '_obj', beamsearch_obj)
                        save(fname + '_path', beamsearch_path)
                        save(fname + '_num_tap', beamsearch_num_tap)
                        save(fname + '_elapsed', beamsearch_elapsed)
                    else:
                        beamsearch_obj = load(fname + '_obj')
                        beamsearch_path = load(fname + '_path')
                        beamsearch_num_tap = load(fname + '_num_tap')
                        beamsearch_elapsed = load(fname + '_elapsed')



                t = PrettyTable()
                t.title = net_name + ' with ' + str(num_broken) + ' broken bridges'
                t.field_names = ['Method', 'Objective', 'Run Time', '# TAP']
                if opt:
                    t.add_row(['OPTIMAL', opt_obj, opt_elapsed, opt_num_tap])
                if approx:
                    t.add_row(['APPROX', approx_obj, approx_elapsed, approx_num_tap])
                if beam_search:
                    t.add_row(['BeamSearch', beamsearch_obj, beamsearch_elapsed, beamsearch_num_tap])
                if full:
                    t.add_row(['ALGORITHM', algo_obj, algo_elapsed, algo_num_tap])
                #t.add_row(['ALGORITHM_low_gap', r_algo_obj,
                #            r_algo_elapsed, r_algo_num_tap])
                t.add_row(['GREEDY', greedy_obj, greedy_elapsed, greedy_num_tap])
                t.add_row(['IMPORTANCE', importance_obj,
                           importance_elapsed, importance_num_tap])
                print(t)

                print('PATHS')
                if full:
                    print('algorithm: ', algo_path)
                    print('---------------------------')
                if beam_search:
                    print('beamsearch: ', beamsearch_path)
                    print('---------------------------')
                print('greedy: ', greedy_soln)
                print('---------------------------')
                print('importance factors: ', importance_soln)

