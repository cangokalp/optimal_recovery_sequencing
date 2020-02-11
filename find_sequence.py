from sequence_utils import *
import random
import operator as op
from functools import reduce
import itertools
from prettytable import PrettyTable
import operator

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

extension = '.pickle'

class Node():
    """A node class for bi-directional search for pathfinding"""

    def __init__(self, visited=None, link_id=None, parent=None, net=None, tstt_after=None, tstt_before=None, level=None, damaged_dict=None, forward=True):
        self.forward = forward
        self.parent = parent
        self.level = level
        self.link_id = link_id
        self.path = []
        self.net = net
        self.tstt_before = tstt_before
        self.tstt_after = tstt_after
        self.damaged_dict = damaged_dict
        self.g = 0
        self.h = 0
        self.f = 0
        self.assign_char()

    def assign_char(self):

        if self.parent is not None:
            self.benefit = self.tstt_after - self.tstt_before
            self.days = self.damaged_dict[self.link_id]
            prev_path = deepcopy(self.parent.path)
            prev_path.append(self.link_id)
            self.path = prev_path
            self.visited = set(self.path)
            self.days_past = self.parent.days_past + self.days
            self.before_eq_tstt = self.parent.before_eq_tstt
            self.after_eq_tstt = self.parent.after_eq_tstt

            self.realized = self.parent.realized + \
                (self.tstt_before - self.before_eq_tstt) * self.days

            self.not_visited = set(self.damaged_dict.keys()
                                   ).difference(self.visited)

            self.forward = self.parent.forward
            if self.forward:
                self.net.not_fixed = self.not_visited
                self.net.fixed = self.visited

            else:
                self.net.not_fixed = self.visited
                self.net.fixed = self.not_visited

        else:
            if self.link_id is not None:
                self.path = [self.link_id]
                self.realized = (self.tstt_before -
                                 self.before_eq_tstt) * self.days
                self.days_past = self.days

            else:
                self.realized = 0
                self.days = 0
                self.days_past = self.days

    def __eq__(self, other):
        return self.net.fixed == other.net.fixed


def create_network(netfile=None, tripfile=None):
    return Network(netfile, tripfile)


def get_successors_f(node, wb, bb):
    """given a state, returns list of bridges that has not been fixed yet"""
    not_visited = node.not_visited
    successors = []

    if node.level != 0:
        tail = node.path[-1]
        for a_link in not_visited:
            if wb[a_link] * node.damaged_dict[tail] - bb[tail] * node.damaged_dict[a_link] > 0:
                continue
            successors.append(a_link)
    else:
        successors = not_visited

    return successors


def get_successors_b(node, wb, bb):
    """given a state, returns list of bridges that has not been removed yet"""
    not_visited = node.not_visited
    successors = []

    if node.level != len(node.damaged_dict.keys()):
        tail = node.path[-1]
        for a_link in not_visited:
            if wb[tail] * node.damaged_dict[a_link] - bb[a_link] * node.damaged_dict[tail] > 0:
                continue
            successors.append(a_link)
    else:
        successors = not_visited

    return successors


def expand_sequence_f(node, a_link, level, damaged_dict):
    """given a link and a node, it expands the sequence"""
    tstt_before = node.tstt_after
    net = deepcopy(node.net)
    net.link[a_link].add_link_back()
    added = [a_link]
    net.not_fixed = set(net.not_fixed).difference(set(added))
    tstt_after = solve_UE(net=net)
    node = Node(link_id=a_link, parent=node, net=net, tstt_after=tstt_after,
                tstt_before=tstt_before, level=level, damaged_dict=damaged_dict)

    return node


def expand_sequence_b(node, a_link, level, damaged_dict):
    """given a link and a node, it expands the sequence"""
    tstt_after = node.tstt_before
    net = deepcopy(node.net)
    net.link[a_link].remove()
    removed = [a_link]
    net.not_fixed = net.not_fixed.union(set(removed))
    tstt_before = solve_UE(net=net)
    node = Node(link_id=a_link, parent=node, net=net, tstt_after=tstt_after,
                tstt_before=tstt_before, level=level, damaged_dict=damaged_dict)
    return node


def orderlists(benefits, days, slack, reverse=True):

    if sum(benefits) > slack:
        benefits = np.array(benefits)
        benefits = [min(slack, x) for x in benefits]

    bang4buck = np.array(benefits) / np.array(days)

    days = [x for _, x in sorted(
        zip(bang4buck, days), reverse=reverse)]
    benefits = [x for _, x in sorted(
        zip(bang4buck, benefits), reverse=reverse)]

    return benefits, days


def get_minlb(node, fwd_node, bwd_node, orderedb_benefits, orderedw_benefits, ordered_days, forward_tstt, backward_tstt, backwards=False):

    slack = forward_tstt - backward_tstt

    if len(ordered_days) == 0:
        node.lb = node.ub = fwd_node.realized + bwd_node.realized
        return

    elif len(ordered_days) == 1:
        node.lb = node.ub = fwd_node.realized + bwd_node.realized + \
            (fwd_node.tstt_after - node.before_eq_tstt) * ordered_days[0]
        return

    b, days_b = orderlists(orderedb_benefits, ordered_days, slack)
    w, days_w = orderlists(orderedw_benefits, ordered_days, slack)

    uborig = node.ub
    backward_lb = node.lb
    backward_ub = node.ub

    ###### FIND LB FROM BACKWARDS #####
    if sum(w) < slack:
        for i in range(len(days_b)):
            if i == 0:
                bwd_w = deepcopy(w)
                bwd_days_w = deepcopy(days_w)
                b_tstt = bwd_node.tstt_before
            else:
                bwd_w = bwd_w[1:]
                bwd_days_w = bwd_days_w[1:]
                bwd_w, bwd_days_w = orderlists(bwd_w, bwd_days_w, slack_available, reverse=False)

            slack_available = forward_tstt - b_tstt
            #check this again
            if i==0:
                benefit = min(bwd_w[-1], slack_available)
            else:
                benefit = min(bwd_w[0], slack_available)
            b_tstt = b_tstt + benefit

            # if b_tstt == forward_tstt:
            # backward_lb += sum(ordered[:]) * (forward_tstt - node.before_eq_tstt)
            # backward_lb += sum(bwd_w[:]) * (forward_tstt - node.before_eq_tstt)
            # break
            if i ==0:
                backward_lb += max((b_tstt - node.before_eq_tstt),0) * bwd_days_w[-1]
            else:
                backward_lb += max((b_tstt - node.before_eq_tstt),
                               0) * bwd_days_w[0]
    else:
        bwd_days_w = deepcopy(days_w)
        maxi = max(bwd_days_w)
        bwd_days_w.remove(maxi)

        mini = min(bwd_days_w)
        bwd_days_w.remove(mini)

        top = mini * (forward_tstt - node.before_eq_tstt)
        bottom = maxi * (bwd_node.tstt_before - node.before_eq_tstt)
        backward_lb += bottom + \
            sum(bwd_days_w) * (bwd_node.tstt_before - node.before_eq_tstt) + top

    ###### FIND LB FROM FORWARDS #####
    if sum(b) > slack:
        for i in range(len(days_b)):

            if i == 0:
                fwd_b = deepcopy(b)
                fwd_days_b = deepcopy(days_b)
                b_tstt = fwd_node.tstt_after

            else:
                slack_available = b_tstt - backward_tstt
                benefit = min(fwd_b[0], slack_available)
                b_tstt = b_tstt - benefit
                fwd_b = fwd_b[1:]
                fwd_days_b = fwd_days_b[1:]
                fwd_b, fwd_days_b = orderlists(
                    fwd_b, fwd_days_b, slack_available)

            if b_tstt == backward_tstt:
                node.lb += sum(fwd_days_b[:]) * \
                    (backward_tstt - node.before_eq_tstt)
                break

            node.lb += max((b_tstt - node.before_eq_tstt), 0) * fwd_days_b[0]
    else:
        fwd_days_b = deepcopy(days_b)
        mini = min(fwd_days_b)
        fwd_days_b.remove(mini)
        node.lb += min(days_b) * (fwd_node.tstt_after - node.before_eq_tstt) + sum(fwd_days_b) * (backward_tstt - node.before_eq_tstt)

    ###### FIND UB FROM FORWARDS #####
    if sum(w) < slack:
        for i in range(len(days_w)):

            if i == 0:
                fwd_w = deepcopy(w)
                fwd_days_w = deepcopy(days_w)
                w_tstt = fwd_node.tstt_after
                node.ub += max((w_tstt - node.before_eq_tstt), 0) * max(days_w)

            else:
                slack_available = w_tstt - backward_tstt
                benefit = min(fwd_w[0], slack_available)
                w_tstt_prev = w_tstt
                w_tstt = w_tstt - benefit
                fwd_w = fwd_w[1:]
                fwd_days_w = fwd_days_w[1:]
                fwd_w, fwd_days_w = orderlists(
                    fwd_w, fwd_days_w, slack_available)

            if w_tstt == backward_tstt:
                node.ub += sum(fwd_days_w[:]) * \
                    (backward_tstt - node.before_eq_tstt)
                break

            node.ub += max((w_tstt - node.before_eq_tstt), 0) * fwd_days_w[0]
    else:
        fwd_days_w = deepcopy(days_w)
        maxi = max(fwd_days_w)
        fwd_days_w.remove(maxi)
        node.ub += max(days_w) * (fwd_node.tstt_after - node.before_eq_tstt) + \
            sum(fwd_days_w) * (forward_tstt - node.before_eq_tstt)

    ###### FIND UB FROM BACKWARDS #####
    if sum(b) > slack:
        for i in range(len(days_b)):
            if i == 0:
                bwd_b = deepcopy(b)
                bwd_days_b = deepcopy(days_b)
                w_tstt = bwd_node.tstt_before
            else:
                bwd_b = bwd_b[1:]
                bwd_days_b = bwd_days_b[1:]
                bwd_b, bwd_days_b = orderlists(
                    bwd_b, bwd_days_b, slack_available)

            slack_available = forward_tstt - w_tstt
            benefit = min(bwd_b[0], slack_available)
            w_tstt = w_tstt + benefit

            if i == len(days_b) - 1:
                backward_ub += bwd_days_b[-1] * \
                    (forward_tstt - node.before_eq_tstt)
                break

            if w_tstt == forward_tstt:
                backward_ub += sum(bwd_days_b[:]) * \
                    (forward_tstt - node.before_eq_tstt)
                break

            backward_ub += max((w_tstt - node.before_eq_tstt),
                               0) * bwd_days_b[0]
    else:
        bwd_days_b = deepcopy(days_b)
        mini = min(bwd_days_b)
        bwd_days_b.remove(mini)
        backward_ub += min(days_b) * (bwd_node.tstt_before + b[-1] - node.before_eq_tstt) + sum(
            bwd_days_b) * (forward_tstt - node.before_eq_tstt)

    if backward_ub < node.ub:
        node.ub = backward_ub

    if backward_lb > node.lb:
        node.lb = backward_lb

    if node.lb > node.ub:
        pdb.set_trace()


def set_bounds_bif(node, wb, bb, open_list_b, end_node, front_to_end=True, debug=False):

    sorted_d = sorted(node.damaged_dict.items(), key=lambda x: x[1])

    remaining = []
    eligible_backward_connects = []

    if front_to_end:
        eligible_backward_connects = [end_node]
    else:
        for other_end in open_list_b:
            if len(set(node.visited).intersection(other_end.visited)) == 0:
                eligible_backward_connects.append(other_end)

    minlb = np.inf
    maxub = -np.inf

    if debug:
        pdb.set_trace()
    if len(eligible_backward_connects) == 0:
        eligible_backward_connects = [end_node]

    for other_end in eligible_backward_connects:
        ordered_days = []
        orderedw_benefits = []
        orderedb_benefits = []

        node.ub = node.realized + other_end.realized
        node.lb = node.realized + other_end.realized

        union = node.visited.union(other_end.visited)
        remaining = set(node.damaged_dict.keys()).difference(union)

        for key, value in sorted_d:
            # print("%s: %s" % (key, value))
            if key in remaining:
                ordered_days.append(value)
                orderedw_benefits.append(wb[key])
                orderedb_benefits.append(bb[key])

        forward_tstt = node.tstt_after
        backward_tstt = other_end.tstt_before
        get_minlb(node, node, other_end, orderedb_benefits,
                  orderedw_benefits, ordered_days, forward_tstt, backward_tstt)

        if node.lb < minlb:
            minlb = node.lb
        if node.ub > maxub:
            maxub = node.ub

    node.lb = minlb
    node.ub = maxub


def set_bounds_bib(node, wb, bb, open_list_f, start_node, front_to_end=True):

    sorted_d = sorted(node.damaged_dict.items(), key=lambda x: x[1])

    remaining = []
    eligible_backward_connects = []

    if front_to_end:
        eligible_backward_connects = [start_node]
    else:
        for other_end in open_list_f:
            if len(set(node.visited).intersection(other_end.visited)) == 0:
                eligible_backward_connects.append(other_end)

    minlb = np.inf
    maxub = -np.inf
    if len(eligible_backward_connects) == 0:
        eligible_backward_connects = [start_node]

    for other_end in eligible_backward_connects:
        ordered_days = []
        orderedw_benefits = []
        orderedb_benefits = []

        node.ub = node.realized + other_end.realized
        node.lb = node.realized + other_end.realized

        union = node.visited.union(other_end.visited)
        remaining = set(node.damaged_dict.keys()).difference(union)

        for key, value in sorted_d:
            # print("%s: %s" % (key, value))
            if key in remaining:
                ordered_days.append(value)
                orderedw_benefits.append(wb[key])
                orderedb_benefits.append(bb[key])
        forward_tstt = other_end.tstt_after
        backward_tstt = node.tstt_before

        get_minlb(node, other_end, node, orderedb_benefits, orderedw_benefits,
                  ordered_days, forward_tstt, backward_tstt, backwards=True)

        if node.lb < minlb:
            minlb = node.lb
        if node.ub > maxub:
            maxub = node.ub

    node.lb = minlb
    node.ub = maxub


def expand_forward(damaged_dict, wb, bb, start_node, end_node, open_list_b, open_list_f, closed_list_b, closed_list_f,  best_soln, best_soln_node, best_path, num_tap_solved, front_to_end):
    debug = False
    # Get the current node
    current_node = open_list_f[0]
    current_index = 0
    for index, item in enumerate(open_list_f):
        if item.f < current_node.f:
            current_node = item
            current_index = index

    # Pop current off open list, add to closed list
    open_list_f.pop(current_index)
    # current_node = min(open_list_f)
    # open_list_f.remove(current_node)
    closed_list_f.append(current_node)

    if current_node.level >= 2:
        cur_visited = current_node.visited
        for other_end in open_list_b + closed_list_b:
            if len(set(other_end.visited).intersection(set(cur_visited))) == 0 and len(damaged_dict) - len(set(other_end.visited).union(set(cur_visited))) == 1:
                lo_link = set(damaged_dict.keys()).difference(
                    set(other_end.visited).union(set(cur_visited)))
                lo_link = lo_link.pop()
                cur_soln = current_node.g + other_end.g + \
                    current_node.tstt_after * damaged_dict[lo_link]
                cur_path = current_node.path + \
                    [str(lo_link)] + other_end.path[::-1]
                if cur_soln <= best_soln:
                    best_soln = cur_soln
                    best_path = cur_path
                    print('-------BEST_SOLN-----: ', best_soln)
                    print(best_path)

            if set(other_end.not_visited) == set(cur_visited):
                cur_soln = current_node.g + other_end.g
                cur_path = current_node.path + other_end.path[::-1]
                if cur_soln <= best_soln:
                    best_soln = cur_soln
                    best_path = cur_path
                    print('-------BEST_SOLN-----: ', best_soln)
                    print(best_path)

    # Found the goal
    if current_node == end_node:
        # print('at the end_node')
        return open_list_f, closed_list_f, best_soln, best_soln_node, best_path, num_tap_solved

    if current_node.f > best_soln:
        # print('current node is pruned')
        return open_list_f, closed_list_f, best_soln, best_soln_node, best_path, num_tap_solved

    # Generate children
    eligible_expansions = get_successors_f(current_node, wb, bb)
    children = []

    for a_link in eligible_expansions:

        # Create new node
        new_node = expand_sequence_f(
            current_node, a_link, level=current_node.level + 1, damaged_dict=damaged_dict)
        num_tap_solved += 1
        # Append
        children.append(new_node)

    # Loop through children
    for child in children:

        # set upper and lower bounds
        set_bounds_bif(child, wb, bb, open_list_b,
                       front_to_end=front_to_end, end_node=end_node, debug=debug)

        if child.ub <= best_soln:

            best_soln = child.ub
            best_soln_node = child
            if len(child.path) == len(child.damaged_dict):
                best_path = child.path

        if child.lb > child.ub:
            pdb.set_trace()

        # Create the f, g, and h values
        child.g = child.realized
        # child.h = child.lb - child.g
        # child.f = child.g + child.h
        child.f = child.lb

        if child.f > best_soln:
            continue

        # Child is already in the open list
        removal = []
        for open_node in open_list_f:
            if child == open_node:
                if child.g > open_node.g:
                    continue
                else:
                    removal.append(open_node)

        for open_node in removal:
            open_list_f.remove(open_node)

        # Child is on the closed list
        for closed_node in closed_list_f:
            if child == closed_node:
                if child.g > closed_node.g:
                    continue

        # Add the child to the open list
        open_list_f.append(child)
    return open_list_f, closed_list_f, best_soln, best_soln_node, best_path, num_tap_solved


def expand_backward(damaged_dict, wb, bb, start_node, end_node, open_list_b, open_list_f, closed_list_b, closed_list_f, best_soln, best_soln_node, best_path, num_tap_solved, front_to_end):

    # Loop until you find the end
    current_node = open_list_b[0]
    current_index = 0

    for index, item in enumerate(open_list_b):
        if item.f < current_node.f:
            current_node = item
            current_index = index

    # Pop current off open list, add to closed list
    open_list_b.pop(current_index)
    # current_node = min(open_list_b)
    # open_list_b.remove(current_node)

    closed_list_b.append(current_node)

    if len(current_node.visited) >= 2:
        cur_visited = current_node.visited
        for other_end in open_list_f + closed_list_f:
            if len(set(other_end.visited).intersection(set(cur_visited))) == 0 and len(damaged_dict) - len(set(other_end.visited).union(set(cur_visited))) == 1:
                lo_link = set(damaged_dict.keys()).difference(
                    set(other_end.visited).union(set(cur_visited)))
                lo_link = lo_link.pop()
                cur_soln = current_node.g + other_end.g + \
                    other_end.tstt_after * damaged_dict[lo_link]
                if cur_soln <= best_soln:
                    best_soln = cur_soln
                    best_path = other_end.path + \
                        [str(lo_link)] + current_node.path[::-1]
                    print('-------BEST_SOLN-----: ', best_soln)
                    print(best_path)

            elif set(other_end.not_visited) == set(cur_visited):
                cur_soln = current_node.g + other_end.g
                # print('current_soln: {}, best_soln: {}'.format(cur_soln, best_soln))
                if cur_soln <= best_soln:
                    best_soln = cur_soln
                    best_path = other_end.path + current_node.path[::-1]
                    print('-------BEST_SOLN-----: ', best_soln)
                    print(best_path)

    # Found the goal
    if current_node == start_node:
        # print('at the start node')
        return open_list_b, closed_list_b, best_soln, best_soln_node, best_path, num_tap_solved

    if current_node.f > best_soln:
        # print('current node is pruned')
        return open_list_b, closed_list_b, best_soln, best_soln_node, best_path, num_tap_solved

    # Generate children
    eligible_expansions = get_successors_b(current_node, wb, bb)

    children = []

    for a_link in eligible_expansions:

        # Create new node
        new_node = expand_sequence_b(
            current_node, a_link, level=current_node.level - 1, damaged_dict=damaged_dict)
        num_tap_solved += 1
        # Append
        children.append(new_node)

    # Loop through children

    for child in children:

        # set upper and lower bounds

        set_bounds_bib(child, wb, bb, open_list_f,
                       front_to_end=front_to_end, start_node=start_node)
        if child.ub <= best_soln:
            best_soln = child.ub
            best_soln_node = child
            if len(child.path) == len(child.damaged_dict):
                best_path = child.path[::-1]

        if child.lb > child.ub:
            pdb.set_trace()

        # Create the f, g, and h values
        child.g = child.realized
        # child.h = child.lb - child.g
        # child.f = child.g + child.h
        child.f = child.lb

        if child.f > best_soln:
            continue

        # Child is already in the open list
        removal = []
        for open_node in open_list_b:
            if child == open_node:
                if child.g > open_node.g:
                    continue
                else:
                    removal.append(open_node)

        for open_node in removal:
            open_list_b.remove(open_node)

        # Child is on the closed list
        for closed_node in closed_list_b:
            if child == closed_node:
                if child.g > closed_node.g:
                    continue

        # Add the child to the open list
        open_list_b.append(child)
    return open_list_b, closed_list_b, best_soln, best_soln_node, best_path, num_tap_solved


def search(damaged_dict, wb, bb, start_node, end_node, best_bound):
    """Returns the best order to visit the set of nodes"""
   

    # ideas in Holte: (search that meets in the middle)
    # another idea is to expand the min priority, considering both backward and forward open list
    # pr(n) = max(f(n), 2g(n)) - min priority expands
    # U is best solution found so far - stops when - U ≤max(C,fminF,fminB,gminF +gminB +ε)
    # this is for front to end

    iter_count = 0
    kb, kf = 0, 0
    best_soln = best_bound  # solution comes from the greedy heuristic
    best_path = None
    best_soln_node = None

    damaged_links = damaged_dict.keys()
    repair_days = damaged_dict.values()

    # Initialize both open and closed list for forward and backward directions
    open_list_f = []
    closed_list_f = []
    open_list_f.append(start_node)

    open_list_b = []
    closed_list_b = []
    open_list_b.append(end_node)

    num_tap_solved = 0

    while len(open_list_f) > 0 or len(open_list_b) > 0:

        iter_count += 1
        search_direction = 'Forward'

        if len(open_list_f) <= len(open_list_b) and len(open_list_f) != 0:
            search_direction = 'Forward'
        elif len(open_list_b) != 0:
                search_direction = 'Backward'
        else:
            break

        # print('search_direction: ', search_direction)
        # print('f length {} b length {}'.format(len(open_list_f), len(open_list_b)))
        
        if search_direction == 'Forward':
            open_list_f, closed_list_f, best_soln, best_soln_node, best_path, num_tap_solved = expand_forward(
                damaged_dict, wb, bb, start_node, end_node, open_list_b, open_list_f, closed_list_b, closed_list_f, best_soln, best_soln_node, best_path, num_tap_solved, front_to_end=False)
        else:
            open_list_b, closed_list_b, best_soln, best_soln_node, best_path, num_tap_solved = expand_backward(
                damaged_dict, wb, bb, start_node, end_node, open_list_b, open_list_f, closed_list_b, closed_list_f, best_soln, best_soln_node, best_path, num_tap_solved, front_to_end=False)

        if iter_count % 40 == 0:
            print('length of forward open list: ', len(open_list_f))
            print('length of backwards open list: ', len(open_list_b))

        # check termination
        if len(open_list_b) > 0:
            current_node = open_list_b[0]
            for index, item in enumerate(open_list_b):
                if item.f <= current_node.f:
                    current_node = item
                    kb = current_node.f

        if len(open_list_f) > 0:
            current_node = open_list_f[0]
            for index, item in enumerate(open_list_f):
                if item.f <= current_node.f:
                    current_node = item
                    kf = current_node.f

        if min(kf, kb) > best_soln:
            print('search ended')
            print('search_dir: ', search_direction)
            if best_path is None:
                pdb.set_trace()
            return best_path, best_soln, num_tap_solved

    if best_path is None:
        pdb.set_trace()
    return best_path, best_soln, num_tap_solved


def worst_benefit(before, links_to_remove, before_eq_tstt):
    fname = before.save_dir + '/worst_benefit_dict'
    if not os.path.exists(fname + extension):
        print('worst benefit analysis is running ...')
        # for each bridge, find the effect on TSTT when that bridge is removed
        # while keeping others

        wb = {}
        not_fixed = []
        for link in links_to_remove:
            test_net = deepcopy(before)
            test_net.link[link].remove()
            not_fixed = [link]
            test_net.not_fixed = set(not_fixed)

            tstt = solve_UE(net=test_net)
            wb[link] = tstt - before_eq_tstt
            # print(tstt)
            # print(link, wb[link])

        save(fname, wb)
    else:
        wb = load(fname)

    return wb


def best_benefit(after, links_to_remove, after_eq_tstt):
    fname = after.save_dir + '/best_benefit_dict'

    if not os.path.exists(fname + extension):
        print('best benefit analysis is running ...')

        # for each bridge, find the effect on TSTT when that bridge is removed
        # while keeping others
        bb = {}
        to_visit = links_to_remove
        added = []
        for link in links_to_remove:
            test_net = deepcopy(after)
            test_net.link[link].add_link_back()
            added = [link]
            not_fixed = set(to_visit).difference(set(added))
            test_net.not_fixed = set(not_fixed)
            tstt_after = solve_UE(net=test_net)

            # seq_list.append(Node(link_id=link, parent=None, net=test_net, tstt_after=tstt_after,
            # tstt_before=after_eq_tstt, level=1, damaged_dict=damaged_dict))
            bb[link] = after_eq_tstt - tstt_after
           
            # print(tstt_after)
            # print(link, bb[link])

        save(fname, bb)
        # save(seq_list, 'seq_list' + SNAME)
    else:
        bb = load(fname)
        # seq_list = load('seq_list' + SNAME)

    return bb


def state_after(damaged_links, save_dir):
    fname = save_dir + '/net_after'
    if not os.path.exists(fname + extension):
        net_after = create_network(NETFILE, TRIPFILE)
        for link in damaged_links:
            net_after.link[link].remove()
        net_after.not_fixed = set(damaged_links)

        after_eq_tstt = solve_UE(net=net_after)

        save(fname, net_after)
        save(fname + '_tstt', after_eq_tstt)

    else:
        net_after = load(fname)
        after_eq_tstt = load(fname + '_tstt')

    return net_after, after_eq_tstt


def state_before(damaged_links, save_dir):
    fname = save_dir + '/net_before'
    if not os.path.exists(fname + extension):

        net_before = create_network(NETFILE, TRIPFILE)
        net_before.not_fixed = set([])

        before_eq_tstt = solve_UE(net=net_before)

        save(fname, net_before)
        save(fname + '_tstt', before_eq_tstt)

    else:
        net_before = load(fname)
        before_eq_tstt = load(fname + '_tstt')

    return net_before, before_eq_tstt


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
        test_net.link[link].add_link_back()
        added.append(link)

    not_fixed = set(damaged_links).difference(set(added))
    test_net.not_fixed = set(not_fixed)

    tstt_after = solve_UE(net=test_net)
    return tstt_after


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def preprocessing(damaged_links, net_after):

    samples = []
    X_train = []
    y_train = []

    ns = 1
    card_P = len(damaged_links)
    denom = 2 ^ card_P
    for i in range(card_P):
        nom = ns * ncr(card_P, i)
        num_to_sample = int(nom / denom)
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

    # poly_features = PolynomialFeatures(degree=2, include_bias=False)
    # X_train = poly_features.fit_transform(X_train)
    reg = RandomForestRegressor(n_estimators=500)

    reg.fit(X_train, y_train)
    preds = reg.predict(X_train)
    mse = mean_squared_error(y_train, preds)
    print(np.sqrt(mse))

    ##Test##
    state = random.sample(damaged_links, 1)
    TSTT = eval_state(state, net_after, damaged_links)
    state = [damaged_links.index(i) for i in state]
    pattern = np.zeros(len(damaged_links))
    pattern[[state]] = 1
    # pattern = poly_features.transform(pattern.reshape(1,-1))
    predicted_TSTT = reg.predict(pattern.reshape(1, -1))
    print('predicted tstt vs real tstt:', predicted_TSTT, TSTT)
    pdb.set_trace()


def importance_factor_solution(net_before, after_eq_tstt, before_eq_tstt):
    start = time.time()
    tap_solved = 0
    
    fname = net_before.save_dir + '/importance_factor_bound'
    if not os.path.exists(fname + extension):
        print('Finding the importance factor solution ...')

        tot_flow = 0
        if_net = deepcopy(net_before)
        for ij in if_net.link:
            tot_flow += if_net.link[ij].flow

        damaged_links = if_net.damaged_dict.keys()
        ffp = 1
        if_dict = {}
        for link_id in damaged_links:
            link_flow = if_net.link[link_id].flow
            if_dict[link_id] = link_flow / tot_flow
            ffp -= if_dict[link_id]

        ffp = ffp * 100

        sorted_d = sorted(if_dict.items(), key=lambda x: x[1])
        path, if_importance = zip(*sorted_d)
        path = path[::-1]
        if_importance = if_importance[::-1]
        print('importance factor path: ', path)
        print('importances: ', if_importance)

        bound, eval_taps, _ = eval_sequence(
            if_net, path, after_eq_tstt, before_eq_tstt, if_dict, importance=True)

        tap_solved += eval_taps


        elapsed = time.time() - start

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


def brute_force(net_after, after_eq_tstt, before_eq_tstt):
    start = time.time()
    tap_solved = 0
    damaged_links = net_after.damaged_dict.keys()
    fname = net_after.save_dir + '/min_seq'
    if not os.path.exists(fname + extension):

        print('Finding the optimal sequence...')

        all_sequences = list(itertools.permutations(damaged_links))

        seq_dict = {}
        i = 0
        min_cost = 1000000000000000
        min_seq = None
        for sequence in all_sequences:

            seq_net = deepcopy(net_after)
            cost, eval_taps, _ = eval_sequence(
                seq_net, sequence, after_eq_tstt, before_eq_tstt)
            tap_solved += eval_taps
            # seq_dict[sequence] = cost

            if cost < min_cost:
                min_cost = cost
                min_seq = sequence

            i += 1

            print(i)
            print('min cost found so far: ', min_cost)
            print('min seq found so far: ', min_seq)

        # import operator
        # sorted_x = sorted(seq_dict.items(), key=operator.itemgetter(1))


        elapsed = time.time() - start
        save(fname + '_obj', min_cost)
        save(fname + '_path', min_seq)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', tap_solved)
    else:
        min_cost = load(fname + '_obj')
        min_seq = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    return min_cost, min_seq, elapsed, tap_solved


def greedy_heuristic(net_after, after_eq_tstt, before_eq_tstt):
    start = time.time()
    
    tap_solved = 0

    fname = net_after.save_dir + '/greedy_solution'

    if not os.path.exists(fname + extension):
        print('Finding the greedy_solution ...')
        tap_solved = 0
        
        damaged_dict = net_after.damaged_dict

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
                
                test_net.link[link].add_link_back()
                added = [link]

                not_fixed = set(eligible_to_add).difference(set(added))
                test_net.not_fixed = set(not_fixed)

                after_fix_tstt = solve_UE(net=test_net)
                tap_solved += 1

                test_net.link[link].remove()

                remaining = (after_ - after_fix_tstt) * (tot_days - damaged_dict[link])

                improvements.append(remaining)
                new_tstts.append(after_fix_tstt)

            min_index = np.argmax(improvements)
            link_to_add = eligible_to_add[min_index]
            path.append(link_to_add)
            after_ = new_tstts[min_index]
            eligible_to_add.remove(link_to_add)

        net = deepcopy(net_after)
        bound, _, _ = eval_sequence(net, path, after_eq_tstt, before_eq_tstt)

        elapsed = time.time() - start
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


def get_marginal_tstts(net, path, after_eq_tstt, before_eq_tstt):
    damaged_dict = net.damaged_dict

    _, _, tstt_list = eval_sequence(deepcopy(net), path, after_eq_tstt, before_eq_tstt)

    # tstt_list.insert(0, after_eq_tstt)
    
    days_list = []
    for link in path:
        days_list.append(damaged_dict[link])
    
    return tstt_list, days_list


def main(save_dir):
    np.random.seed(SEED)
    damaged_dict = read_scenario(sname=SNAME)
    damaged_links = [link for link in damaged_dict.keys()]

    take_out = np.random.choice(damaged_links, len(damaged_links) - int(NUM_LINKS), replace=False)
    save(save_dir + '/broken_links', take_out)
    
    for j in take_out:
        damaged_links.remove(j)
        del damaged_dict[j]

    num_damaged = len(damaged_dict)

    net_after, after_eq_tstt = state_after(damaged_links, save_dir)
    net_before, before_eq_tstt = state_before(damaged_links, save_dir)

    net_before.damaged_dict = damaged_dict
    net_after.damaged_dict = damaged_dict
    net_before.save_dir = save_dir
    net_after.save_dir = save_dir
    

    # preprocessing(damaged_links, net_after)
    benefit_analysis_st = time.time()
    wb = worst_benefit(net_before, damaged_links, before_eq_tstt)
    bb = best_benefit(net_after, damaged_links, after_eq_tstt)

    wb, bb = safety(wb, bb)
    # Create start and end node
    start_node = Node(tstt_after=after_eq_tstt,
                      net=net_after, damaged_dict=damaged_dict)
    start_node.before_eq_tstt = before_eq_tstt
    start_node.after_eq_tstt = after_eq_tstt
    start_node.realized = 0
    start_node.level = 0
    start_node.visited, start_node.not_visited = set([]), set(damaged_links)
    start_node.net.fixed, start_node.net.not_fixed = set(
        []), set(damaged_links)

    end_node = Node(tstt_before=before_eq_tstt,
                    net=net_before, damaged_dict=damaged_dict, forward=False)
    end_node.before_eq_tstt = before_eq_tstt
    end_node.after_eq_tstt = after_eq_tstt
    end_node.level = len(damaged_links)
    end_node.visited, end_node.not_visited = set([]), set(damaged_links)
    end_node.net.fixed, end_node.net.not_fixed = set(damaged_links), set([])
    benefit_analysis_elapsed = time.time() - benefit_analysis_st

    ### Get greedy solution ###
    greedy_obj, greedy_soln, greedy_elapsed, greedy_num_tap = greedy_heuristic(net_after, after_eq_tstt, before_eq_tstt)

    ### Get feasible solution using importance factor ###
    importance_obj, importance_soln, importance_elapsed, importance_num_tap = importance_factor_solution(net_before, after_eq_tstt, before_eq_tstt)

    best_benefit_taps = num_damaged
    worst_benefit_taps = num_damaged

    # feasible_soln_taps = num_damaged * (num_damaged + 1) / 2.0
    algo_num_tap = best_benefit_taps + worst_benefit_taps
    best_bound = np.inf

    fname = net_after.save_dir + '/algo_solution'

    if not os.path.exists(fname + extension):
        search_start = time.time()
        algo_path, algo_obj, search_tap_solved = search(
            damaged_dict, wb, bb, start_node, end_node, best_bound)
        search_elapsed = time.time() - search_start

        algo_num_tap += search_tap_solved
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

    ## Get optimal solution via brute force ###
    opt_obj, opt_soln, opt_elapsed, opt_num_tap = brute_force(net_after, after_eq_tstt, before_eq_tstt)

    # print('Sequence found: {}, cost: {}, number of TAPs solved: {}, time elapsed: {}'.format(
    #     algo_path, algo_obj, algo_num_tap, algo_elapsed))

    # shutil.rmtree(net_after.save_dir)

    t = PrettyTable()
    t.title = 'SiouxFalls' + ' with ' + NUM_LINKS + ' broken bridges' 
    t.field_names = ['Method', 'Objective', 'Run Time', '# TAP']
    t.add_row(['OPTIMAL', opt_obj, opt_elapsed, opt_num_tap])
    t.add_row(['ALGORITHM', algo_obj, algo_elapsed, algo_num_tap])
    t.add_row(['GREEDY', greedy_obj, greedy_elapsed, greedy_num_tap])
    t.add_row(['IMPORTANCE', importance_obj, importance_elapsed, importance_num_tap])
    print(t)


    # GRAPH THE RESULTS
    paths = [algo_path, opt_soln, greedy_soln, importance_soln]
    names = ['algorithm', 'brute-force', 'greedy', 'importance-factor']
    for path, name in zip(paths, names):
        tstt_list, days_list = get_marginal_tstts(net_after, path, after_eq_tstt, before_eq_tstt)
        graph_current(tstt_list, days_list, before_eq_tstt, after_eq_tstt, path, save_dir, name)
    
    pdb.set_trace()

if __name__ == '__main__':

    broken_bridges = ['6', '7', '8']
    repetitions = [10, 10, 5]

    for broken in broken_bridges:
        for reps in range(repetitions[broken_bridges.index(broken)]):

            NUM_LINKS = broken

            ULT_SCENARIO_DIR = os.path.join(SCENARIO_DIR, NUM_LINKS)
            ULT_SCENARIO_REP_DIR = os.path.join(ULT_SCENARIO_DIR, str(reps))
            
            os.makedirs(ULT_SCENARIO_REP_DIR, exist_ok=True)

            main(save_dir=ULT_SCENARIO_REP_DIR)
