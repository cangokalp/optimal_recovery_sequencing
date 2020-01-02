from test_add_remove import *

SNAME = 'Moderate_5'
NETFILE = "SiouxFalls/SiouxFalls_net.tntp"
TRIPFILE = "SiouxFalls/SiouxFalls_trips.tntp"


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
            self.realized = self.parent.realized + self.tstt_before * self.days
            
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
                self.realized = self.tstt_before * self.days
                self.days_past = self.days

            else:
                self.realized = 0
                self.days = 0
                self.days_past = self.days

    # def __eq__(self, other):
        # return self.visited == other.visited


def set_bounds_f(node, wb, bb, remaining):
    node.ub = node.realized
    node.lb = node.realized

    ordered_days = []
    orderedw_benefits = []
    orderedb_benefits = []

    sorted_d = sorted(node.damaged_dict.items(), key=lambda x: x[1])

    for key, value in sorted_d:
        # print("%s: %s" % (key, value))
        if key in remaining:
            ordered_days.append(value)
            orderedw_benefits.append(wb[key])
            orderedb_benefits.append(bb[key])

    bang4buck_b = np.array(orderedb_benefits) / np.array(ordered_days)
    bang4buck_w = np.array(orderedw_benefits) / np.array(ordered_days)

    days_b = [x for _, x in sorted(zip(bang4buck_b, ordered_days))]
    b = [x for _, x in sorted(zip(bang4buck_b, orderedb_benefits))]

    days_w = [x for _, x in sorted(zip(bang4buck_w, ordered_days))]
    w = [x for _, x in sorted(zip(bang4buck_w, orderedw_benefits))]

    # b
    for i in range(len(days_b)):
        if i == 0:
            b_tstt = node.tstt_after
        else:
            b_tstt = b_tstt - b[i - 1]

        node.lb += b_tstt * days_b[i]

    # w
    for i in range(len(days_w)):
        if i == 0:
            w_tstt = node.tstt_after
        else:
            w_tstt = w_tstt - w[i - 1]

        node.ub += w_tstt * days_w[i]

def set_bounds_b(node, wb, bb, remaining):
    node.ub = node.realized
    node.lb = node.realized

    ordered_days = []
    orderedw_benefits = []
    orderedb_benefits = []

    sorted_d = sorted(node.damaged_dict.items(), key=lambda x: x[1])

    for key, value in sorted_d:
        # print("%s: %s" % (key, value))
        if key in remaining:
            ordered_days.append(value)
            orderedw_benefits.append(wb[key])
            orderedb_benefits.append(bb[key])

    bang4buck_b = np.array(orderedb_benefits) / np.array(ordered_days)
    bang4buck_w = np.array(orderedw_benefits) / np.array(ordered_days)

    days_b = [x for _, x in sorted(zip(bang4buck_b, ordered_days))]
    b = [x for _, x in sorted(zip(bang4buck_b, orderedb_benefits))]

    days_w = [x for _, x in sorted(zip(bang4buck_w, ordered_days))]
    w = [x for _, x in sorted(zip(bang4buck_w, orderedw_benefits))]

    # b
    for i in range(len(days_b)):
        if i == 0:
            b_tstt = node.tstt_before
        else:
            b_tstt = b_tstt - b[i - 1]

        node.lb += b_tstt * days_b[i]

    # w
    for i in range(len(days_w)):
        if i == 0:
            w_tstt = node.tstt_before
        else:
            w_tstt = w_tstt - w[i - 1]

        node.ub += w_tstt * days_w[i]

def get_minlb(node, orderedb_benefits, orderedw_benefits, ordered_days, minlb, lb, ub):
    bang4buck_b = np.array(orderedb_benefits) / np.array(ordered_days)
    bang4buck_w = np.array(orderedw_benefits) / np.array(ordered_days)

    days_b = [x for _, x in sorted(
        zip(bang4buck_b, ordered_days), reverse=True)]
    b = [x for _, x in sorted(
        zip(bang4buck_b, orderedb_benefits), reverse=True)]

    days_w = [x for _, x in sorted(
        zip(bang4buck_w, ordered_days), reverse=True)]
    w = [x for _, x in sorted(
        zip(bang4buck_w, orderedw_benefits), reverse=True)]

    for i in range(len(days_b)):
        if i == 0:
            b_tstt = node.tstt_after
        else:
            b_tstt = b_tstt - b[i - 1]
            if b_tstt < 0:
                b_tstt = 0
                break
        lb += b_tstt * days_b[i]

    # w
    for i in range(len(days_w)):
        if i == 0:
            w_tstt = node.tstt_after
        else:
            w_tstt = w_tstt - w[i - 1]
            if w_tstt < 0:
                w_tstt = 0
                break

        ub += w_tstt * days_w[i]

    if lb < minlb:
        minlb = lb
        node.lb = lb
        node.ub = ub

    return minlb


def set_bounds_bif(node, wb, bb, open_list_b):
    ordered_days = []
    orderedw_benefits = []
    orderedb_benefits = []

    sorted_d = sorted(node.damaged_dict.items(), key=lambda x: x[1])

    remaining = {}
    eligible_backward_connects = []

    for other_end in open_list_b:
        if len(set(node.net.fixed).intersection(other_end.net.fixed)) == 0:
            eligible_backward_connects.append(other_end)
            remaining[other_end] = set(node.damaged_dict.keys()).difference(
                node.visited.union(other_end.visited))

    if len(eligible_backward_connects) == 0:
        go_through = sorted_d
        ub = node.realized
        lb = node.realized
        remaining = set(node.not_visited).difference(node.visited)
        backward_exists = False
    else:
        go_through = eligible_backward_connects
        ub = node.realized + other_end.realized
        lb = node.realized + other_end.realized
        backward_exists = True
    
    minlb = np.inf

    for key, value in sorted_d:
        if key in remaining:
            ordered_days.append(value)
            orderedw_benefits.append(wb[key])
            orderedb_benefits.append(bb[key])
    minlb = get_minlb(node, orderedb_benefits, orderedw_benefits, ordered_days, minlb, lb, ub)  


    if backward_exists:
        pdb.set_trace()
        for a_node in go_through:
            for key, value in sorted_d:
                # print("%s: %s" % (key, value))
                if key in remaining[a_node]:
                    ordered_days.append(value)
                    orderedw_benefits.append(wb[key])
                    orderedb_benefits.append(bb[key])
            min_lb = get_minlb(node, orderedb_benefits, orderedw_benefits, ordered_days, minlb, lb, ub)  



def set_bounds_bib(node, wb, bb, open_list_f, best_ub):

    ordered_days = []
    orderedw_benefits = []
    orderedb_benefits = []

    sorted_d = sorted(node.damaged_dict.items(), key=lambda x: x[1])

    remaining = {}
    eligible_backward_connects = []

    for other_end in open_list_f:
        if len(set(node.visited).intersection(other_end.visited)) == 0:
            eligible_backward_connects.append(other_end)
            remaining[other_end] = set(node.damaged_dict.keys()).difference(
                node.visited.union(other_end.visited))

    minlb = np.inf
    for other_end in eligible_backward_connects:
        node.ub = node.realized + other_end.realized
        node.lb = node.realized + other_end.realized

        for key, value in sorted_d:
            # print("%s: %s" % (key, value))
            if key in remaining[other_end]:
                ordered_days.append(value)
                orderedw_benefits.append(wb[key])
                orderedb_benefits.append(bb[key])

        bang4buck_b = np.array(orderedb_benefits) / np.array(ordered_days)
        bang4buck_w = np.array(orderedw_benefits) / np.array(ordered_days)

        days_b = [x for _, x in sorted(
            zip(bang4buck_b, ordered_days), reverse=True)]
        b = [x for _, x in sorted(
            zip(bang4buck_b, orderedb_benefits), reverse=True)]

        days_w = [x for _, x in sorted(
            zip(bang4buck_w, ordered_days), reverse=True)]
        w = [x for _, x in sorted(
            zip(bang4buck_w, orderedw_benefits), reverse=True)]

        for i in range(len(days_b)):
            if i == 0:
                b_tstt = node.tstt_before
            else:
                b_tstt = b_tstt - b[i - 1]
                if b_tstt < 0:
                    b_tstt = 0
            lb += b_tstt * days_b[i]

        # w
        for i in range(len(days_w)):
            if i == 0:
                w_tstt = node.tstt_before
            else:
                w_tstt = w_tstt - w[i - 1]
                if w_tstt < 0:
                    w_tstt = 0

            ub += w_tstt * days_w[i]

        if lb < minlb:
            minlb = lb
            node.lb = lb
            node.ub = ub


def get_successors_f(node, wb, bb):
    """given a state, returns list of bridges that has not been fixed yet"""
    not_visited = node.not_visited
    successors = []

    if node.level != 0:
        tail = node.path[-1]
        for a_link in not_visited:
            pdb.set_trace()
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

    if node.level != len(node.damaged_dict.keys()) + 1:
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
    net_after.link[a_link].remove()
    tstt_before = solve_UE(net=net)

    # tstt_before = find_tstt(net=net)
    node = Node(link_id=a_link, parent=node, net=net, tstt_after=tstt_after,
                tstt_before=tstt_before, level=level, damaged_dict=damaged_dict)
    return node


def expand_forward(damaged_dict, wb, bb, start_node, end_node, open_list_b, open_list_f, closed_list_b, closed_list_f,  best_soln, best_path):
    # Loop until you find the end
    print('forward search is in progress...')
    tap_solved = 0

    # Get the current node
    current_node = open_list_f[0]
    current_index = 0
    for index, item in enumerate(open_list_f):
        if item.f < current_node.f:
            current_node = item
            current_index = index

    # Pop current off open list, add to closed list
    open_list_f.pop(current_index)
    closed_list_f.append(current_node)

    if current_node.level > 3:
        cur_visited = current_node.visited
        for other_end in closed_list_b:
            if set(other_end.not_visited) == set(cur_visited):
                print('we have a solution')
                cur_soln = current_node.g + other_end.g
                best_soln = min(best_soln, cur_soln)
                best_path = current_node.path + other_end.path[::-1]

    # Found the goal
    # FIX: Fix below 
    if current_node == end_node:
        return current_node.path, None, None

    if current_node.f > best_soln:
        return None, open_list_f, closed_list_f

    # Generate children
    eligible_expansions = get_successors_f(current_node, wb, bb)
    children = []

    for a_link in eligible_expansions:

        # Create new node
        new_node = expand_sequence_f(
            current_node, a_link, level=current_node.level + 1, damaged_dict=damaged_dict)
        tap_solved += 1
        # Append
        children.append(new_node)

    print('open_list_f length: ', len(open_list_f))
    print('number of taps solved: ', tap_solved)

    # Loop through children
    for child in children:

        # set upper and lower bounds
        set_bounds_bif(child, wb, bb, open_list_b)
        best_soln = min(best_soln, child.ub)
        print(child.visited)
        print(child.ub, child.lb)

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
            # FIX: fix equal
            if child == open_node:
                if child.g > open_node.g:
                    continue
                else:
                    removal.append(open_node)

        for open_node in removal:
            open_list_f.remove(open_node)

        # Child is on the closed list
        for closed_node in closed_list_f:
            # FIX
            if child == closed_node:
                if child.g > closed_node.g:
                    continue

        # Add the child to the open list
        open_list_f.append(child)

    return None, open_list_f, closed_list_f, best_soln, best_path


def expand_backward(damaged_dict, wb, bb, start_node, end_node, open_list_b, open_list_f, closed_list_b, closed_list_f, best_soln, best_path):

    # Loop until you find the end
    print('backwards search is in progress...')
    tap_solved = 0

    # Get the current node
    pdb.set_trace()
    current_node = open_list_b[0]
    current_index = 0
    for index, item in enumerate(open_list_b):
        if item.f < current_node.f:
            current_node = item
            current_index = index

    # Pop current off open list, add to closed list
    open_list_b.pop(current_index)
    closed_list_b.append(current_node)

    if abs(len(current_node.damaged_dict) - current_node.level) > 3:
        cur_visited = current_node.visited
        for other_end in closed_list_b:
            if set(other_end.not_visited) == set(cur_visited):
                print('we have a solution')
                cur_soln = current_node.g + other_end.g
                best_soln = min(best_soln, cur_soln)
                best_path = other_end.path + current_node.path[::-1]

    # Found the goal
    if current_node == start_node:
        return current_node.path, None, None

    if current_node.f > best_ub:
        return None, open_list_b, closed_list_b

    # Generate children
    eligible_expansions = get_successors_b(current_node, wb, bb)

    children = []

    for a_link in eligible_expansions:

        # Create new node
        new_node = expand_sequence_b(
            current_node, a_link, level=current_node.level - 1, damaged_dict=damaged_dict)
        tap_solved += 1
        # Append
        children.append(new_node)

    print('open_list_b length: ', len(open_list_b))
    print('number of taps solved: ', tap_solved)
    # Loop through children
    for child in children:

        # set upper and lower bounds
        set_bounds_bib(child, wb, bb, open_list_f, best_ub)
        best_ub = min(best_ub, child.ub)

        # Create the f, g, and h values
        child.g = child.realized
        # child.h = child.lb - child.g
        # child.f = child.g + child.h
        child.f = child.lb

        if child.f > best_ub:
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
    return None, open_list_b, closed_list_b, best_soln, best_path


def search(damaged_dict, wb, bb, start_node, end_node, best_bound):
    """Returns the best order to visit the set of nodes"""
    kb, kf = 0, 0
    best_soln = best_bound  # solution comes from the greedy heuristic
    best_path = None

    damaged_links = damaged_dict.keys()
    repair_days = damaged_dict.values()

    # Initialize both open and closed list for forward and backward directions
    open_list_f = []
    closed_list_f = []
    # Add the start node
    open_list_f.append(start_node)

    open_list_b = []
    closed_list_b = []
    open_list_b.append(end_node)

    search_direction = 'Forward'

    while open_list_f and open_list_b:
        if len(open_list_f) <= len(open_list_b):
            # ideas in Holte: (search that meets in the middle)
            # another idea is to expand the min priority, considering both backward and forward open list
            # pr(n) = max(f(n), 2g(n)) - min priority expands
            # U is best solution found so far - stops when - U ≤max(C,fminF,fminB,gminF +gminB +ε)
            # this is for front to end
            search_direction = 'Forward'
        else:
            search_direction = 'Backward'

        if search_direction == 'Forward':
            path, open_list_f, closed_list_f, best_soln, best_path = expand_forward(
                damaged_dict, wb, bb, start_node, end_node, open_list_b, open_list_f, closed_list_b, closed_list_f, best_soln, best_path)
        else:
            pdb.set_trace()

            path, open_list_b, closed_list_b, best_soln, best_path = expand_backward(
                damaged_dict, wb, bb, start_node, end_node, open_list_b, open_list_f, closed_list_b, closed_list_f, best_soln, best_path)

        # check termination

        current_node = open_list_b[0]
        for index, item in enumerate(open_list_b):
            if item.f < current_node.f:
                current_node = item
                kb = current_node.f

        current_node = open_list_f[0]
        for index, item in enumerate(open_list_f):
            if item.f < current_node.f:
                current_node = item
                kf = current_node.f

        if max(kf, kb) >= best_soln:
            return best_soln, best_path

        if path is not None:
            return path


def create_network(netfile=None, tripfile=None):
    return Network(netfile, tripfile)

def worst_benefit(before, links_to_remove, before_eq_tstt):
    if not os.path.exists('saved_dictionaries/' + 'worst_benefit_dict' + SNAME + '.pickle'):

        # for each bridge, find the effect on TSTT when that bridge is removed
        # while keeping others
        wb = {}
        not_fixed = []
        print('worst_benefit')
        for link in links_to_remove:
            test_net = deepcopy(before)
            test_net.link[link].remove()
            not_fixed.append(link)
            test_net.not_fixed = set(not_fixed)

            tstt = solve_UE(net=test_net)
            print(tstt)
            wb[link] = tstt - before_eq_tstt
        # save dictionary
        save(wb, 'worst_benefit_dict' + SNAME)
        pdb.set_trace()
    else:
        wb = load('worst_benefit_dict' + SNAME)

    return wb

def best_benefit(after, links_to_remove, after_eq_tstt):
    if not os.path.exists('saved_dictionaries/' + 'best_benefit_dict' + SNAME + '.pickle'):

        # for each bridge, find the effect on TSTT when that bridge is removed
        # while keeping others
        bb = {}
        to_visit = links_to_remove
        added = []
        print('best_benefit')
        for link in links_to_remove:
            test_net = deepcopy(after)
            test_net.link[link].add_link_back()
            added.append(link)
            not_fixed = set(to_visit).difference(set(added))
            test_net.not_fixed = set(not_fixed)
            tstt_after = solve_UE(net=test_net)

            print(tstt_after)

            # seq_list.append(Node(link_id=link, parent=None, net=test_net, tstt_after=tstt_after,
                                 # tstt_before=after_eq_tstt, level=1, damaged_dict=damaged_dict))
            bb[link] = after_eq_tstt - tstt_after
        pdb.set_trace()

        save(bb, 'best_benefit_dict' + SNAME)
        # save(seq_list, 'seq_list' + SNAME)

    else:
        bb = load('best_benefit_dict' + SNAME)
        # seq_list = load('seq_list' + SNAME)

    return bb

def state_after(damaged_links):
    if not os.path.exists('saved_dictionaries/' + 'net_after' + SNAME + '.pickle'):

        net_after = create_network(NETFILE, TRIPFILE)
        for link in damaged_links:
            net_after.link[link].remove()
        net_after.not_fixed = set(damaged_links)

        after_eq_tstt = solve_UE(net=net_after)
        print('after eq: ', after_eq_tstt)
        save(net_after, 'net_after' + SNAME)
        save(after_eq_tstt, 'tstt_after' + SNAME)

    else:
        net_after = load('net_after' + SNAME)
        after_eq_tstt = load('tstt_after' + SNAME)

    return net_after, after_eq_tstt

def state_before(damaged_links):

    if not os.path.exists('saved_dictionaries/' + 'net_before' + SNAME + '.pickle'):
        net_before = create_network(NETFILE, TRIPFILE)
        net_before.not_fixed = set([])
        
        before_eq_tstt = solve_UE(net=net_before)
        print('before eq: ', before_eq_tstt)

        save(net_before, 'net_before' + SNAME)
        save(before_eq_tstt, 'tstt_before' + SNAME)

    else:
        net_before = load('net_before' + SNAME)
        before_eq_tstt = load('tstt_before' + SNAME)

    return net_before, before_eq_tstt

def eval_sequence(net, order_list, after_eq_tstt, before_eq_tstt, if_list=None, importance=False):
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
    print('eval_sequence')
    for link_id in order_list:
        level += 1
        days_list.append(damaged_dict[link_id])
        net.link[link_id].add_link_back()
        added.append(link_id)
        not_fixed = set(to_visit).difference(set(added))
        net.not_fixed = set(not_fixed)
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

        tot_area += tstt * days_list[i]

    return tot_area

def main():
    print('bi-directional')
    damaged_dict = read_scenario(sname=SNAME)
    damaged_links = [link for link in damaged_dict.keys()]

    net_after, after_eq_tstt = state_after(damaged_links)
    net_before, before_eq_tstt = state_before(damaged_links)

    wb = worst_benefit(net_before, damaged_links, before_eq_tstt)
    bb = best_benefit(net_after, damaged_links, after_eq_tstt)

    # Create start and end node
    start_node = Node(tstt_after=after_eq_tstt,
                      net=net_after, damaged_dict=damaged_dict)
    start_node.realized = 0
    start_node.level = 0
    start_node.visited, start_node.not_visited = set([]), set(damaged_links)
    start_node.net.fixed, start_node.net.not_fixed = set([]), set(damaged_links)


    end_node = Node(tstt_before=before_eq_tstt,
                    net=net_before, damaged_dict=damaged_dict, forward=False)
    end_node.level = len(damaged_links)
    end_node.visited, end_node.not_visited = set([]), set(damaged_links)
    end_node.net.fixed, end_node.net.not_fixed = set(damaged_links), set([])

    ### Get a feasible solution using importance factor ###
    if not os.path.exists('saved_dictionaries/' + 'best_bound' + SNAME + '.pickle'):
        net_before.damaged_dict = damaged_dict
        tot_flow = 0
        if_net = deepcopy(net_before)
        for ij in if_net.link:
            tot_flow += if_net.link[ij].flow

        ffp = 1
        if_dict = {}
        for link_id in damaged_links:
            link_flow = if_net.link[link_id].flow
            if_dict[link_id] = link_flow / tot_flow
            ffp -= if_dict[link_id]

        ffp = ffp * 100

        sorted_d = sorted(if_dict.items(), key=lambda x: x[1])
        if_order, if_importance = zip(*sorted_d)
        if_order = if_order[::-1]
        if_importance = if_importance[::-1]
        print('if_order: ', if_order)
        print('if_importance: ', if_importance)

        best_bound = eval_sequence(
            if_net, if_order, after_eq_tstt, before_eq_tstt, if_dict, importance=True)
        save(best_bound, 'best_bound' + SNAME)
        print('best_bound: ', best_bound)
    else:
        best_bound = load('best_bound' + SNAME)

    # if not os.path.exists('saved_dictionaries/' + 'tap_counter' + SNAME + '.pickle'):
    #     save(TAP_COUNTER, 'tap_counter' + SNAME)
    # else:
    #     TAP_COUNTER = load('tap_counter' + SNAME)

    #### get optimal solution to check correctness
    if not os.path.exists('saved_dictionaries/' + 'min_seq' + SNAME + '.pickle'):
        print('finding the optimal sequence...')
        import itertools
        all_sequences = list(itertools.permutations(damaged_links))
        
        seq_dict = {}
        i = 0
        min_cost = 1000000000000000
        min_seq = None
        net_after.damaged_dict = damaged_dict
        for sequence in all_sequences:
            
            seq_net = deepcopy(net_after)
            cost = eval_sequence(seq_net, sequence, after_eq_tstt, before_eq_tstt)
            # seq_dict[sequence] = cost

            if cost < min_cost:
              min_cost = cost
              min_seq = sequence

            i += 1

            print(i)
            print('min cost found so far: ', min_cost)
            print('min seq found so far: ', min_seq)
        print('optimal sequence: ', min_seq)
        # import operator
        # sorted_x = sorted(seq_dict.items(), key=operator.itemgetter(1))
        save(min_seq, 'min_seq' + SNAME)
    else:
        min_seq = load('min_seq' + SNAME)

    path = search(damaged_dict, wb, bb, start_node, end_node, best_bound)
    print("sequence found: ", path)
    pdb.set_trace()

if __name__ == '__main__':
    main()
