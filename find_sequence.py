from test_add_remove import *

SNAME = 'Moderate_5'
NETFILE = "SiouxFalls/SiouxFalls_net.tntp"
TRIPFILE = "SiouxFalls/SiouxFalls_trips.tntp"


class Node():
    """A node class for bi-directional search for pathfinding"""

    def __init__(self, visited=None, link_id=None, parent=None, net=None, tstt_after=None, tstt_before=None, level=None, damaged_dict=None):
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

        else:
            if self.link_id is not None:
                self.path = [self.link_id]
                self.realized = self.tstt_before * self.days
                self.days_past = self.days

            else:
                self.realized = 0
                self.days = 0
                self.days_past = self.days

    def __eq__(self, other):
        return self.visited == other.visited


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


def get_successors_f(node, wb, bb, to_visit):
    """given a state, returns list of bridges that has not been fixed yet"""
    not_visited = to_visit.difference(node.visited)
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

def get_successors_b(node, wb, bb, to_visit):
    """given a state, returns list of bridges that has not been removed yet"""
    not_visited = node.visited.intersection(to_visit)
    successors = []


    if node.level != len(node.damaged_dict.keys()) + 1:
        tail = node.path[-1]
        for a_link in not_visited:
            if wb[a_link] * node.damaged_dict[tail] - bb[tail] * node.damaged_dict[a_link] > 0:
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
    solve_UE(net=net)
    tstt_after = find_tstt(net=net)
    node = Node(link_id=a_link, parent=node, net=net, tstt_after=tstt_after,
                tstt_before=tstt_before, level=level, damaged_dict=damaged_dict)
    return node

def expand_sequence_b(node, a_link, level, damaged_dict):
    """given a link and a node, it expands the sequence"""
    tstt_after = node.tstt_before
    net = deepcopy(node.net)
    net_after.link[a_link].remove()
    solve_UE(net=net)
    tstt_before = find_tstt(net=net)
    node = Node(link_id=a_link, parent=node, net=net, tstt_after=tstt_after,
                tstt_before=tstt_before, level=level, damaged_dict=damaged_dict)
    return node


def expand_forward(damaged_dict, wb, bb, start_node, end_node, best_ub, open_list_b, open_list_f, closed_list_b, closed_list_f):
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

    # Found the goal
    if current_node == end_node:
        return current_node.path

    if current_node.f > best_ub:
        continue



    # Generate children
    eligible_expansions = get_successors_f(current_node, wb, bb, to_visit)
    
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

        remaining = get_successors_f(child, wb, bb, to_visit)
        # set upper and lower bounds
        set_bounds_f(child, wb, bb, remaining)
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

    return open_list_f, closed_list_f

def expand_backward(damaged_dict, wb, bb, start_node, end_node, best_ub, open_list_b, open_list_f, closed_list_b, closed_list_f):
    
    # Loop until you find the end
    print('backwards search is in progress...')
    tap_solved = 0
    
    # Get the current node
    current_node = open_list_b[0]
    current_index = 0
    for index, item in enumerate(open_list_b):
        if item.f < current_node.f:
            current_node = item
            current_index = index

    # Pop current off open list, add to closed list
    open_list_b.pop(current_index)
    closed_list_b.append(current_node)

    # Found the goal
    if current_node == start_node:
        return current_node.path

    if current_node.f > best_ub:
        continue

    # Generate children
    eligible_expansions = get_successors_b(current_node, wb, bb, to_visit)
    
    children = []

    for a_link in eligible_expansions:

        # Create new node
        new_node = expand_sequence_b(
            current_node, a_link, level=current_node.level -1 , damaged_dict=damaged_dict)
        tap_solved += 1
        # Append
        children.append(new_node)

    print('open_list_b length: ', len(open_list_b))
    print('number of taps solved: ', tap_solved)
    # Loop through children
    for child in children:

        remaining = get_successors_b(child, wb, bb, to_visit)
        # set upper and lower bounds
        set_bounds_f(child, wb, bb, remaining)
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
    return open_list_b, closed_list_b

def search(damaged_dict, wb, bb, start_node, end_node):
    """Returns the best order to visit the set of nodes"""

    damaged_links = damaged_dict.keys()
    repair_days = damaged_dict.values()
    to_visit = set(damaged_links)

    # Initialize both open and closed list for forward and backward directions
    open_list_f = []
    closed_list_f = []
    # Add the start node
    best_ub = np.inf
    open_list_f.append(start_node)

    open_list_b = []
    closed_list_b = []
    open_list_b.append(end_node)

    open_list_f, closed_list_f = expand_forward(damaged_dict, wb, bb, start_node, end_node, best_ub, open_list_b, open_list_f, closed_list_b, closed_list_f, to_visit)
    open_list_b, closed_list_b = expand_backward(damaged_dict, wb, bb, start_node, end_node, best_ub, open_list_b, open_list_f, closed_list_b, closed_list_f, to_visit)


def create_network(netfile=None, tripfile=None):
    return Network(netfile, tripfile)

def worst_benefit(before, links_to_remove, before_eq_tstt):
    if not os.path.exists('saved_dictionaries/' + 'worst_benefit_dict' + SNAME + '.pickle'):

        # for each bridge, find the effect on TSTT when that bridge is removed
        # while keeping others
        wb = {}
        for link in links_to_remove:
            test_net = deepcopy(before)
            test_net.link[link].remove()
            solve_UE(net=test_net)
            wb[link] = find_tstt(test_net) - before_eq_tstt
        # save dictionary
        save(wb, 'worst_benefit_dict' + SNAME)

    else:
        wb = load('worst_benefit_dict' + SNAME)

    return wb

def best_benefit(after, links_to_remove, after_eq_tstt):
    if not os.path.exists('saved_dictionaries/' + 'best_benefit_dict' + SNAME + '.pickle'):

        # for each bridge, find the effect on TSTT when that bridge is removed
        # while keeping others
        bb = {}

        for link in links_to_remove:
            test_net = deepcopy(after)
            test_net.link[link].add_link_back()
            solve_UE(net=test_net)
            tstt_after = find_tstt(test_net)

            seq_list.append(Node(link_id=link, parent=None, net=test_net, tstt_after=tstt_after,
                                tstt_before=after_eq_tstt, level=1, damaged_dict=damaged_dict))
            bb[link] = after_eq_tstt - tstt_after
        save(bb, 'best_benefit_dict' + SNAME)
        save(seq_list, 'seq_list' + SNAME)

    else:
        bb = load('best_benefit_dict' + SNAME)
        # seq_list = load('seq_list' + SNAME)

    return bb

def state_after(damaged_links):
    if not os.path.exists('saved_dictionaries/' + 'net_after' + SNAME + '.pickle'):

        net_after = create_network(NETFILE, TRIPFILE)
        for link in damaged_links:
            net_after.link[link].remove()
        solve_UE(net=net_after)
        after_eq_tstt = find_tstt(net=net_after)
        
        save(net_after, 'net_after' + SNAME)
        save(after_eq_tstt, 'tstt_after' + SNAME)

    else:
        net_after = load('net_after' + SNAME)
        after_eq_tstt = load('tstt_after' + SNAME)

    return net_after, after_eq_tstt

def state_before(damaged_links):
    
    if not os.path.exists('saved_dictionaries/' + 'net_before' + SNAME + '.pickle'):
        net_before = create_network(NETFILE, TRIPFILE)
        solve_UE(net=net_before)
        before_eq_tstt = find_tstt(net=net_before)
        
        save(net_before, 'net_before' + SNAME)
        save(before_eq_tstt, 'tstt_before' + SNAME)

    else:
        net_before = load('net_before' + SNAME)
        before_eq_tstt = load('tstt_before' + SNAME)

    return net_before, before_eq_tstt

def main():

    damaged_dict = read_scenario(sname=SNAME)
    damaged_links = [link for link in damaged_dict.keys()]

    net_after, after_eq_tstt = state_after(damaged_links)
    net_before, before_eq_tstt = state_before(damaged_links)

    wb = worst_benefit(net_before, damaged_links, before_eq_tstt)
    bb = best_benefit(net_after, damaged_links, after_eq_tstt)

    # Create start and end node
    start_node = Node(tstt_after=after_eq_tstt, net=net_after)
    start_node.realized = 0
    start_node.level = 0
    start_node.visited = set([])

    end_node = Node(tstt_before=before_eq_tstt, net=net_before)
    end_node.level = len(damaged_links) + 1
    end_node.visited = set(damaged_links).union(['end'])

    search(damaged_dict, wb, bb, start_node, end_node)

if __name__ == '__main__':
    main()
