from test_add_remove import *


class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, level=None, set_visited=None):
        self.parent = parent
        self.set_visited = set_visited
        self.level = level

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def test_func(self):
    	return None


def get_successors(node):
	"""given a state, returns list of bridges that has not been fixed yet"""
	not_visited = to_visit.difference(node.set_visited)
	return not_visited


def astar(damage_dict, wb, bb):
    """Returns the best order to visit the set of nodes"""

	damaged_links = damage_dict.keys()
	repair_days = damage_dict.values()
	to_visit = set(damaged_links)

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Create start and end node
    start_node = Node(level=0, set_visited=set([]))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(level=len(to_visit) + 1,
                    set_visited=set(to_visit).union('end'))
    end_node.g = end_node.h = end_node.f = 0
    

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path

        # Generate children
        eligible_expansions = get_successors(current_node)
        children = []

        for a_link in eligible_expansions:

            # Create new node
            new_node = Node(parent=current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) **
                       2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)


def main():

	sname = 'Moderate_5'
	damage_dict = read_scenario(sname=sname)

	wb = load('worst_benefit_dict' + sname)
	bb = load('best_benefit_dict' + sname)

    astar(damage_dict, wb, bb)


if __name__ == '__main__':
    main()



