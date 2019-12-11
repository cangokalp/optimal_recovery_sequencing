from network import *
import numpy as np
import cProfile
import pstats
import pandas as pd
import pickle
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
from tabulate import tabulate
import os.path

solve_count = 0
class Seq:

	def __init__(self, link_id=None, parent=None, net=None, tstt_after=None, tstt_before=None, level=None, damaged_dict=None):
		self.link_id = link_id
		self.parent = parent
		self.fathomed = False
		self.path = []
		self.days_past = 0
		self.net = net
		self.days = damage_dict[link_id]
		self.tstt_before = tstt_before
		self.tstt_after = tstt_after
		self.benefit = tstt_after - tstt_before
		self.level = level
		self.damage_dict = damage_dict
		self.assignChar()

	def assignChar(self):
		if self.parent != None:
			prev_path = deepcopy(self.parent.path)
			prev_path.append(self.link_id)
			self.path = prev_path
			self.path_set = set(self.path)
			self.days_past = self.parent.days_past + self.days
			self.realized = self.parent.realized + self.tstt_before*self.days

		else:
			self.path = [self.link_id]
			self.realized = self.tstt_before*self.days
			self.days_past = self.days

	def setBounds(self, wb, bb):
		remaining = list(set(damaged_links) - set(self.path))
		self.ub = self.realized
		self.lb = self.realized

		ordered_days = []
		orderedw_benefits = []
		orderedb_benefits = []

		sorted_d = sorted(damage_dict.items(), key=lambda x: x[1])

		for key, value in sorted_d:
		    # print("%s: %s" % (key, value))
		    if key in remaining:
		    	ordered_days.append(value)
		    	orderedw_benefits.append(wb[key])
		    	orderedb_benefits.append(bb[key])

		bang4buck_b = np.array(orderedb_benefits)/np.array(ordered_days)
		bang4buck_w = np.array(orderedw_benefits)/np.array(ordered_days)

		days_b = [x for _,x in sorted(zip(bang4buck_b, ordered_days))]
		b = [x for _,x in sorted(zip(bang4buck_b, orderedb_benefits))]
		
		days_w = [x for _,x in sorted(zip(bang4buck_w, ordered_days))]
		w = [x for _,x in sorted(zip(bang4buck_w, orderedw_benefits))]

		##b
		for i in range(len(days_b)):
			if i==0:
				b_tstt = self.tstt_after
			else:
				b_tstt = b_tstt - b[i-1]

			self.lb += b_tstt*days_b[i]

		##w
		for i in range(len(days_w)):
			if i==0:
				w_tstt = self.tstt_after
			else:
				w_tstt = w_tstt - w[i-1]

			self.ub += w_tstt*days_w[i]



def solve_UE(net=None):
	net.userEquilibrium("FW", 1e4, 1e-3, net.averageExcessCost)


def read_scenario(fname='ScenarioAnalysis.xlsx', sname='Moderate_1'):
	scenario_pd = pd.read_excel(fname, sname)
	dlinks = scenario_pd[scenario_pd['Link Condition'] == 1]['Link'].tolist()
	cdays = scenario_pd[scenario_pd['Link Condition'] == 1][
		'Closure day (day)'].tolist()

	damage_dict = {}
	for i in range(len(dlinks)):
		damage_dict[dlinks[i]] = cdays[i]
	return damage_dict


def save(dict_to_save, fname):
	fname = 'saved_dictionaries/' + fname + '.pickle'
	with open(fname, 'wb') as f:
		pickle.dump(dict_to_save, f)


def load(fname):
	fname = 'saved_dictionaries/' + fname + '.pickle'
	with open(fname, 'rb') as f:
		item = pickle.load(f)
	return item


def find_tstt(net=None):
	tx = 0
	for ij in net.link:
		tx += net.link[ij].cost * net.link[ij].flow
	return tx


def fix_one(net1, fixed_state, to_fix, tstt_state, days_state):
	net = deepcopy(net1)
	duration = damage_dict[to_fix]
	net.link[to_fix].add_link_back()
	solve_UE(net=net)
	tstt_after = find_tstt(net=net)
	fixed_state.append(to_fix)
	tstt_state.append(tstt_after)
	days_state.append(duration)

	return net, fixed_state, tstt_state, days_state


def eval_sequence(net, order_list, after_eq_tstt, before_eq_tstt, if_list=None, importance=False):
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

	T = 0
	for link_id in order_list:
		T += damage_dict[link_id]

	level = 0
	prev_linkid = None
	tstt_before = after_eq_tstt
	for link_id in order_list:
		level += 1
		days_list.append(damage_dict[link_id])
		net.link[link_id].add_link_back()
		solve_UE(net=net)
		tstt_after = find_tstt(net=net)
		tstt_list.append(tstt_after)

		seq = Seq(link_id=link_id, parent=prev_linkid, net=deepcopy(net), tstt_after=tstt_after,
				  tstt_before=tstt_before, level=level, days=damage_dict[link_id], T=T)
		seq_list.append(seq)
		prev_linkid = seq
		tstt_before = tstt_after

		if importance:
			curfp += if_list[link_id]
			fp.append(curfp * 100)

	tot_area = 0
	for i in range(len(days_list)):
		if i == 0:
			tot_area += days_list[i] * (after_eq_tstt - before_eq_tstt)
		else:
			tot_area += days_list[i] * (tstt_list[i - 1] - before_eq_tstt)
	return tot_area, seq_list

def find_max_relief(links_eligible_addback, fixed=[], cur_tstt=None):
	days_list = []
	tstt_list = []
	relief_list = []
	tot_day = sum(damage_dict.values())
	for link_id in fixed:
		tot_day -= damage_dict[link_id]

	for link_id in links_eligible_addback:
		wouldaDay = damage_dict[link_id]
		days_list.append(wouldaDay)
		mynet.link[link_id].add_link_back()
		solve_UE(net=mynet)
		wouldaTSTT = find_tstt(net=mynet)
		tstt_list.append(wouldaTSTT)

		# relief_list.append((tot_day - wouldaDay)*(cur_tstt - wouldaTSTT))
		relief_list.append((tot_day - wouldaDay) * (cur_tstt - wouldaTSTT))

		mynet.link[link_id].remove()
	# ind = tstt_list.index(min(tstt_list))
	# ind = relief_list.index(max(relief_list))
	ind = relief_list.index(min(relief_list))

	return links_eligible_addback[ind], tstt_list[ind], days_list[ind]

def graph_current(updated_net, fixed_state, tstt_state, days_state, before_eq_tstt, after_eq_tstt, N):

	cur_tstt = after_eq_tstt

	order_list = fixed_state
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

	if days != N:
		tot_area += N - days * (tstt_state[-1] - before_eq_tstt)
		tstt_g.append(tstt_state[-1])
		days_state = days_state + [N - days]

	x = [0] + days_state
	y = [after_eq_tstt] + tstt_g + [tstt_g[-1]]

	for j in range(len(days_state)):
		x[j] = days_state[j]
		if j != 0:
			x[j] = sum(days_state[:j])

	x[0] = 0
	x[j + 1] = sum(days_state[:j + 1])
	x.append(x[-1])

	# x = days_list
	# plt.subplot(211)
	plt.figure()
	plt.fill_between(x, y, before_eq_tstt, step="pre",
					 color='green', alpha=0.4)
	plt.step(x, y, label='tstt')
	plt.xlabel('Days')
	plt.ylabel('TSTT')

	tt = 'Total Area: ' + str(tot_area)
	xy = (0.2, 0.2)
	plt.annotate(tt, xy, xycoords='figure fraction')
	plt.legend()
	plt.savefig(sname + '_MRF_TSTT')
	plt.show()

def prune_by_visited(seq_list):
	path_set_list = []
	removal_list = []
	for seq in seq_list:
		path_set = seq.path_set
		if path_set not in path_set_list:
			path_set_list.append(path_set)
		else:
			seq2 = seq_list[path_set_list.index(path_set)]
			if seq.realized < seq2.realized:
				removal_list.append(seq2)
			elif seq2.realized < seq.realized:
				removal_list.append(seq)

	seq_list = list(set(seq_list) - set(removal_list))

	return seq_list

def prune_by_bounds(seq_list):
	path_set_list = []
	removal_list = []

	for seq in seq_list:
		for seq2 in seq_list:
			if seq != seq2:			
				if seq.ub < seq2.lb :
					removal_list.append(seq2)
				elif seq2.ub < seq.lb :
					removal_list.append(seq)

	seq_list = list(set(seq_list) - set(removal_list))

	return seq_list

def prune(seq_list):
	seq_list = prune_by_visited(seq_list)
	seq_list = prune_by_bounds(seq_list)
	return seq_list


def expand_seq(seq, lad, level):
	tstt_before = seq.tstt_after
	net = deepcopy(seq.net)
	net.link[lad].add_link_back()
	solve_UE(net=net)
	tstt_after = find_tstt(net=net)
	seq = Seq(link_id=lad, parent=seq, net=net, tstt_after=tstt_after, tstt_before=tstt_before, level=level, damaged_dict=damage_dict)
	seq.setBounds(wb, bb)
	return seq

sname = 'Moderate_5'
# for sname in snames:
mydict = {}
damage_dict = read_scenario(sname=sname)

damaged_links = damage_dict.keys()
alldays = damage_dict.values()

N = 0
for i in alldays:
	N += i
links_to_remove = damaged_links

# Find before earthquake equilibrium
# before = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
# solve_UE(net=before)
# before_eq_tstt = find_tstt(before)

# Find after earthquake equilibrium
# after = Network("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp")
# for link in links_to_remove:
# 	after.link[link].remove()
# solve_UE(net=after)
# after_eq_tstt = find_tstt(after)

###### WORSE BENEFIT ANALYSIS #######

# if analysis haven't been done yet
if not os.path.exists('saved_dictionaries/' + 'worst_benefit_dict' + sname + '.pickle'):

	# for each bridge, find the effect on TSTT when that bridge is removed
	# while keeping others
	wb = {}
	for link in links_to_remove:
		test_net = deepcopy(before)
		test_net.link[link].remove()
		solve_UE(net=test_net)
		wb[link] = find_tstt(test_net) - before_eq_tstt
	# save dictionary
	save(wb, 'worst_benefit_dict' + sname)

else:
	wb = load('worst_benefit_dict' + sname)

###### BEST BENEFIT ANALYSIS #######
seq_list = []
# if analysis haven't been done yet:
if not os.path.exists('saved_dictionaries/' + 'best_benefit_dict' + sname + '.pickle'):

	# for each bridge, find the effect on TSTT when that bridge is removed
	# while keeping others
	bb = {}

	for link in links_to_remove:
		test_net = deepcopy(after)
		test_net.link[link].add_link_back()
		solve_UE(net=test_net)
		tstt_after = find_tstt(test_net)

		seq_list.append(Seq(link_id=link, parent=None, net=test_net, tstt_after=tstt_after, tstt_before=after_eq_tstt, level=1, damaged_dict=damage_dict))
		bb[link] = after_eq_tstt - tstt_after
	save(bb, 'best_benefit_dict' + sname)
	save(seq_list, 'seq_list' + sname)


else:
	bb = load('best_benefit_dict' + sname)
	seq_list = load('seq_list' + sname)
###### FIND PRECEDENCE RELATIONSHIPS ######
if not os.path.exists('saved_dictionaries/' + 'precedence_dict' + sname + '.pickle'):

	precedence = {} #if 1: 3,4 means 1 has to come before 3 and also 4
	following = {} #if 3: 1,2 means 3 has to come after 1 and also 2

	for a_link in links_to_remove:
		for other in links_to_remove:
			if a_link != other:
				if wb[a_link] * damage_dict[other] - bb[other] * damage_dict[a_link] > 0:
					if a_link in precedence.keys():
						precedence[a_link].append(other)
					else:
						precedence[a_link] = [other]
					if other in following.keys():
						following[other].append(a_link)
					else:
						following[other] = [a_link]

	save(precedence, 'precedence_dict'+ sname)
	save(following, 'following_dict'+ sname)
else:
	precedence = load('precedence_dict'+ sname)
	following = load('following_dict'+ sname)

# start sequences
## first pruning by precedence, if 2 has to follow something you cannot start with 2
fathomed_seqlist = []
for seq in seq_list:
	if seq.link_id in following.keys():
		fathomed_seqlist.append(seq)

#TODO 
# implemennt 2 pair elimination 1-2 vs 2-1 without solving
# implement that if 3-2-1 solved 3-1-2 you can get the solution from the other

alive_list = list(set(seq_list) - set(fathomed_seqlist))
### expand sequences here

level = 1
print('max_length: ', len(damaged_links))
tot_solved = 2*len(damaged_links) + 2
while level < len(damaged_links):
	level += 1
	new_seqs = []
	for aseq in alive_list:
		possible_additions = list(set(damaged_links)-set(aseq.path))
		following_keys = [i for i in following.keys()]
		for j in possible_additions:
			if j in following_keys:
				follows = following[j]
				if len(set(follows).difference(set(aseq.path))) == 0:
					seq = expand_seq(seq=aseq, lad=j, level=level)
					new_seqs.append(seq)
			else:
				seq = expand_seq(seq=aseq, lad=j, level=level)
				new_seqs.append(seq)

	seq_list = new_seqs	
	print('-------')
	print(level)
	counting = 0
	for i in seq_list:
		counting +=1
		print('level ' + str(level) + ', seq ' + str(counting) + ':' + '\n')
		print(i.path)

	tot_solved += counting
	print('length of sequence list before pruning: ', len(seq_list))
	alive_list = prune(seq_list)
	print('pruning completed')

	counting = 0
	for i in alive_list:
		counting +=1
		print('alive in level ' + str(level) + ', seq ' + str(counting) + ':' + '\n')
		print(i.path)
	print('length of alive sequences: ', len(alive_list))


pdb.set_trace()
print('# TAP solved: ', tot_solved)
print(alive_list[0].path)
save(alive_list[0], 'dp_soln' + sname)


##### Compare it to importance factor

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

	# print(tabulate(sorted_x, headers=['Sequence','Total_Cost']))
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
