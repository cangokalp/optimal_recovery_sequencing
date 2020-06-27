import pandas as pd
import json
import numpy as np
import pdb
import math

def distance(origin, destination):
    lon1, lat1 = origin
    lon2, lat2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

def get_broken_links(JSONFILE, scenario_file):

	scenario = pd.read_csv(scenario_file)


	with open(JSONFILE) as f:
	  netgeo = json.load(f)

	correspondence = {}
	for index,row in scenario.iterrows():
		state = row['State']
		if state == 'No Damage':
			continue
		fac_id = row['FacilityID'] 
		lon = row['Longitude']
		lat = row['Latitude']
		r_time = row['RepairTime']
		dest = (lon, lat)
		min_dist = np.inf
		star_dict = None
		for adict in netgeo['features']:
			dist = distance(adict['geometry']['coordinates'][0], dest)
			dist = min(distance(adict['geometry']['coordinates'][1], dest), dist)
			if dist < min_dist:
				min_dist = dist
				star_dict = adict
		
		if min_dist > 0.3:
			continue
		st = star_dict['properties']['init_node']
		tm = star_dict['properties']['term_node']
		cur_key = (st,tm)

		if cur_key not in correspondence:
			correspondence[cur_key] = (fac_id, min_dist, r_time)
		else:
			_, old_dist, _ = correspondence[cur_key]
			if min_dist < old_dist:
				correspondence[cur_key] = (fac_id, min_dist, r_time)


	damaged_dict = {}

	for k,v in correspondence.items():
		link_name = '(' + str(k[0]) +',' + str(k[1]) + ')'
		damaged_dict[link_name] = v[2]

	# scenario['exists_in_netfile'] = 0
	# scenario.loc[scenario['FacilityID'].isin(all_facilities), ['exists_in_netfile']] = 1
	# scenario.to_csv('Scenario_1_edited.csv')


	## export new_json

	# bridgejson = {}
	# bridgejson['features'] = []


	# all_facilities = []
	# for k,v in correspondence.items():
	# 	for index,row in scenario.iterrows():
	# 		fac_id = row['FacilityID']
	# 		if fac_id != v[0]:
	# 			continue
	# 		all_facilities.append(fac_id)
	# 		lon = row['Longitude']
	# 		lat = row['Latitude']
	# 		bridgejson['features'].append({
	# 	    'type': 'Feature',
	# 	    'properties': {},
	# 		'geometry': {
	# 	    'type': 'Point',
	# 		'coordinates': [lon, lat],
	# 		}
	# 		})


	# with open('bridge_new.json', 'w') as outfile:
	#     json.dump(bridgejson, outfile)

	return damaged_dict

