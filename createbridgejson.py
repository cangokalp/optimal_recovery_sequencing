import pandas as pd
import json

scenario = pd.read_csv('Scenario_1.csv')



bridgejson = {}
bridgejson['features'] = []




for index,row in scenario.iterrows():

	lon = row['Longitude']
	lat = row['Latitude']
	bridgejson['features'].append({
    'type': 'Feature',
    'properties': {},
	'geometry': {
    'type': 'Point',
	'coordinates': [lon, lat],
	}
	})


with open('bridge.json', 'w') as outfile:
    json.dump(bridgejson, outfile)