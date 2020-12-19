import pandas as pd
import json

scenario = pd.read_csv('Scenario_1.csv')






with open('bridge.json', 'w') as outfile:
    json.dump(bridgejson, outfile)