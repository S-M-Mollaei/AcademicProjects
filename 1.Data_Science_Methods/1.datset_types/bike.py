# -*- coding: utf-8 -*-

import json
import sys
from math import cos, acos, sin

def distance_coords(lat1, lng1, lat2, lng2):
    """Compute the distance among two points."""
    deg2rad = lambda x: x * 3.141592 / 180
    lat1, lng1, lat2, lng2 = map(deg2rad, [ lat1, lng1, lat2, lng2 ])
    R = 6378100 # Radius of the Earth, in meters
    return R * acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lng1 - lng2))

with open("bike.json") as f:
    obj = json.load(f)

count_act = 0
count_bike = 0
count_slot = 0

for x in obj['network']['stations']:
    if x['extra']['status'] == 'online':
        count_act += 1
    count_bike += x['free_bikes']
    count_slot += x['empty_slots']
    
    
print(f'num of active stations is {count_act}')
print(f'num of free bikes is {count_bike}')
print(f'num of emty slots is {count_slot}')

latitude, longitude = input('enter coordinates: ').split()

min = sys.maxsize
for x in obj['network']['stations']:
    dis = distance_coords(float(latitude), float(longitude), x['latitude'], x['longitude'])
    if dis < min:
        min = dis
        target = x['name']

print('colset on is',target)
    

