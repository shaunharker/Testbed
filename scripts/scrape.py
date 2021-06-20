import json, time
with open("2021-06-19-TestBed.ipynb", 'r') as infile:
    data = json.load(infile)

with open('scrape'+str(int(time.time()))+'.py', 'w') as outfile:
    outfile.write(''.join([ ''.join(x['source'])+'\n' for x in data['cells'] if x['cell_type'] == 'code']))
