import json
import re
import sys
import matplotlib.pyplot as plt

files = ['wgpu-scale-out-scale-for.json', 'scale-out-nvidia-wgpu.json', 'scale-out-scale-up-parallel-xor.json', 'wgpu-uber-shader-atomic-add-identify-parallel-xor.json']
labels = ['scale-out for', 'scale-out', 'scale-out p xor', 'uber parallel']
#files = [#'wgpu-expand-1.json',
#         #'wgpu-expand-2.json',
#         'wgpu-expand-4.json',
#         'wgpu-expand-8.json',
#         'wgpu-expand-16.json',
#         'wgpu-expand-32.json',
#         'wgpu-expand-64.json',
#         'wgpu-expand-128.json',
#         'wgpu-expand-256.json',
#         'wgpu-expand-512.json',
#         'wgpu-expand-1024.json']
scales = {}
for x in files:
    print("data/results/" + x)
    with open("data/results/" + x, 'r') as file:
        data = json.load(file)
    for test in data['benchmarks']:
        match = re.search('Scale(.*)Pairs(.*)', test['name'])
        scale = match.group(1)
        pairs = match.group(2)
        if not scale in scales:
            scales[scale] = {};
        if not x in scales[scale]:
            scales[scale][x] = { "xpoints": [], 'ypoints': []}
        scales[scale][x]['xpoints'].append(int(pairs))
        scales[scale][x]['ypoints'].append(float(test['real_time']))

print(scales)
fig, axs = plt.subplots(2, 2, constrained_layout=True)
xindex = 0
yindex = 0
for scale in scales:
    result = scales[scale]
    axs[xindex, yindex].set_xlabel("pairs")
    axs[xindex, yindex].set_ylabel("execution time in seconds")
    axs[xindex, yindex].set_title("Scale factor " + scale)
    axs[xindex, yindex].set_xscale('log', base=2)

    for x, file in enumerate(result):
        axs[xindex, yindex].scatter(result[file]["xpoints"], result[file]["ypoints"], label=labels[x], s=1);
    xindex += 1
    if xindex > 1:
        xindex = 0
        yindex += 1
plt.subplots_adjust(right=0.65)
plt.legend(bbox_to_anchor=(0.65, 0, 1, 1))
plt.savefig("output.pdf")
