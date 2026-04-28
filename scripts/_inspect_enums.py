import urllib.request, json, pprint
d = json.load(urllib.request.urlopen('http://127.0.0.1:8189/object_info'))
names = [
    'NukeMax_AudioToFloatCurve', 'NukeMax_AudioDriveMask',
    'NukeMax_ComputeOpticalFlow', 'NukeMax_MaterialDecomposerModels',
    'NukeMax_FlowBackwardWarp', 'NukeMax_RotoShapeToDiffusionGuidance',
    'NukeMax_LightRigBuilder', 'NukeMax_ThreePointRelight',
    'NukeMax_RotoSplineEditor',
]
for n in names:
    print('---', n)
    pprint.pp(d[n]['input']['required'])
