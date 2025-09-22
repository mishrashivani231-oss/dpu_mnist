
import numpy as np
from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse


# In[ ]:


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


import numpy as np
from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse


# In[ ]:


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


# In[ ]:


def runDPU(id,start,dpu,img):
    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    # we can avoid output scaling if use argmax instead of softmax
    #output_fixpos = outputTensors[0].get_attr("fix_point")
    #output_scale = 1 / (2**output_fixpos)

    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    write_index = start
    ids=[]
    ids_max = 50
    outputData = []
    for i in range(ids_max):
        outputData.append([np.empty(output_ndim, dtype=np.int8, order="C")])
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count

        '''prepare batch input/output '''
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]

        '''init input image to input buffer '''
        '''run with batch '''
        job_id = dpu.execute_async(inputData,outputData[len(ids)])
        ids.append((job_id,runSize,start+count))
        count = count + runSize
        if count<n_of_images:
            if len(ids) < ids_max-1:
                continue
        ids=[]


# In[ ]:


divider = '------------------------------------'

runTotal = 100
threads = 2

model= "deploy.xmodel"
out_q = [None] * runTotal
g = xir.Graph.deserialize(model)
subgraphs = get_child_subgraph_dpu(g)
all_dpu_runners = []

for i in range(threads):
    all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
input_scale = 2**input_fixpos

image = np.random.rand(28,28,1)
image = image * input_scale
image = image.astype(np.int8)

img = []
for i in range(runTotal):
    img.append(image[i])

threadAll = []
start=0
for i in range(threads):
    if (i==threads-1):
        end = len(img)
    else:
        end = start+(len(img)//threads)
    in_q = img[start:end]
    t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q))
    threadAll.append(t1)
    start=end

time1 = time.time()
for x in threadAll:
    x.start()
for x in threadAll:
    x.join()
time2 = time.time()
timetotal = time2 - time1

fps = float(runTotal / timetotal)
print (divider)
print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, runTotal, timetotal))
                                                                                                                                                                         
