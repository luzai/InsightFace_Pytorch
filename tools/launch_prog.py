#!/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


from __future__ import print_function
import os, sys
import subprocess

# if len(sys.argv) != 4:
#   print("usage: %s <prog>" % sys.argv[0])
#   sys.exit(1)

host_file = "./hosts"
user = "xinglu"
# prog_name = sys.argv[1]
my_env = os.environ.copy()
export = ""
ks = ["PATH", "LD_LIBRARY_PATH"]
for k, v in my_env.items():
    if k in ks:
        export += f"export {k}={v}&&"
kill_cmd = (
         export + 'cd ~/prj/InsightFace_Pytorch&&'
                  'CUDA_VISIBLE_DEVICES=0,1,2,3 '
                  'python -m torch.distributed.launch '
                  '--nproc_per_node=3 '
                  '--nnodes=2 '
                  '--node_rank=%d '
                  '--master_addr="10.13.72.84" --master_port=12345 ./train.py'
)
print(kill_cmd)

# Kill program on remote machines
threads = []
from threading import Thread

with open(host_file, "r") as f:
    for ind, host in enumerate(f):
        if ':' in host:
            host = host[:host.index(':')]
        if '84' in host: host = "amax84"
        if '85' in host: host = "amax85"
        print(host)
        thread = Thread(
            target=(lambda: subprocess.check_call(["ssh", "-oStrictHostKeyChecking=no", "%s" % host, kill_cmd % ind],
                                                  # target=(lambda: subprocess.check_call(["ssh", "-oStrictHostKeyChecking=no", "%s" % host, "cd ~/prj/InsightFace_Pytorch/tools&&sh test.sh"],
            
                                                  )), args=())
        thread.setDaemon(True)
        thread.start()
        threads.append(thread)
        # subprocess.check_call(["ssh", "-oStrictHostKeyChecking=no", "%s" % host, kill_cmd], )
for thread in threads:
    while thread.isAlive():
        thread.join(100)
    # Kill program on local machine
# os.system(kill_cmd)
