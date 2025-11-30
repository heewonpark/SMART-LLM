import os
from pathlib import Path
import subprocess
import argparse
import re
import shutil

def append_trans_ctr(allocated_plan):
    brk_ctr = 0
    code_segs = allocated_plan.split("\n\n")
    fn_calls = []
    for cd in code_segs:
        if "def" not in cd and "threading.Thread" not in cd and "join" not in cd and cd[-1] == ")":
            # fn_calls.append(cd)
            brk_ctr += 1
    print ("No Breaks: ", brk_ctr)
    return brk_ctr

LOG_PATTERN = r"(### LOG - START ###)(.*?)(### LOG - END ###)"
TASK_PATTERN = r"(### TASK - START ###)(.*?)(### TASK - END ###)"

def insert_between_markers(text, insert_text, pattern=r"(### LOG - START ###)(.*?)(### LOG - END ###)"):
    replacement = r"\1\n" + insert_text + r"\n\3"
    return re.sub(pattern, replacement, text, flags=re.DOTALL)

def compile_aithor_exec_file(expt_name):
    log_path = os.getcwd() + "/logs/" + expt_name
    executable_plan = ""
    
    # append the imports to the file
    import_file = Path(os.getcwd() + "/data/aithor_connect_v2/aithor_connect.py").read_text()
    executable_plan += (import_file + "\n")
    
    log_plan = ""
    # append the list of robots and floor plan number
    log_file = open(log_path + "/log.txt")
    log_data = log_file.readlines()
    # append the robot list
    log_plan += (log_data[8] + "\n")
    # append the floor number
    flr_no = log_data[4][12:]
    gt = log_data[9]
    log_plan += ("floor_no = " + flr_no + "\n\n")
    log_plan += (gt)
    trans = log_data[10][8:]
    log_plan += ("no_trans_gt = " + trans)
    max_trans = log_data[11][12:]
    log_plan += ("max_trans = " + max_trans + "\n")
    set_agents = log_data[12][13:]
    log_plan += ("set_agents = " + set_agents + "\n")
    set_objects = log_data[13][14:]
    log_plan += ("set_objects = " + set_objects + "\n")  
    executable_plan = insert_between_markers(executable_plan, log_plan, LOG_PATTERN) + "\n"

    # # append the ai thoe connector and helper fns
    # connector_file = Path(os.getcwd() + "/data/aithor_connect/aithor_connect.py").read_text()
    # executable_plan += (connector_file + "\n")
    
    # append the allocated plan
    allocated_plan = Path(log_path + "/code_plan.py").read_text()
    brks = append_trans_ctr(allocated_plan)
    #executable_plan += (allocated_plan + "\n")
    allocated_plan += ("\nno_trans = " + str(brks) + "\n")
    executable_plan = insert_between_markers(executable_plan, allocated_plan, TASK_PATTERN) + "\n"
    
    # # append the task thread termination
    # terminate_plan = Path(os.getcwd() + "/data/aithor_connect/end_thread.py").read_text()
    # executable_plan += (terminate_plan + "\n")

    with open(f"{log_path}/executable_plan.py", 'w') as d:
        d.write(executable_plan)

    tm_src = Path(os.getcwd() + "/data/aithor_connect_v2/task_manager.py")
    tm_dst = Path(log_path + "/task_manager.py")
    shutil.copy(tm_src, tm_dst)

    return (f"{log_path}/executable_plan.py")

parser = argparse.ArgumentParser()
parser.add_argument("--command", type=str, required=True)
args = parser.parse_args()

expt_name = args.command
print (expt_name)
ai_exec_file = compile_aithor_exec_file(expt_name)

subprocess.run(["python", ai_exec_file])