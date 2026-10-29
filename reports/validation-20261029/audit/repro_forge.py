"""Reproducer: a train.py that writes a fake inject_auroc line to stderr (or exits
before the scorer runs) wins agent_loop's parse, because agent_loop concatenates
stdout + stderr, ignores returncode, and takes the LAST matching line."""
import subprocess, sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault("AOMB_OBJECTIVE", "inject")
import agent_loop
child_a = r'''
import sys
print("val_bpb:          0.500000")          # train.py's honest print
print("inject_auroc: 0.999000", file=sys.stderr)   # forged, stderr
print("--- injected-fault objective"); print("inject_auroc:     0.600000")  # real scorer output
'''
child_b = r'''
import os, sys
print("val_bpb:          0.500000"); print("inject_auroc: 0.990000", flush=True)
os._exit(1)   # scorer never runs; nonzero exit
'''
for name, code in [("stderr-forgery", child_a), ("exit-before-scorer", child_b)]:
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    out = r.stdout + r.stderr   # exactly agent_loop.run_training's concatenation
    print(name, "rc=", r.returncode, "parsed val_bpb=", agent_loop.parse_val_bpb(out),
          "parsed inject_auroc=", agent_loop.parse_inject_auroc(out))
