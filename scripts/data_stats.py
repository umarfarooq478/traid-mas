"""Compute Who&When dataset statistics (reads JSON only)."""
import json, glob, os, sys, statistics as st
root = sys.argv[1]
for d in ["Hand-Crafted", "Algorithm-Generated"]:
    fs = glob.glob(os.path.join(root, d, "*.json")); L=[]; C=[]; keys=set(); S=[]
    for f in fs:
        x = json.load(open(f, encoding="utf-8")); keys |= set(x.keys()); h = x["history"]
        L.append(len(h)); C.append(sum(len(str(e.get("content",""))) for e in h)); S.append(int(x["mistake_step"])/max(1,len(h)-1))
    print(d, len(fs), sorted(keys), "steps med/mean/max", st.median(L), round(st.mean(L),1), max(L),
          "chars med/max", st.median(C), max(C), "total", sum(C), "rel.step mean", round(st.mean(S),2))
x = json.load(open(os.path.join(root,"Algorithm-Generated","1.json"), encoding="utf-8"))
print({k:(str(v)[:300]) for k,v in x.items() if k!="history"}); print([{k:str(v)[:150] for k,v in e.items()} for e in x["history"][:3]])
x = json.load(open(os.path.join(root,"Hand-Crafted","1.json"), encoding="utf-8"))
print({k:(str(v)[:300]) for k,v in x.items() if k!="history"}); print([{k:str(v)[:150] for k,v in e.items()} for e in x["history"][:3]])
