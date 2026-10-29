import json, os, sys, statistics as st, collections
from eval.in_domain import load_session_file, baseline_scores, fit_duration_stats, fit_novelty, ValueStats, min_lines
from eval.metrics import auroc
out={}
for sysname in ("online-boutique","train-ticket"):
    rep=f"reports/public-accuracy/rcaeval-re3-{sysname}-eos-20261009"
    saved=json.load(open(rep+"/sessions.json"))
    train, ev, group_of, meta = load_session_file(os.path.expanduser(f"~/.cache/aomb-datasets/rcaeval/sessions/re3_{sysname}.jsonl"))
    assert [s.session_id for s in ev]==saved["session_ids"]
    base=baseline_scores(ev, fit_duration_stats(train), fit_novelty(train), ValueStats(train), min_lines(train))
    sub=meta["subset_ids"]
    res={}
    for view in ("all","subset"):
        idx=[i for i,s in enumerate(ev) if view=="all" or s.session_id in sub]
        by=collections.defaultdict(list)
        for i in idx: by[group_of[ev[i].session_id]].append(i)
        usable={g:ix for g,ix in by.items() if len({saved["labels"][i] for i in ix})==2}
        def percase(scores): return st.mean(auroc([saved["labels"][i] for i in ix],[scores[i] for i in ix]) for ix in usable.values())
        model=[percase(s["bpb_top10"]) for s in saved["model_scores_by_seed"].values()]
        res[view]={"cases_usable":len(usable),"cases_total":len(by),
                   "model_bpb_top10":(round(st.mean(model),3),round(st.pstdev(model),3)),
                   **{b:round(percase(base[b]),3) for b in ("heuristic","novelty","rule","error_lines")}}
    out[sysname]=res
print(json.dumps(out,indent=1))
