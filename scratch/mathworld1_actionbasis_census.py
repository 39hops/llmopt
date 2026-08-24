import json, sys, collections, statistics
sys.path.insert(0,'/Users/artin/code/llmopt')
import os; os.chdir('/Users/artin/code/llmopt')
from scratch.mathworld1_birth import GCTok
tok=GCTok()

states={}
for l in open('logs/mathworld1/states.jsonl'):
    r=json.loads(l); states[(r['episode_id'],r['step_id'])]=r
acts=collections.defaultdict(list)
for l in open('logs/mathworld1/actions.jsonl'):
    r=json.loads(l); acts[(r['episode_id'],r['step_id'])].append(r)

prog_lens=[]; child_lens=[]; amb_groups=collections.Counter()
completeness_fail=0; rows=0
decile_rows=[]  # (child_len, parent_prefix_len, prog_len)
overflow_decisions=[]
for key, alist in acts.items():
    st=states[key]
    pre=len(tok.encode(f"Current: {st['state_before']}\nHints: none\nStep: "))
    # label per action: rule@target or bare rule (match booked convention)
    labels=[]
    for a in sorted(alist, key=lambda a: a['idx']):
        lab=(f"{a['rule']}@{a['rule_target']}" if a['rule_target'] else a['rule'])
        labels.append((lab,a))
    lab_count=collections.Counter(l for l,_ in labels)
    for lab,cnt in lab_count.items():
        if cnt>1: amb_groups[cnt]+=1
    seen=collections.Counter()
    maxcand=0
    for lab,a in labels:
        k=seen[lab]; seen[lab]+=1
        prog = f"{lab}#{k}\n" if lab_count[lab]>1 else f"{lab}\n"
        pl=len(tok.encode(prog))
        cl=len(tok.encode(a['child']+"\n"))
        prog_lens.append(pl); child_lens.append(cl); rows+=1
        decile_rows.append((cl,pre,pl))
        maxcand=max(maxcand,cl)
    if pre+maxcand>4096:
        overflow_decisions.append((key,pre,maxcand))

prog_lens.sort(); child_lens.sort()
med_p=statistics.median(prog_lens); med_c=statistics.median(child_lens)
decile_rows.sort(key=lambda t:-t[0])
top=decile_rows[:max(1,len(decile_rows)//10)]
maxppp=max(p+pl for c,p,pl in top)
cv=lambda xs: statistics.pstdev(xs)/statistics.mean(xs)
print(json.dumps({
 "actions":rows,"decisions":len(acts),
 "prog_len":{"med":med_p,"p90":prog_lens[int(.9*rows)],"max":max(prog_lens),"cv":round(cv(prog_lens),3)},
 "child_len":{"med":med_c,"p90":child_lens[int(.9*rows)],"max":max(child_lens),"cv":round(cv(child_lens),3)},
 "median_compression":round(med_c/med_p,2),
 "per_action_ratio_med":round(statistics.median(c/ (p or 1) for (c,_,p) in decile_rows),2),
 "threshold_a_med_prog_le_25pct_med_child": med_p<=0.25*med_c,
 "top_decile_max_prefix_plus_prog": maxppp,
 "threshold_b_fits_512": maxppp<=512,
 "ambiguity_groups":dict(amb_groups),
 "ambiguous_actions": sum(k*v for k,v in amb_groups.items()),
 "completeness_fail":completeness_fail,
 "overflow_decisions_on_corpus":[ [list(k),p,m] for k,p,m in overflow_decisions]
},indent=1))
EOF_MARKER_UNUSED = None
