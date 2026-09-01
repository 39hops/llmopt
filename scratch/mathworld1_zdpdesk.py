"""MATH-CYBER-1 ZERO-DEPTH-PRIOR-ASSESSMENT-0 desk census over
booked artifacts only (population files, residual-census raws,
training corpus): Q2 state-invariance, Q3-Q5 evaluation
geometry + 5040-order brute force, Q6 training Markov
marginals, Q7 gain/harm, Q9 R feasibility. No model, no mask,
no training. Output logs/mathworld1/zdp/desk_census.json
(booked at RESULTS L62394).

    .venv/bin/python scratch/mathworld1_zdpdesk.py           (Mac)
"""
import json, itertools, math, collections, statistics, sys
sys.path.insert(0,'.')
from scratch.mathworld1_svpcode import factor_symbols
from scratch.mathworld1_svpforder import pf_encode, PERM
POPS={'heldout':'logs/mathworld1/svpdiet3/heldout_test16.jsonl','calibration':'logs/mathworld1/svpdiet3/covered_calibration.jsonl'}
def load(pop):
    rows=[json.loads(l) for l in open(POPS[pop])]
    if pop=='heldout': rows=[r for r in rows if r['site_role']=='heldout-I1']
    out=[]
    for r in rows:
        codes=[tuple(c['factor_code']) for c in r['candidates']]
        li=[i for i,c in enumerate(r['candidates']) if c['is_label']][0]
        out.append({'bid':r['block_id'],'codes':codes,'gold':li,'cur':r['cur']})
    return out
pops={p:load(p) for p in POPS}
res={}
# ---- Q3/Q4/Q5 geometry ----
for p,st in pops.items():
    uniq=sorted(set(c for s in st for c in s['codes']))
    gold_f=collections.Counter(s['codes'][s['gold']] for s in st)
    cand_f=collections.Counter(c for s in st for c in s['codes'])
    sizes=collections.Counter(len(s['codes']) for s in st)
    sig=collections.Counter((tuple(sorted(s['codes'])),s['codes'][s['gold']]) for s in st)
    setsig=collections.Counter(tuple(sorted(s['codes'])) for s in st)
    # brute force fixed rankings over uniq codes (7)
    best=0; bests=[]; per_order={}
    for perm in itertools.permutations(uniq):
        rank={c:i for i,c in enumerate(perm)}
        ok=[rank[s['codes'][s['gold']]]<min(rank[c] for c in s['codes'] if c!=s['codes'][s['gold']]) for s in st]
        n=sum(ok)
        if n>best: best=n; bests=[(perm,ok)]
        elif n==best: bests.append((perm,ok))
    missed=[set(i for i,o in enumerate(ok) if not o) for _,ok in bests]
    miss_union=set().union(*missed); miss_inter=set.intersection(*missed)
    # pairwise precedence evidence: gold > rival counts
    prec=collections.Counter()
    for s in st:
        g=s['codes'][s['gold']]
        for c in s['codes']:
            if c!=g: prec[(g,c)]+=1
    contradictions=[(a,b,prec[(a,b)],prec[(b,a)]) for (a,b) in prec if prec[(b,a)]>0 and a<b]
    # matched conflict groups: same code multiset, different gold
    groups=collections.defaultdict(collections.Counter)
    for s in st: groups[tuple(sorted(s['codes']))][s['codes'][s['gold']]]+=1
    conflict=[(k,dict(v)) for k,v in groups.items() if len(v)>1]
    conf_states=sum(sum(v.values()) for k,v in conflict)
    conf_ceiling=sum(max(v.values()) for k,v in conflict)  # fixed ranking best inside conflict groups
    nonconf_states=sum(sum(v.values()) for k,v in groups.items() if len(v)==1)
    # macro-by-signature ceiling for best fixed order(s): mean over signature cells of accuracy
    def macro(ok):
        cells=collections.defaultdict(list)
        for i,s in enumerate(st): cells[(tuple(sorted(s['codes'])),s['codes'][s['gold']])].append(ok[i])
        return statistics.mean(statistics.mean(v) for v in cells.values())
    macro_best=max(macro(ok) for _,ok in bests)
    # ceiling under equal-weight signatures (over ALL orders)
    macro_all=0
    for perm in itertools.permutations(uniq):
        rank={c:i for i,c in enumerate(perm)}
        ok=[rank[s['codes'][s['gold']]]<min(rank[c] for c in s['codes'] if c!=s['codes'][s['gold']]) for s in st]
        macro_all=max(macro_all,macro(ok))
    res[p]={'n_unique_codes':len(uniq),'codes':[list(c) for c in uniq],
        'gold_freq':{str(list(k)):v for k,v in gold_f.items()},'cand_freq':{str(list(k)):v for k,v in cand_f.items()},
        'set_sizes':dict(sizes),'n_set_signatures':len(setsig),'n_setgold_signatures':len(sig),
        'sig_multiplicities':sorted(sig.values(),reverse=True),
        'fixed_ranking_ceiling':best,'n_optimal_orders':len(bests),
        'missed_union':sorted(miss_union),'missed_intersection':sorted(miss_inter),
        'contradictory_pairs':[(list(a),list(b),x,y) for a,b,x,y in contradictions],
        'conflict_groups':[(str([list(c) for c in k]),{str(list(g)):n for g,n in v.items()}) for k,v in conflict],
        'conflict_states':conf_states,'conflict_fixed_ceiling':conf_ceiling,'nonconflict_states':nonconf_states,
        'macro_sig_acc_of_best_orders':round(macro_best,4),'macro_sig_ceiling_all_orders':round(macro_all,4),
        'optimal_order_example':[list(c) for c in bests[0][0]]}
    st_map=st
    res[p]['_st']=st
# ---- Q2/Q7/Q9 from census raws ----
def load_raw(path):
    d={}
    for l in open(path):
        t=json.loads(l)
        if t['mask'] in (0,255): d[(t['pop'],t['arm'],t['mask'])]=t
    return d
raws={'19001':load_raw('logs/mathworld1/respath/raw_census.jsonl'),'20001':load_raw('logs/mathworld1/respath20/raw_census.jsonl')}
q2={}; q7={}; q9={}
for seed,d in raws.items():
    for (pop,arm,mask),t in d.items():
        st=pops[pop]
        if mask==0:
            # code -> sums across states (arm-specific serialization: PF uses pf_encode of tuple? codes in pop are factor codes; PF cont = perm of factor code)
            spread=collections.defaultdict(list)
            for s,row in zip(st,t['states']):
                assert row['block_id']==s['bid']
                for c,v in zip(s['codes'],row['sums']): spread[c].append(v)
            q2[f'{seed}:{pop}:{arm}']={'n_codes':len(spread),'max_spread':max(max(v)-min(v) for v in spread.values()),
                'code_scores':{str(list(c)):round(statistics.mean(v),4) for c,v in spread.items()}}
    for pop in POPS:
        for arm in ('CANONICAL','PARAM_FIRST'):
            t0=d[(pop,arm,0)]; tf=d[(pop,arm,255)]
            g=h=bc=bw=0; gains=[]; harms=[]
            for i,(r0,rf) in enumerate(zip(t0['states'],tf['states'])):
                a,b=r0['top1'],rf['top1']
                if b and not a: g+=1; gains.append(i)
                elif a and not b: h+=1; harms.append(i)
                elif a and b: bc+=1
                else: bw+=1
            # R = S_full - S0 ranking (descriptive feasibility)
            rtop=0
            for s,r0,rf in zip(pops[pop],t0['states'],tf['states']):
                R=[f-z for f,z in zip(rf['sums'],r0['sums'])]
                gl=s['gold']; rtop+= all(R[j]<R[gl] for j in range(len(R)) if j!=gl)
            q7[f'{seed}:{pop}:{arm}']={'mask0':sum(r['top1'] for r in t0['states']),'full':sum(r['top1'] for r in tf['states']),'gain':g,'harm':h,'both_correct':bc,'both_wrong':bw,'gain_states':gains,'harm_states':harms}
            q9[f'{seed}:{pop}:{arm}']={'R_top1':rtop}
res['Q2']=q2; res['Q7']=q7; res['Q9_R_feasibility']=q9
# ---- Q6 training marginals ----
def code_of(r):
    return tuple(factor_symbols(r['rule'],r['site_kind'],r['site_ordinal'],r['param_kind'],r['param_index']))
tr=collections.Counter(); trans={'CANONICAL':collections.Counter(),'PARAM_FIRST':collections.Counter()}
first={'CANONICAL':collections.Counter(),'PARAM_FIRST':collections.Counter()}
for path in ('data/matsub_paired.jsonl','logs/mathworld1/svpdiet/balanced_grid_train.jsonl'):
    for l in open(path):
        r=json.loads(l); c=code_of(r); tr[c]+=1
        pf=[c[PERM[i]] for i in range(8)]
        for arm,seq in (('CANONICAL',list(c)),('PARAM_FIRST',pf)):
            first[arm][seq[0]]+=1
            for a,b in zip(seq,seq[1:]): trans[arm][(a,b)]+=1
            trans[arm][(seq[-1],'EOS')]+=1
def markov_score(seq,arm):
    tot=sum(first[arm].values()); s=math.log((first[arm][seq[0]]+0.5)/(tot+4))
    for a,b in zip(seq,seq[1:]+['EOS']):
        row=sum(v for (x,y),v in trans[arm].items() if x==a); s+=math.log((trans[arm][(a,b)]+0.5)/(row+4.5))
    return s
q6={'n_training_rows':sum(tr.values())}
for p,st in pops.items():
    uniq=sorted(set(c for s in st for c in s['codes']))
    q6[p]={'whole_code_train_freq':{str(list(c)):tr[c] for c in uniq}}
    for arm in ('CANONICAL','PARAM_FIRST'):
        seqs={c:(list(c) if arm=='CANONICAL' else [c[PERM[i]] for i in range(8)]) for c in uniq}
        emp={c:markov_score(seqs[c],arm) for c in uniq}
        emp_rank=sorted(uniq,key=lambda c:-emp[c])
        q6[p][arm]={'empirical_markov_order':[list(c) for c in emp_rank]}
        for seed in raws:
            cs=q2[f'{seed}:{p}:{arm}']['code_scores']; m0=sorted(uniq,key=lambda c:-cs[str(list(c))])
            # spearman between emp rank and mask0 rank
            re={c:i for i,c in enumerate(emp_rank)}; rm={c:i for i,c in enumerate(m0)}
            n=len(uniq); dsq=sum((re[c]-rm[c])**2 for c in uniq); rho=1-6*dsq/(n*(n*n-1))
            q6[p][arm][f'mask0_order_{seed}']=[list(c) for c in m0]; q6[p][arm][f'spearman_emp_v_mask0_{seed}']=round(rho,3)
res['Q6']=q6
for p in POPS: res[p].pop('_st')
json.dump(res,open('logs/mathworld1/zdp/desk_census.json','w'),indent=1)
print('done')
