import json, os, shutil, hashlib, sys
sys.path.insert(0, '.')
from huggingface_hub import snapshot_download
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<22), b''): h.update(c)
    return h.hexdigest()
cen=json.load(open('logs/k2h/stagecensus/receipt.json'))
res=json.load(open('logs/k2h/residues/receipt.json'))
out={}
for tag, old in (('pretrain_500000', cen['tags']['pretrain_500000']['shard_sha256']), ('mid_2_47684', cen['tags']['mid_2_final']['shard_sha256']), ('mid_1_50000', cen['tags']['mid_1_50000']['shard_sha256'])):
    cache=f'logs/k2h/_tagmove/{tag}'
    p=snapshot_download('IFM/K2-Horizon-0.9B', revision=tag, allow_patterns=['*.json','*.safetensors'], cache_dir=cache)
    commit=os.path.basename(p)
    idx=json.load(open(os.path.join(p,'model.safetensors.index.json')))
    new={s:sha(os.path.join(p,s)) for s in sorted(set(idx['weight_map'].values()))}
    same=sum(new.get(k)==v for k,v in old.items())
    out[tag]={'new_commit':commit,'shards_identical':f'{same}/{len(old)}','n_new_shards':len(new)}
    print(tag, out[tag], flush=True)
    shutil.rmtree(cache, ignore_errors=True)
json.dump(out, open('/tmp/prband/tagmove.json','w'), indent=1)
