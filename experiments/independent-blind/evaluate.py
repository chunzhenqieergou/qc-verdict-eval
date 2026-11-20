import gc,json,os
from pathlib import Path
from freeze import R,OLD,sha
for k in ("TEMP","TMP","HF_HOME","TORCH_HOME","TRITON_CACHE_DIR","TORCHINDUCTOR_CACHE_DIR","CUDA_CACHE_PATH"):
    d=R/".cache"/k.lower();d.mkdir(parents=True,exist_ok=True);os.environ[k]=str(d)
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,GenerationConfig
def load(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def main():
    manifest=load(R/"manifest.json")
    assert manifest["cases"]==sha(R/"cases.json") and manifest["protocol"]==sha(R/"protocol.md")
    cases=load(R/"cases.json")
    base=Path(load(OLD/"environment.json")["base"])
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True,padding_side="left");tok.pad_token=tok.eos_token
    prompts=load(OLD/"order-training/prompts.json")
    outdir=R/"outputs";outdir.mkdir(exist_ok=True)
    for seed in [None,42,43,44]:
        for order in ["original","reordered"]:
            name=f"{'base' if seed is None else seed}-{order}"
            out=outdir/(name+".json")
            if out.exists():
                assert [r["id"] for r in load(out)]==[r["id"] for r in cases]
                continue
            model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,device_map={"":0},dtype=torch.bfloat16,attn_implementation="sdpa",quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_type="nf4"))
            if seed is not None:
                path=OLD/("runs" if order=="original" else "order-training")/f"scenario-{seed}"/"adapter"
                f=path/"adapter_model.safetensors"
                assert sha(f)==manifest["weights"][str(f.relative_to(OLD))]
                model=PeftModel.from_pretrained(model,path)
            model.eval()
            eos=model.generation_config.eos_token_id
            cfg=GenerationConfig(max_new_tokens=96,do_sample=False,eos_token_id=eos,pad_token_id=tok.pad_token_id,use_cache=True)
            rows=[]
            for i in range(0,len(cases),12):
                group=cases[i:i+12]
                chats=[[dict(role="system",content=prompts[order]),dict(role="user",content=r["question"])] for r in group]
                enc=tok.apply_chat_template(chats,add_generation_prompt=True,tokenize=True,padding=True,return_tensors="pt",return_dict=True).to("cuda")
                with torch.inference_mode(): ids=model.generate(**enc,generation_config=cfg)[:,enc.input_ids.shape[1]:].tolist()
                for r,seq in zip(group,ids):rows.append(dict(id=r["id"],text=tok.decode(seq,skip_special_tokens=True).strip(),capped=not any(e in seq for e in (eos if isinstance(eos,list) else [eos]))))
            tmp=out.with_suffix(".tmp");tmp.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf8");tmp.replace(out)
            print(name,len(rows),flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
if __name__=="__main__":main()
