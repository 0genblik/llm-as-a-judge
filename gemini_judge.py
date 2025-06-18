import os, json, time, random, pathlib, importlib
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd 
import streamlit as st 

from datasets import load_dataset
from scipy.stats import bootstrap
import google.generativeai as genai

MT_DATASET_ID = "lmsys/mt_bench_human_judgement"
FLASH_MODEL_ID = "gemini-2.0-flash-lite"
RATE_LIMIT_RPM = 30 
CACHE_DIR = pathlib.Path(".gemini_cache")
CACHE_DIR.mkdir(exist_ok=True)
SEED = 42
random.seed(SEED)


# Calculate agreement rate between two judges on a given turn: 


def canonical_judge(judge):
    if isinstance(judge, list) and judge and judge[0] == "gpt-4":
        return "gpt4-pair"
    if isinstance(judge, str) and judge.startswith(("expert", "author")):
        return "human"
    return judge


def fold_tie(v):
    return "tie" if "tie" in v else v


def build_vote_bag(rows):
    """Build a bag of votes from the rows of the dataset."""
    bag = [defaultdict(dict), defaultdict(dict)]

    for row in rows:
        turn = row["turn"] - 1 # change from 1-indexed to 0-indexed

        if row["model_a"] < row["model_b"]:
            key = (row["question_id"], row["model_a"], row["model_b"])
            label = row["winner"]

        else:
            key = (row["question_id"], row["model_b"], row["model_a"])
            label = {"model_a" : "model_b", "model_b" : "model_a"}.get(row["winner"], row["winner"])

        judge = canonical_judge(row["judge"])

        bag[turn].setdefault(key, {}).setdefault(judge, []).append(label)

    return bag


def agree_turn(bag_turn, judgeA="gemini", judgeB="human", drop_ties=True):
    """
    Return (agree, total) for one turn.
    Mirrors MT-Bench reference exactly.
    """
    agree = tot = 0
    for votes in bag_turn.values():
        if judgeA not in votes or judgeB not in votes:
            continue                                # need both judges
        vA = fold_tie(votes[judgeA][0])             # LLM gives 1 vote
        if drop_ties and vA == "tie":
            continue
        for vB in votes[judgeB]:                    # humans may vote many times
            vB = fold_tie(vB)
            if drop_ties and vB == "tie":
                continue
            tot   += 1
            agree += (vA == vB)                     # bool adds as 1/0
    return agree, tot

def msg_text(msg: Dict) -> str:
    return msg.get("value") or msg.get("content", "")

# Gemini judge set-up and cache 

def _cache_path(key: Tuple) -> pathlib.Path:
    """Get the cache path for a given key."""
    return CACHE_DIR / f"{hash(key)}.json"

def gemini_vote(question: str, ans_a: str, ans_b: str, api_key: str) -> str:
    key = (question, ans_a, ans_b)
    cpath = _cache_path(key)
    if cpath.exists():
        return json.loads(cpath.read_text())
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(FLASH_MODEL_ID)

    prompt = f"""
You are a fair judge. Read QUESTION and two ANSWERS (A and B). Respond with one line only: 
 model_a
 model_b
 tie
QUESTION: {question}

ANSWER A: {ans_a}
ANSWER B: {ans_b}
"""
    try:
        out = model.generate_content(prompt, generation_config={"temperature": 0.0})
    except Exception as e:
        vote = "tie"
        cpath.write_text(json.dumps(vote))
        return vote
    
    decision = out.text.lower().strip()
    if "model_a" in decision and "model_b" in decision:
        vote = "tie"
    elif "model_a" in decision:
        vote = "model_a"
    elif "model_b" in decision:
        vote = "model_b"
    else:
        vote = "tie"
    cpath.write_text(json.dumps(vote))
    return vote


# Streamlit UI


st.set_page_config("Gemini Flash-Lite • MT-Bench", layout="wide")
st.title("Gemini 2.0 Flash-Lite as MT-Bench Judge")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google API Key", type="password")
    sample_size = st.slider("Sample size (turn-1 pairs w/ human votes)", 100, 3355, 800, 100)
    run = st.button("Run Evaluation", disabled=not bool(api_key))
    st.caption("*Pairs are pre-filtered to guarantee ≥1 human vote.*")

if not run:
    st.stop()


# 1 Load dataset & intersect keys

st.write("### 1. Loading MT‑Bench splits …")
try:
    gpt_rows   = load_dataset(MT_DATASET_ID, split="gpt4_pair")
    human_rows = load_dataset(MT_DATASET_ID, split="human")
except Exception as e:
    st.error(f"Dataset load failed → {e}")
    st.stop()

human_keys = set()
for r in human_rows:
    if r["turn"] != 1:
        continue
    models_sorted = tuple(sorted((r["model_a"], r["model_b"])))
    human_keys.add((r["question_id"], *models_sorted))

eligible = [r for r in gpt_rows if r["turn"] == 1 and (
    (r["question_id"], *(sorted((r["model_a"], r["model_b"])))) in human_keys)]

if not eligible:
    st.error("No eligible pairs found — unlikely but fatal.")
    st.stop()

if sample_size > len(eligible):
    st.warning(f"Only {len(eligible)} eligible pairs, using all of them.")
    sample_size = len(eligible)

sampled_pairs = random.sample(eligible, sample_size)


# 2. Gemini judging loop


st.write("### 2. Evaluating with Gemini Flash‑Lite …")
prog = st.progress(0.0)
start_time = time.time()

gem_rows = []
for i, row in enumerate(sampled_pairs, 1):
    if i % RATE_LIMIT_RPM == 0:
        time.sleep(max(0, 60 - (time.time() - start_time) % 60))
    q   = msg_text(row["conversation_a"][0])
    ans_a = msg_text(row["conversation_a"][-1])
    ans_b = msg_text(row["conversation_b"][-1])
    vote = gemini_vote(q, ans_a, ans_b, api_key)
    gem_rows.append({**row, "judge": "gemini", "winner": vote})
    prog.progress(i / sample_size, text=f"{i}/{sample_size}")

prog.empty()


# 3 Aggregate & agreement


# Convert HF Datasets -> lists before concatenation
bag = build_vote_bag(list(gpt_rows) + list(human_rows) + gem_rows)
agree, total, flags = agree_turn(bag[0], "gemini", "human")
ratio = agree / total if total else 0.0

if len(flags) >= 2:
    ci = bootstrap((flags,), lambda x: x.mean(axis=-1), n_resamples=10_000,
                   confidence_level=0.95, method="basic").confidence_interval
    ci_low, ci_high = ci.low, ci.high
else:
    ci_low = ci_high = ratio


# Display


st.subheader("Agreement with Human Judges (no‑tie, turn‑1)")
col1, col2 = st.columns([1, 2])
col1.metric("Gemini ↔ Human", f"{ratio*100:.1f}%",
            help=f"N={total}\n95 % CI {ci_low*100:.1f} – {ci_high*100:.1f}")

# Table and viz
metrics_df = pd.DataFrame({"Agreement": [ratio], "CI low": [ci_low], "CI high": [ci_high]})
col2.dataframe(metrics_df, use_container_width=True)


# Save download 

out_file = "gemini_vs_human_stats.json"
with open(out_file, "w") as f:
    json.dump({
        "agree": int(agree),
        "total": int(total),
        "ratio": ratio,
        "ci": [ci_low, ci_high],
        "sample": sample_size,
    }, f, indent=2)

st.download_button("Download JSON", file_name=out_file, mime="application/json",
                   data=open(out_file, "rb").read())

st.balloons()
