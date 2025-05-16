#!/usr/bin/env python3
import os, time, pathlib, requests, bs4, argparse
from urllib.parse import urljoin
from tqdm import tqdm

BASE = "https://openvoiceos.github.io/message_spec/index.html"
UA   = {"User-Agent":"Grace-harvester/4.0"}

TOKEN=os.getenv("GITHUB_TOKEN")
if TOKEN:
    UA["Authorization"]=f"token {TOKEN}"

def fetch(url):
    ok=False
    for _ in range(3):
        r=requests.get(url, headers=UA, timeout=40)
        if r.ok: return r.text
        time.sleep(1)
    r.raise_for_status()

# ------------ BUS COMMANDS ------------
def section_links():
    soup = bs4.BeautifulSoup(fetch(BASE), "html.parser")
    return [urljoin(BASE, a["href"])
            for a in soup.select("a[href]") if a["href"].endswith(("/",".html"))]

def walk_pages(start):
    page=start
    while page:
        yield page
        s=bs4.BeautifulSoup(fetch(page), "html.parser")
        nxt=s.find("a", string=lambda t:t and "next" in t.lower())
        page=urljoin(page, nxt["href"]) if nxt else None

def bus_cmds():
    out=set()
    for sec in tqdm(section_links(), desc="Spec sections"):
        for pg in walk_pages(sec):
            soup=bs4.BeautifulSoup(fetch(pg), "html.parser")
            for code in soup.select("code"):
                txt=code.get_text(strip=True)
                if ("." in txt or ":" in txt) and "test" not in txt.lower():
                    out.add(txt.replace("mycroft.","ovos."))
    return out

# ------------ INTENT COMMANDS ------------
ORG="https://api.github.com/orgs/OpenVoiceOS/repos?per_page=100"

def repos():
    page=1
    while True:
        url=f"{ORG}&page={page}"
        data=requests.get(url, headers=UA, timeout=40).json()
        if isinstance(data, dict) and data.get("message","").startswith("API rate"):
            time.sleep(60) ; continue   # backoff when hitting rate limit
        if not data: break
        yield from [r for r in data if r["name"].startswith("skill-")]
        page+=1
        if not TOKEN: time.sleep(1.2)   # slow when anonymous

def intents(repo):
    tree_url=f"https://api.github.com/repos/{repo['full_name']}/git/trees/{repo['default_branch']}?recursive=1"
    tree=requests.get(tree_url, headers=UA, timeout=60).json().get("tree",[])
    return {pathlib.Path(t["path"]).stem for t in tree
            if t["path"].endswith(".intent") and "test" not in t["path"].lower()}

def intent_cmds():
    out=set()
    for r in tqdm(list(repos()), desc="Skill repos"):
        out |= intents(r)
    return out

# ------------ MAIN ------------
def main(out):
    bus = bus_cmds()
    intents = intent_cmds()
    cmds = sorted(bus|intents)
    pathlib.Path(out).write_text("\n".join(cmds), encoding="utf-8")
    print(f"Wrote {len(cmds)} commands → {out}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="ovos_commands.txt")
    main(ap.parse_args().out)
