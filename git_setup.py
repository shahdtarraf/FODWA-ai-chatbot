import subprocess
import sys

def run(cmd):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error: {res.stderr}")
    else:
        print(res.stdout)

run("git init")
run("git config user.name 'Fodwa AI Builder'")
run("git config user.email 'bot@shahd.ai'")
run("git add .")
run("git commit -m \"Initial commit of production-ready AI RAG Chatbot\"")
