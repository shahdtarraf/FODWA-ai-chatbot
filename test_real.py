"""
Real endpoint test — starts Django server and sends a test query.
"""

import urllib.request
import json
import sys
import subprocess
import time
import os

def test_query():
    print("Starting temporary Django test server...")
    env = os.environ.copy()
    env["DJANGO_SETTINGS_MODULE"] = "fodwa_project.settings"
    proc = subprocess.Popen(
        [sys.executable, "manage.py", "runserver", "10001", "--noreload"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    time.sleep(5)
    
    try:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            print("Server failed to start")
            print(stderr.decode())
            sys.exit(1)
            
        print("Sending request: 'كيف يمكنني إلغاء إعلان مخالف؟'")
        payload = json.dumps({"message": "كيف يمكنني إلغاء إعلان مخالف؟"}).encode("utf-8")
        req = urllib.request.Request("http://127.0.0.1:10001/chat", data=payload, headers={"Content-Type": "application/json"})
        
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"🤖 Response:\n{data['response']}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    test_query()
