import urllib.request
import json
import sys
import subprocess
import time

def test_query():
    print("Starting temporary test server...")
    proc = subprocess.Popen(["uvicorn", "app.main:app", "--port", "10001"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(4)
    
    try:
        if proc.poll() is not None:
            print("Server failed to start")
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
