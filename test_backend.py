"""
Backend integration test for the Django version.
Starts django dev server, tests GET / and POST /chat.
"""

import subprocess
import time
import urllib.request
import json
import sys
import os

def run_tests():
    print("Starting Django development server...")
    env = os.environ.copy()
    env["DJANGO_SETTINGS_MODULE"] = "fodwa_project.settings"
    proc = subprocess.Popen(
        [sys.executable, "manage.py", "runserver", "10000", "--noreload"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    time.sleep(5)  # Wait for server to start

    try:
        # Check if process died
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            print("Server failed to start!")
            print(stderr.decode())
            sys.exit(1)

        print("✅ Server started. Testing GET / ...")
        req = urllib.request.Request("http://127.0.0.1:10000/")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"
            print("✅ Health check passed!")

        print("Testing POST /chat ...")
        payload = json.dumps({"message": "كيف يمكنني إزالة إعلاناتك؟"}).encode("utf-8")
        req = urllib.request.Request("http://127.0.0.1:10000/chat", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            assert "response" in data
            print("✅ Chat endpoint passed!")
            print(f"🤖 Response: {data['response']}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    finally:
        print("Shutting down server...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    run_tests()
    print("✅ All tests completed successfully!")
