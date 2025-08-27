import time
import random

def run_motion():
    print("[MOTION MODEL] Running...")
    time.sleep(2)
    if random.choice([True, False]):
        print("🚨 Emergency Triggered via Motion!")
        return "motion_detected"
    return "no_motion"
