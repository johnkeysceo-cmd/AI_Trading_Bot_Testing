import threading
import time

class Scheduler:
    """
    Periodic task scheduler
    """
    def __init__(self):
        self.tasks = []

    def add_task(self, func, interval_sec: int):
        self.tasks.append((func, interval_sec))
        threading.Thread(target=self._run_task, args=(func, interval_sec), daemon=True).start()

    def _run_task(self, func, interval_sec):
        while True:
            try:
                func()
                time.sleep(interval_sec)
            except Exception as e:
                print(f"Scheduler task error: {e}")
