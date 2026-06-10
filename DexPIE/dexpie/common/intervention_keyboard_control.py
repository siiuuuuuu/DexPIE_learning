import threading

from pynput import keyboard


class InterventionKeyboardControl:
    """Tracks recording, intervention, and collection-stop keyboard state."""

    def __init__(self):
        self.recording = threading.Event()
        self.intervention = threading.Event()
        self.collect_end = threading.Event()
        self.listener = keyboard.Listener(on_press=self._on_press)

    def _on_press(self, key):
        try:
            if not hasattr(key, "char") or key.char is None:
                return

            k = key.char.lower()
            if k == "s" and not self.recording.is_set():
                self.recording.set()
                print("\n[INFO] Key 's' pressed: start teleoperation recording.")
            elif k == "s" and self.recording.is_set():
                self.recording.clear()
                print("\n[INFO] Key 's' pressed: stop teleoperation recording.")
            elif k == "e" and not self.intervention.is_set():
                self.intervention.set()
                print("\n[INFO] Key 'e' pressed: start human intervention.")
            elif k == "e" and self.intervention.is_set():
                self.intervention.clear()
                print("\n[INFO] Key 'e' pressed: stop human intervention.")
            elif k == "c":
                self.collect_end.set()
                self.recording.clear()
                print("\n[INFO] Key 'c' pressed: finish collection.")
        except AttributeError:
            pass
        except Exception as e:
            print(f"Error in key press handler: {e}")

    def start(self):
        self.listener.start()

    def stop(self):
        if self.listener.running:
            self.listener.stop()

    def wait_recording(self):
        while not self.collect_end.is_set() and not self.recording.is_set():
            self.recording.wait(timeout=0.1)

    def is_recording(self):
        return self.recording.is_set()

    def is_intervening(self):
        return self.intervention.is_set()

    def should_stop_collection(self):
        return self.collect_end.is_set()
