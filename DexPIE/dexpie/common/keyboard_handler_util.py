def create_on_press_handler(start_recording, human_intervention, complete_collect, flag_lock, flag: list):

        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char is not None:
                    k = key.char.lower()
                    if k == 's' and not start_recording.is_set():
                        start_recording.set()
                        print("\n[INFO] Key 's' pressed: start teleoperation recording.")
                    elif k == 's' and start_recording.is_set():
                        start_recording.clear()
                        print("\n[INFO] Key 's' pressed: stop teleoperation recording.")
                    elif k == 'e' and not human_intervention.is_set():
                        human_intervention.set()
                        with flag_lock:
                            flag[0] = 0  # Reset flag to 0 as the reference pose when entering human intervention.
                        print("\n[INFO] Key 'e' pressed: start human intervention.")
                    elif k == 'e' and human_intervention.is_set():
                        human_intervention.clear()
                        print("\n[INFO] Key 'e' pressed: stop human intervention.")
                    elif k == 'c':
                        complete_collect.set()
                        print("\n[INFO] Key 'c' pressed: finish collection.")
            except AttributeError:
                pass
            except Exception as e:
                print(f"Error in key press handler: {e}")
        
        return on_press
