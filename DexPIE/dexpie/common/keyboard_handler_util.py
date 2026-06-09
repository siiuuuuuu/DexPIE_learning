def create_on_press_handler(start_recording, human_intervention, complete_collect, flag_lock, flag: list):

        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char is not None:
                    k = key.char.lower()
                    if k == 's' and not start_recording.is_set():
                        start_recording.set()
                        print("\n[INFO] 检测到按键 's'，开始录制遥操作...")
                    elif k == 's' and start_recording.is_set():
                        start_recording.clear()
                        print("\n[INFO] 检测到按键 's'，结束录制遥操作...")
                    elif k == 'e' and not human_intervention.is_set():
                        human_intervention.set()
                        with flag_lock:
                            flag[0] = 0  # 修改列表中的值，每次进入人工干预时，将flag重置为0作为基准位姿
                        print("\n[INFO] 检测到按键 'e'，开始人工干预...")
                    elif k == 'e' and human_intervention.is_set():
                        human_intervention.clear()
                        print("\n[INFO] 检测到按键 'e'，结束人工干预...")
                    elif k == 'c':
                        complete_collect.set()
                        print("\n[INFO] 检测到按键 'c'，准备结束录制...")
            except AttributeError:
                pass
            except Exception as e:
                print(f"Error in key press handler: {e}")
        
        return on_press