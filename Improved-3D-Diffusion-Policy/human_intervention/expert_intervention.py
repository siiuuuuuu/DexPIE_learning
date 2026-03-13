import numpy as np
from human_intervention.manus_inspire import inspire_Manus
from human_intervention import triad_openvr
import time
#封装tele作为后台人类干预子进程，使用键盘来进行切换
class ExpertIntervention(object):
    def __init__(self, use_right_hand=True, use_left_hand=False):
        #初始化
        self.human_intervention = False
        self.hand = inspire_Manus(use_right_hand=use_right_hand, use_left_hand=use_left_hand)
        self.tracker = triad_openvr.triad_openvr()#连接tracker(无需start其已经是运行的子进程了)
        self.tracker.print_discovered_objects()

    def start(self):
        self.hand.start()#开启hand子进程

    #回调获取当前人类干预动作
    def __call__(self):

        tracker_data=np.eye(4)
        pose_mat=self.tracker.devices["tracker_1"].get_pose_matrix()#获取当前tracker位姿
        
        pose_np = np.array([
            [pose_mat[0][0], pose_mat[0][1], pose_mat[0][2], pose_mat[0][3]],
            [pose_mat[1][0], pose_mat[1][1], pose_mat[1][2], pose_mat[1][3]],
            [pose_mat[2][0], pose_mat[2][1], pose_mat[2][2], pose_mat[2][3]]
        ])
        tracker_data[:3,:4] = pose_np 
        hand_action_dict=self.hand()["right"] #回调获取最新手套动作

        if pose_mat is None:
            return None, hand_action_dict #本次回调不是有效数据，不执行
        else:
            return tracker_data, hand_action_dict
        
    def finalize(self):
        self.hand.finalize()
        del self.tracker
