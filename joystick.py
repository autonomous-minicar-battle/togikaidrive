# coding:utf-8
import config
import pygame
import os
import sys
import numpy as np

class Joystick(object):
    def __init__(self, dev_fn=config.JOYSTICK_DEVICE_FILE):
        self.stick_left = config.JOYSTICK_AXIS_LEFT
        self.stick_right = config.JOYSTICK_AXIS_RIGHT
        self.button_Y = config.JOYSTICK_Y
        self.button_X = config.JOYSTICK_X
        self.button_A = config.JOYSTICK_A
        self.button_B = config.JOYSTICK_B
        self.button_S = config.JOYSTICK_S
        self.steer = 0.
        self.accel = 0.
        self.accel1 = 0.
        self.accel2 = 0.
        self.breaking = 0
        self.mode = ["auto","auto_str","user"]
        self.recording = True
        # pygameの初期化
        pygame.init()
        # ジョイスティックの初期化
        pygame.joystick.init()
        # ジョイスティックインスタンスの生成
        try:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print('ジョイスティックの名前:', self.joystick.get_name())
            print('ボタン数 :', self.joystick.get_numbuttons())
        except pygame.error:
            print('ジョイスティックが接続されていません')
    
    #def poll(self,steer,accel,accel1,accel2,breaking):
    def poll(self):
        # イベントがある場合は更新
        for e in pygame.event.get():
            self.steer = round(self.joystick.get_axis(self.stick_left),2)
            self.accel = round(self.joystick.get_axis(self.stick_right),2)
            self.accel1 = self.joystick.get_button(self.button_A)
            self.accel2 = self.joystick.get_button(self.button_B)
            self.breaking = self.joystick.get_button(self.button_X)
            if self.joystick.get_button(self.button_S):
                self.mode = np.roll(self.mode,1)
                print(" mode:",self.mode[0])
            if self.joystick.get_button(self.button_Y):
                self.recording = not self.recording

            #print()
        #return steer,accel,accel1,accel2, breaking

if __name__ == "__main__":
    joystick = Joystick()
    #steer,accel1,accel2 = 0., 0, 0
    while True:
        #steer,accel1,accel2 = joystick.poll(steer,accel1,accel2)
        #print("Str:",steer,"Acc1:",accel1,"Acc2:",accel2)
        for e in pygame.event.get():
            print(e)

