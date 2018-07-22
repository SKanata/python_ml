#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import smbus2 as smbus
import time

class Hitman(object):
    
    CHANNEL    = 1      # i2c割り当てチャンネル 1 or 0
    ICADDR_1    = 0x20   # スレーブ側ICアドレス1
    ICADDR_2    = 0x21   # スレーブ側ICアドレス1
    REG_IODIR_A = 0x00   # 入出力設定レジスタ
    REG_IODIR_B = 0x01   # 入出力設定レジスタ
    REG_OLAT_A  = 0x14   # 出力レジスタA
    REG_OLAT_B  = 0x15   # 出力レジスタB
    
    def __init__(self):
        self.alphabet = {
            'q': int('10000000',2), 'a': int('01000000',2), 'z': int('00100000',2), 'w': int('00010000',2),
            's': int('00001000',2), 'x': int('00000100',2), 'e': int('00000010',2), 'd': int('00000001',2),
            'c': int('00000001',2), 'r': int('00000010',2), 'f': int('00000100',2), 'v': int('00001000',2),
            't': int('00010000',2), 'g': int('00100000',2), 'b': int('01000000',2), 'y': int('10000000',2), 
            'h': int('00000001',2), 'n': int('00000010',2), 'u': int('00000100',2), 'j': int('00001000',2), 
            'm': int('00010000',2), 'i': int('00100000',2), 'k': int('01000000',2), ',': int('10000000',2),
            'o': int('10000000',2), 'l': int('01000000',2), 'p': int('00100000',2), '-': int('00010000',2)
        }
        
        self.icaddr = {
            'q': self.ICADDR_1, 'w': self.ICADDR_1, 'e': self.ICADDR_1, 'r': self.ICADDR_1, 't': self.ICADDR_1, 'y': self.ICADDR_1,
            'a': self.ICADDR_1, 's': self.ICADDR_1, 'd': self.ICADDR_1, 'f': self.ICADDR_1, 'g': self.ICADDR_1,
            'z': self.ICADDR_1, 'x': self.ICADDR_1, 'c': self.ICADDR_1, 'v': self.ICADDR_1, 'b': self.ICADDR_1,

                                'u': self.ICADDR_2, 'i': self.ICADDR_2, 'o': self.ICADDR_2, 'p': self.ICADDR_2,
            'h': self.ICADDR_2, 'j': self.ICADDR_2, 'k': self.ICADDR_2, 'l': self.ICADDR_2, '-': self.ICADDR_2,
            'n': self.ICADDR_2, 'm': self.ICADDR_2, ',': self.ICADDR_2
        }

        self.register = {
            'q': self.REG_OLAT_A, 'w': self.REG_OLAT_A, 'e': self.REG_OLAT_A, 'r': self.REG_OLAT_B, 't': self.REG_OLAT_B, 'y': self.REG_OLAT_B,
            'a': self.REG_OLAT_A, 's': self.REG_OLAT_A, 'd': self.REG_OLAT_A, 'f': self.REG_OLAT_B, 'g': self.REG_OLAT_B,
            'z': self.REG_OLAT_A, 'x': self.REG_OLAT_A, 'c': self.REG_OLAT_B, 'v': self.REG_OLAT_B, 'b': self.REG_OLAT_B,

                                'u': self.REG_OLAT_B, 'i': self.REG_OLAT_B, 'o': self.REG_OLAT_A, 'p': self.REG_OLAT_A,
            'h': self.REG_OLAT_B, 'j': self.REG_OLAT_B, 'k': self.REG_OLAT_B, 'l': self.REG_OLAT_A, '-': self.REG_OLAT_A,
            'n': self.REG_OLAT_B, 'm': self.REG_OLAT_B, ',': self.REG_OLAT_B
        }
        
        self.bus = smbus.SMBus(self.CHANNEL)

    def initialize(self):
        # ピンの入出力設定
        self.bus.write_byte_data(self.ICADDR_1, self.REG_IODIR_A, 0x00)
        self.bus.write_byte_data(self.ICADDR_1, self.REG_IODIR_B, 0x00)
        self.bus.write_byte_data(self.ICADDR_2, self.REG_IODIR_A, 0x00)
        self.bus.write_byte_data(self.ICADDR_2, self.REG_IODIR_B, 0x00)

    def hit_keys(self, word, interval=0.02):
        for i in word:
            print('Hitting the key: %(word)s. ' % {'word': i})
            self.bus.write_byte_data(self.icaddr[i], self.register[i], self.alphabet[i])
            time.sleep(interval)
            self.bus.write_byte_data(self.icaddr[i], self.register[i], 0x00)
            time.sleep(interval)
