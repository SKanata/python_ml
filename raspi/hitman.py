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
            'q': int('10000000',2), 'w': int('01000000',2), 'e': int('00100000',2), 'r': int('00010000',2),
            't': int('00001000',2), 'y': int('00000100',2), 'g': int('00000010',2), 'f': int('00000001',2),
            'd': int('10000000',2), 's': int('01000000',2), 'a': int('00100000',2), 'v': int('00010000',2),
            'c': int('00001000',2), 'x': int('00000100',2), 'z': int('00000010',2), 'u': int('10000000',2),
            'i': int('01000000',2), 'o': int('00100000',2), 'p': int('00010000',2), '-': int('00001000',2),
            'l': int('00000100',2), 'k': int('00000010',2), 'j': int('00000001',2), 'h': int('00000001',2),
            'b': int('00000010',2), 'n': int('00000100',2), 'm': int('00001000',2), ',': int('00010000',2)
        }
        
        self.icaddr = {
            'q': self.ICADDR_2, 'w': self.ICADDR_2, 'e': self.ICADDR_2, 'r': self.ICADDR_2, 't': self.ICADDR_2,
            'y': self.ICADDR_2, 'g': self.ICADDR_2, 'f': self.ICADDR_2, 'd': self.ICADDR_2, 's': self.ICADDR_2,
            'a': self.ICADDR_2, 'v': self.ICADDR_2, 'c': self.ICADDR_2, 'x': self.ICADDR_2, 'z': self.ICADDR_2,
            'u': self.ICADDR_1, 'i': self.ICADDR_1, 'o': self.ICADDR_1, 'p': self.ICADDR_1, '-': self.ICADDR_1,
            'l': self.ICADDR_1, 'k': self.ICADDR_1, 'j': self.ICADDR_1, 'h': self.ICADDR_1, 'b': self.ICADDR_1,
            'n': self.ICADDR_1, 'm': self.ICADDR_1, ',': self.ICADDR_1
        }

        self.register = {
            'q': self.REG_OLAT_A, 'w': self.REG_OLAT_A, 'e': self.REG_OLAT_A, 'r': self.REG_OLAT_A, 't': self.REG_OLAT_A,
            'y': self.REG_OLAT_A, 'g': self.REG_OLAT_A, 'f': self.REG_OLAT_A, 'd': self.REG_OLAT_B, 's': self.REG_OLAT_B,
            'a': self.REG_OLAT_B, 'v': self.REG_OLAT_B, 'c': self.REG_OLAT_B, 'x': self.REG_OLAT_B, 'z': self.REG_OLAT_B,
            'u': self.REG_OLAT_A, 'i': self.REG_OLAT_A, 'o': self.REG_OLAT_A, 'p': self.REG_OLAT_A, '-': self.REG_OLAT_A,
            'l': self.REG_OLAT_A, 'k': self.REG_OLAT_A, 'j': self.REG_OLAT_A, 'h': self.REG_OLAT_B, 'b': self.REG_OLAT_B,
            'n': self.REG_OLAT_B, 'm': self.REG_OLAT_B, ',': self.REG_OLAT_B
        }
        
        self.bus = smbus.SMBus(self.CHANNEL)

    def initialize(self):
        # ピンの入出力設定
        self.bus.write_byte_data(self.ICADDR_1, self.REG_OLAT_A, 0x00)
        self.bus.write_byte_data(self.ICADDR_1, self.REG_OLAT_B, 0x00)
        self.bus.write_byte_data(self.ICADDR_2, self.REG_OLAT_A, 0x00)
        self.bus.write_byte_data(self.ICADDR_2, self.REG_OLAT_B, 0x00)

    def hit_keys(self, word, interval=0.3):
        for i in word:
            print('Hitting the key: %(word)s. ' % {'word': i})
            self.bus.write_byte_data(self.icaddr[i], self.register[i], self.alphabet[i])
            time.sleep(interval)
            self.bus.write_byte_data(self.icaddr[i], self.register[i], 0x00)

