#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import smbus2 as smbus
import time

CHANNEL    = 1      # i2c割り当てチャンネル 1 or 0
ICADDR_1    = 0x20   # スレーブ側ICアドレス1
REG_IODIR_A = 0x00   # 入出力設定レジスタ
REG_IODIR_B = 0x01   # 入出力設定レジスタ
REG_OLAT_A  = 0x14   # 出力レジスタA
REG_OLAT_B  = 0x15   # 出力レジスタB

alphabet = {
    'q': int('10000000',2),
    'w': int('01000000',2),
    'e': int('00100000',2),
    'r': int('00010000',2),
    't': int('00001000',2),
    'y': int('00000100',2),
    'g': int('00000010',2),
    'f': int('00000001',2),
    'd': int('10000000',2),
    's': int('01000000',2),
    'a': int('00100000',2),
    'v': int('00010000',2),
    'c': int('00001000',2),
    'x': int('00000100',2),
    'z': int('00000010',2)
}
#alphabet = {
#    'q': 0x80,
#    'w': 0x40,
#    'e': 0x20, 
#    'r': 0x10,
#    't': 0x08,
#    'y': 0x04,
#    'g': 0x02, 
#    'f': 0x01, 
#    'd': 0x80,
#    's': 0x40,
#    'a': 0x20,
#    'v': 0x10,
#    'c': 0x08,
#    'x': 0x04,
#    'z': 0x02
#}

register = {
    'q': REG_OLAT_A,
    'w': REG_OLAT_A,
    'e': REG_OLAT_A,
    'r': REG_OLAT_A,
    't': REG_OLAT_A,
    'y': REG_OLAT_A,
    'g': REG_OLAT_A,
    'f': REG_OLAT_A,
    'd': REG_OLAT_B,
    's': REG_OLAT_B,
    'a': REG_OLAT_B,
    'v': REG_OLAT_B,
    'c': REG_OLAT_B,
    'x': REG_OLAT_B,
    'z': REG_OLAT_B
}


bus = smbus.SMBus(CHANNEL)
word = sys.argv[1]

# ピンの入出力設定
bus.write_byte_data(ICADDR_1, REG_OLAT_A, 0x00)
bus.write_byte_data(ICADDR_1, REG_OLAT_B, 0x00)

for i in word:
    bus.write_byte_data(ICADDR_1, register[i], alphabet[i])
    time.sleep(0.01)
    bus.write_byte_data(ICADDR_1, register[i], 0x00)
