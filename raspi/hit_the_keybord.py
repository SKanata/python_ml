#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import smbus2 as smbus
import time

CHANNEL    = 1      # i2c割り当てチャンネル 1 or 0
ICADDR_1    = 0x20   # スレーブ側ICアドレス1
ICADDR_2    = 0x21   # スレーブ側ICアドレス1
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
    'z': int('00000010',2),
    'u': int('10000000',2),
    'i': int('01000000',2),
    'o': int('00100000',2),
    'p': int('00010000',2),
    '-': int('00001000',2),
    'l': int('00000100',2),
    'k': int('00000010',2),
    'j': int('00000001',2),
    'h': int('00000001',2),
    'b': int('00000010',2),
    'n': int('00000100',2),
    'm': int('00001000',2),
    ',': int('00010000',2)
}

icaddr = {
    'q': ICADDR_2, 'w': ICADDR_2, 'e': ICADDR_2, 'r': ICADDR_2, 't': ICADDR_2,
    'y': ICADDR_2, 'g': ICADDR_2, 'f': ICADDR_2, 'd': ICADDR_2, 's': ICADDR_2,
    'a': ICADDR_2, 'v': ICADDR_2, 'c': ICADDR_2, 'x': ICADDR_2, 'z': ICADDR_2,
    'u': ICADDR_1, 'i': ICADDR_1, 'o': ICADDR_1, 'p': ICADDR_1, '-': ICADDR_1,
    'l': ICADDR_1, 'k': ICADDR_1, 'j': ICADDR_1, 'h': ICADDR_1, 'b': ICADDR_1,
    'n': ICADDR_1, 'm': ICADDR_1, ',': ICADDR_1
}
register = {
    'q': REG_OLAT_A, 'w': REG_OLAT_A, 'e': REG_OLAT_A, 'r': REG_OLAT_A, 't': REG_OLAT_A,
    'y': REG_OLAT_A, 'g': REG_OLAT_A, 'f': REG_OLAT_A, 'd': REG_OLAT_B, 's': REG_OLAT_B,
    'a': REG_OLAT_B, 'v': REG_OLAT_B, 'c': REG_OLAT_B, 'x': REG_OLAT_B, 'z': REG_OLAT_B,
    'u': REG_OLAT_A, 'i': REG_OLAT_A, 'o': REG_OLAT_A, 'p': REG_OLAT_A, '-': REG_OLAT_A,
    'l': REG_OLAT_A, 'k': REG_OLAT_A, 'j': REG_OLAT_A, 'h': REG_OLAT_B, 'b': REG_OLAT_B,
    'n': REG_OLAT_B, 'm': REG_OLAT_B, ',': REG_OLAT_B
}


bus = smbus.SMBus(CHANNEL)
word = sys.argv[1]

# ピンの入出力設定
bus.write_byte_data(ICADDR_1, REG_OLAT_A, 0x00)
bus.write_byte_data(ICADDR_1, REG_OLAT_B, 0x00)
bus.write_byte_data(ICADDR_2, REG_OLAT_A, 0x00)
bus.write_byte_data(ICADDR_2, REG_OLAT_B, 0x00)

for i in word:
    print('Hitting the key: %(word)s. ' % {'word': i})
    bus.write_byte_data(icaddr[i], register[i], alphabet[i])
    time.sleep(0.3)
    bus.write_byte_data(icaddr[i], register[i], 0x00)
#    time.sleep(0.1)
