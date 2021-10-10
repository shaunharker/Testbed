import types
import numpy as np

def utf8encode(char_sequence):
    if type(char_sequence) == types.GeneratorType:
        def stream():
            for c in char_sequence:
                for b in bytes(c, encoding='utf8'):
                    yield b
        result = stream()
    else:
        result = bytes(char_sequence, encoding='utf8')
    return result

def utf8decode(byte_sequence):
    def is_valid_utf8_byte(b):
        return b&0b11111000 != 0b11111000
    def is_payload_utf8_byte(b):
        return b&0b11000000 == 0b10000000
    def is_header_utf8_byte(b):
        return is_valid_utf8_byte(b) and not is_payload_utf8_byte(b)
    def char_width(b):
        if b&0b10000000 == 0:
            return 1
        elif b&0b11100000 == 0b11000000:
            return 2
        elif b&0b11110000 == 0b11100000:
            return 3
        elif b&0b11111000 == 0b11110000:
            return 4
        return None
    def stream():
        (word, width) = ([], 0)
        for b in byte_sequence:
            if is_header_utf8_byte(b):
                (word, width) = ([b], char_width(b))
            elif is_payload_utf8_byte(b):
                word.append(b)
            if len(word) == width:
                try:
                    yield bytes(word).decode('utf8')
                except:
                    # There are still undecodables we catch here.
                    # e.g. bytes(map(lambda x: int(x,base=2),['0b11000000', '0b10000000'])).decode('utf8') raises UnicodeDecodeError
                    pass
    if type(byte_sequence) == types.GeneratorType:
        return stream()
    else:
        return ''.join(list(stream()))

def utf8bitsencode(char_seq: str):
    return np.unpackbits(np.frombuffer(bytes(char_seq, encoding='utf-8'), dtype=np.uint8),
        bitorder='little').tolist()

def utf8bitsdecode(bits):
    result = bytes()
    idx = 0
    while idx+7 < len(bits):
        if bits[idx+7] == 0:
            result += bytes([sum(2**i * bits[idx + i] for i in range(8))])
            idx += 8
        elif idx+15 < len(bits) and  (bits[idx+5] == 0 and bits[idx+6] == 1 and bits[idx+7] == 1 and
              bits[idx+14] == 0 and bits[idx+15] == 1):
            result += bytes([sum(2**i * bits[idx + i] for i in range(8)),
                       sum(2**i * bits[idx + i + 8] for i in range(8))])
            idx += 16
        elif idx+23 < len(bits) and (bits[idx+4] == 0 and bits[idx+5] == 1 and bits[idx+6] == 1 and bits[idx+7] == 1 and
              bits[idx+14] == 0 and bits[idx+15] == 1 and
              bits[idx+22] == 0 and bits[idx+23] == 1):
            result += bytes([sum(2**i * bits[idx + i] for i in range(8)),
                       sum(2**i * bits[idx + i + 8] for i in range(8)),
                       sum(2**i * bits[idx + i + 16] for i in range(8))])
            idx += 24
        elif idx+31 < len(bits) and (bits[idx+3] == 0 and bits[idx+4] == 1 and bits[idx+5] == 1 and bits[idx+6] == 1 and bits[idx+7] == 1 and
              bits[idx+14] == 0 and bits[idx+15] == 1 and
              bits[idx+22] == 0 and bits[idx+23] == 1 and
              bits[idx+30] == 0 and bits[idx+31] == 1):
            result += bytes([sum(2**i * bits[idx + i] for i in range(8)),
                       sum(2**i * bits[idx + i + 8] for i in range(8)),
                       sum(2**i * bits[idx + i + 16] for i in range(8)),
                       sum(2**i * bits[idx + i + 24] for i in range(8))])
            idx += 32
        else:
            idx += 1
    return result.decode('utf-8')
