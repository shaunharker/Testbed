import asyncio
import socket
from time import sleep
import numpy as np

# Shaun Harker, 2021-08-09

class FloatProtocol(asyncio.Protocol):
    def __init__(self, data):
        self.partial = bytes()
        self.data = data

    def connection_made(self, transport):
        #print("connection_made")
        self.transport = transport

    def data_received(self, bs):
        self.partial += bs
        L = len(self.partial)
        (x, self.partial) = (self.partial[:4*(L//4)], self.partial[4*(L//4):])
        self.data += np.frombuffer(x).tolist()

    def connection_lost(self, exc):
        #print("connection_lost")
        pass


class MagicList:
    def __init__(self):
        self.data = []
        sock, self.sock = socket.socketpair()
        loop = asyncio.get_running_loop()
        self.protocol = FloatProtocol(self.data)
        connection = loop.create_connection(protocol_factory=lambda: self.protocol, sock=sock)
        asyncio.create_task(connection)

    def __del__(self):
        self.protocol.transport.close()
        self.sock.close()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def __iadd__(self, fs):
        self.sock.send(np.array(fs).tobytes())
        return self

    def append(self, f):
        self.sock.send(np.array([f]).tobytes())
        return self

    def extend(self, fs):
        self.sock.send(np.array(fs).tobytes())
        return self
