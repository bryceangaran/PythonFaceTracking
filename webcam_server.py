import argparse
import socket
import sys
import threading
import time

import numpy as np
import cv2

HOST = '127.0.0.1'  # localhost
PORT = 12121
DEVICE = 0

CHUNKSIZE = 0x1000
ENCODING = 'utf-8'


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='A small socket server that transmits python OpenCV webcam images.')
    parser.add_argument(
        '-d', '--device',
        default=DEVICE,
        help='The webcam device to open with cv2.VideoCapture. Default: %(default)s'
    )
    parser.add_argument(
        '-a', '--address',
        default=HOST,
        help='The host address to which to bind the server. Default: %(default)s'
    )
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=PORT,
        help='The port to which to bind the server. Default: %(default)s'
    )
    parser.add_argument(
        '-c', '--client',
            action='store_true',
    help='If true, run as a client that connects to a server and displays the received images.'
  )
  parser.add_argument(
    '-s', '--size',
    help='Set the size of the image, e.g. "640x480".'
  )
  return parser.parse_args(args)



def capture(server):
  cap = server.cap
  while server.capture:
    ret, frame = cap.read()
    with server.lock:
      server.frame = frame



class WebcamServer:
  def __init__(self, device=DEVICE, host=HOST, port=PORT, size=None):
    self.device = device
    self.host = host
    self.port = port
    self.size = size

    self.cap = None
    self.socket = None
    self.listen = False
    self.capture = False

    self.lock = threading.Lock()
    self.frame = None
    self.capture_thread = None

  def start_capture(self):
    self.cap = cv2.VideoCapture(self.device)
    if self.size is not None:
      try:
        width, height = (int(s) for s in self.size.split('x', 1))
      except ValueError:
        pass
      else:
        print('setting size to {:d}x{:d}'.format(width, height))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    self.capture = True
    self.capture_thread = threading.Thread(target=capture, args=(self,))
    self.capture_thread.start()

  def stop_capture(self):
    self.capture = False
    self.capture_thread.join()
    self.cap.release()

  def start_listening(self):
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print('binding', self.host, self.port)
    self.socket.bind((self.host, self.port))
    self.socket.listen(1)
    self.listen = True

  def stop_listening(self):
    self.listen = False
    self.socket.close()

  def __enter__(self):
    self.start_capture()
    self.start_listening()
    return self.cap, self.socket

  def __exit__(self, typ, val, traceback):
    self.stop_listening()
    self.stop_capture()

  def serve(self):
    with self as (cap, sock):
      while self.listen and self.capture:
        clientsock, address = self.socket.accept()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print('[{}] accepted connection from {}'.format(timestamp, address))
        try:
          frame = self.frame
          for dim in frame.shape:
            clientsock.send(str(dim).encode(ENCODING))
            clientsock.send(b'\n')
          clientsock.send(frame.tobytes())
        except (TypeError, AttributeError):
          pass
        finally:
          clientsock.close()

  def retrieve(self):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((self.host, self.port))
    bs = bytes()
    try:
      chunk = sock.recv(CHUNKSIZE)
      while chunk:
        bs += chunk
        chunk = sock.recv(CHUNKSIZE)
    finally:
      sock.close()
    w, h, d, dat = bs.split(b'\n', 3)
    w, h, d = (int(x.decode()) for x in (w, h, d))
    img = np.frombuffer(dat, dtype=np.uint8)
    return img.reshape(w, h, d)



if __name__ == '__main__':
  pargs = parse_args()
  try:
    ws = WebcamServer(
      device=pargs.device,
      host=pargs.address,
      port=pargs.port,
      size=pargs.size
    )
    if pargs.client:
      while True:
        arr = ws.retrieve()
        cv2.imshow('Received Image', arr)
        cv2.waitKey(0)
    else:
      ws.serve()
  except KeyboardInterrupt:
    ws.listen = False
    cv2.destroyAllWindows()
    sys.exit()
  except socket.error as e:
    sys.exit('socket error: {}'.format(e))
