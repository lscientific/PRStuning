import sys,os

class logger(object):
  def __init__(self, file):
    self.terminal = sys.stdout
    dir = os.path.dirname(file)
    if dir=='': dir='.'
    if not os.path.exists(dir):
      try:
        os.makedirs(os.path.dirname(file))
      except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST: raise
    self.log = open(file, "w")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)  

  def flush(self):
    #this flush method is needed for python 3 compatibility.
    #this handles the flush command by doing nothing.
    #you might want to specify some extra behavior here.
    self.log.flush()
    # typically the above line would do. however this is used to ensure that the file is written
    os.fsync(self.log.fileno())
    pass

  def __del__(self):
    self.log.close()
