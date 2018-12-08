# Check if the python modules can load properly
#Test importing numpy -- fails for earlier python version

def main():
  
  print('testPythonPath.py called')

  import os,sys

  try:
    import numpy
  except OSError:
    print('UNIT TEST FAILED: Cannot import numpy!');
    exit(0)

if __name__== "__main__":
  main()
