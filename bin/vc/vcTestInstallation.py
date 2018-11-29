#!/opt/ds/bin/python

#This is the wrapper to launch the VC analysis code
#Robert Bussell December 2018

import os
thisFn=os.path.realpath(__file__)

vcModulePath='/opt/ds/lib/'

print('hello from '+__name__)

print('.loading vc module from ' + vcModulePath )
import sys
sys.path.append(vcModulePath)

import vc as vc

def checkForTestModule():
	vc.test.greetModules()

def cleanUp(retVal=1):
	print(thisFn + ' exiting. Nice knowing ya')
	return retVal

vc.test.greetModules()
cleanUp()
