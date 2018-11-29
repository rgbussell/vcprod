def hello():
	print('hello from' + __name__ +' in vc test module')

def greetModules():
	import vc as vc
	vc.test.hello()
	vc.run.hello()
	vc.fit.hello()
	vc.load.hello()
	vc.view.hello()


