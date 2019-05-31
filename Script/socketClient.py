import sys, ctypes, struct, threading, socket, re, time, logging
# from rapidjson import loads,dumps 

class Client(object):
	def __init__(self, endpoint):
		'''
		Parameters:
		endpoint: a tuple (ip, port)
		'''
		self.BUFSIZE = 4096
		self.endpoint = endpoint
		self.sender = None # if socket == None, means client is not connected

	def connect(self, timeout = 1):
		'''
		Try to connect to server, return whether connection successful
		'''
		if self.isConnected():
			return True
			
		try:
			# self.wait_connected.clear()
			s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			s.connect(self.endpoint)
			self.sender = s
			self.wfile = self.sender.makefile('wb', 0)
			time.sleep(0.1)
			return True

		except Exception as e:
			print('Can not connect to ', str(self.endpoint))
			print("Error", e)
			self.sender = None
			return False

	def isConnected(self):
		return self.sender is not None

	def disconnect(self):
		if self.isConnected():
			self.sender.shutdown(socket.SHUT_RD)
			if self.sender: 
				self.sender.close()
				self.sender = None
			time.sleep(0.1) # TODO, this is tricky

	def send(self, message):
		'''
		Send message out, return whether the message was successfully sent
		'''
		if self.isConnected():
			# print('Client: Send message ', message)
			try:
				self.wfile.write(message)
				self.wfile.flush()
				# wfile.close()
			except Exception as e:
				print('Fail to send message ', e) 
				return False
			return True
		else:
			print('Fail to send message, client is not connected')
			return False


# msg = "MoveForward"
# msg = msg+"\n"
# print(msg)
# endpoint = ("127.0.0.1",10020)
# client = Client(endpoint)
# client.connect()
# client.send(msg)
# client.disconnect()