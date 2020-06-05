import sys, ctypes, struct, threading, socket, re, time, logging
from rapidjson import loads,dumps 
from multiprocessing import Process

class Server(object):
	def __init__(self,endpoint):
		self.BUFSIZE = 30000000
		self.endpoint = endpoint
		self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.listener.bind(endpoint)
		self.listener.settimeout(30.0)
		self.conn = None
		self.receiving_thread = None
		self.connected = False
		self.buf = ""
		self.delim = "\nEND OF FILE\n"
		self.bufblock = False
		# socket.setdefaulttimeout(3)

	def listen(self):
		print("Server starts to listen!")
		self.listener.listen(self.BUFSIZE)
		self.conn, addr = self.listener.accept()
		print("Server accepts a connection!")
		if not self.conn:
			return False
		self.receiving_thread = threading.Thread(target = self.__receiving)
		# self.receiving_thread = Process(target = self.__receiving)
		self.receiving_thread.daemon = True
		self.receiving_thread.start()
		# self.receiving_thread.join()
		return True

	def __receiving(self):
		print("Server start to receive data!")
		while True:
			# print("thread running")
			try:
				if not self.conn:
					print("Connection break")
					break
				data = self.conn.recv(self.BUFSIZE)
				if data == '' or data == None:
					continue
				# else:
					# print("Received data", data)
				while self.bufblock == True:
					pass
				self.buf += data
			except Exception:
				continue
			
			# print("Received data", data)
			# if "Connected" in data:
				# self.connected = True

			# Overwrite buffer only when incoming data is not none
			
			# print("data in buffer", self.buf)
		print("thread ending")

	def getBuffer(self, timeout = 3):
		start_time = time.time()
		while True:
			buf = self.buf
			if buf and self.delim in buf:
				self.bufblock = True
				self.buf = buf.split(self.delim, 1)[1]
				self.bufblock = False
				# print(buf.split(delim, 1)[0])
				break
				# print("remaining", self.buf)
			elif time.time()-start_time > timeout:
				break
		return buf.split(self.delim, 1)[0]

	def isConnected(self, timeout=0.1):
		start = time.time()
		while True:	
			buf = self.getBuffer()
			if not buf:
				continue
			if "Connected" in buf:
				self.connected = True
				return True
			end = time.time()
			if end - start > timeout:
				return False


	def stop(self):
		self.listener.close()
		self.listener = None
		if not self.conn:
			return False
		self.conn.close()
		self.conn = None
		self.buf = None
		return True



# endpoint = ("127.0.0.1", 10021)
# server = Server(endpoint)
# server.listen()
# server.stop()
# print(server.isReceiving())

