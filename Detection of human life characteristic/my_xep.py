import sys
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy.signal as signal
from matplotlib.animation import FuncAnimation
from pymoduleconnector import ModuleConnector
import time
from time import sleep

__version__ = 3

class data_cache:
	def __init__(self,buffer_size=100):
		self.buffer = []
		self.buffer_size = buffer_size
	
	def add(self,new_frame):
		if len(self.buffer)+1 > self.buffer_size:
			self.buffer = self.buffer[1:]
		self.buffer.append(new_frame)

class data_process:
	def __init__(self,fps):
		self.fps = fps
		self.maxDoor = -1
		self.meanValue = 0
		self.meanValueState = 0 #0代表还未计算第一次均值
		self.preFirstFrame = []
		self.result_data = [] #最大距离门信号
	
	def zero_mean(self,frame_data,buffer_size):
		if not self.meanValueState:
			self.meanValue = np.mean(frame_data,1)
			self.meanValueState = 1
		else:
			self.meanValue += (frame_data[:,-1]-self.preFirstFrame)/buffer_size
		mean_data = np.transpose(np.tile(self.meanValue,(buffer_size,1)))
		frame_data_new = frame_data - mean_data
		return frame_data_new
	
	def signal_divide(self,signal_data):
		#简单利用带通滤波器来对信号进行分离
		fl,fh = 0.1,0.5
		fll,fhh = 0.05,0.55
		pass_band = [2*fl/self.fps,2*fh/self.fps]
		stop_band = [2*fll/self.fps,2*fhh/self.fps]
		b,a = signal.iirdesign(pass_band,stop_band,2,40)
		r_signal = signal.lfilter(b,a,signal_data)
		
		fl_h,fh_h = 0.8,1.9
		fll_h,fhh_h = 0.7,2.0
		pass_band_h = [2*fl_h/self.fps,2*fh_h/self.fps]
		stop_band_h = [2*fll_h/self.fps,2*fhh_h/self.fps]
		b_h,a_h = signal.iirdesign(pass_band_h,stop_band_h,2,40)
		h_signal = signal.lfilter(b_h,a_h,signal_data)
		return r_signal,h_signal


	def freq_get(self,frame_data,fI,start_point,mid_point,end_point,buffer_size):
		self.preFirstFrame = frame_data[:,0]
		frame_data = self.zero_mean(frame_data,buffer_size)
		if self.maxDoor<0:
			sum_frame = np.sum(frame_data**2,1)
			#print (sum_frame)
			self.maxDoor = np.argmax(sum_frame)
			print (self.maxDoor)
			# fig1,ax1 = plt.subplots()
			# im=ax1.imshow(frame_data)
			# fig1.colorbar(im)
			# plt.show()
			# plt.plot(frame_data[self.maxDoor,:])
		self.result_data = frame_data[self.maxDoor,:]
		spectrum_data = abs(fft(self.result_data))
		max_point_rf = np.argmax(spectrum_data[start_point:mid_point])
		max_point_hf = np.argmax(spectrum_data[mid_point:end_point])
		return fI[start_point+max_point_rf],fI[mid_point+max_point_hf]

class Connect_xep:

	def __init__(self,fps,dac_min,dac_max,frame_area,area_offset,device,baseband=False):
		self.fps = fps
		self.dac_min = dac_min
		self.dac_max = dac_max
		self.frame_area = frame_area
		self.area_offset = area_offset
		self.baseband = baseband
		self.device_name = device

	def reset(self,device_name):
		mc = ModuleConnector(device_name)
		r = mc.get_xep()
		r.module_reset()
		mc.close()
		sleep(3)
		
	def set_parameters(self,mc):
		# Assume an X4M300/X4M200 module and try to enter XEP mode
		app = mc.get_x4m300()
		# Stop running application and set module in manual mode.
		try:
			app.set_sensor_mode(0x13, 0) # Make sure no profile is running.
		except RuntimeError:
			# Profile not running, OK
			pass

		try:
			app.set_sensor_mode(0x12, 0) # Manual mode.
		except RuntimeError:
			# Sensor already stopped, OK
			pass
			
		r = mc.get_xep()
		# Set DAC range
		r.x4driver_set_dac_min(self.dac_min)
		r.x4driver_set_dac_max(self.dac_max)

		# Set integration
		r.x4driver_set_iterations(16)
		r.x4driver_set_pulses_per_step(26)
		
		r.x4driver_set_frame_area(self.frame_area[0],self.frame_area[1])
		r.x4driver_set_frame_area_offset(self.area_offset)
		
		#判断时采集基带信号还是频带信号
		if self.baseband:
			r.x4driver_set_downconversion(1)
		else:
			r.x4driver_set_downconversion(0)
		
		# Start streaming of data
		r.x4driver_set_fps(self.fps)
		
		return r
	
	def clear_buffer(self,r):
		"""Clears the frame buffer"""
		while r.peek_message_data_float():
			_=r.read_message_data_float()
	
	def read_frame(self,r):
		"""Gets frame data from module"""
		d = r.read_message_data_float()
		frame = np.array(d.data)
		 # Convert the resulting frame to a complex array if downconversion is enabled
		if self.baseband:
			n=len(frame)
			frame = frame[:n//2] + 1j*frame[n//2:]
		return frame
	
	def animate(self,i):
		if self.baseband:
			self.line.set_ydata(abs(self.read_frame(self.r)))  # update the data
		else:
			self.line.set_ydata(self.read_frame(self.r))  # update the data
		return self.line,
	
	def cache_load(self,new_frame):
		if len(self.all_frame) <100:
			self.all_frame.append(new_frame)
		else:
			self.all_frame = self.all_frame[1:]+[new_frame]

	def simple_xep_operate(self):
		
		self.reset(self.device_name)
		mc = ModuleConnector(self.device_name)
		self.r=self.set_parameters(mc)
		
		fig = plt.figure()
		fig.suptitle("simple_xep_plot version %d. Baseband = %r"%(__version__, self.baseband))
		ax = fig.add_subplot(1,1,1)
		frame = self.read_frame(self.r)

		if self.baseband:
			frame = abs(frame)

		self.line, = ax.plot(frame)
		
		self.clear_buffer(self.r)
		
		ani = FuncAnimation(fig, self.animate, interval=self.fps)
		plt.show()
		
		#存储数据
		buffer_size = 200
		fI=np.linspace(0,self.fps-self.fps/buffer_size,buffer_size)
		start_freq,mid_freq,end_freq = 0.1,0.8,2 #体征信号频率范围为0.1Hz-2Hz
		start_point,mid_point,end_point = int(start_freq*buffer_size/self.fps),int(mid_freq*buffer_size/self.fps),int(end_freq*buffer_size/self.fps)
		print (start_point,end_point)
		myBuffer = data_cache(buffer_size)
		vital_detection = data_process(self.fps)
		iter = 1;
		while iter<210:
			new_frame = abs(self.read_frame(self.r))
			new_frame = list(new_frame)
			myBuffer.add(new_frame)
			if iter%50==0:
				print (len(myBuffer.buffer))
			#time.sleep(self.fps/10000)
			if iter>buffer_size:
				freq_rf,freq_hf = vital_detection.freq_get(np.transpose(np.array(myBuffer.buffer)),fI,start_point,mid_point,end_point,buffer_size)
				print ('Rf is :',freq_rf,freq_hf)
			iter += 1
		
		frame_data = np.transpose(np.array(myBuffer.buffer))
		[m,n]=frame_data.shape
		mean_data = np.transpose(np.tile(np.mean(frame_data,1),(n,1)))
		frame_data_new = frame_data-mean_data

		fig1,ax1 = plt.subplots()
		print ('frame_data_shape:',frame_data_new.shape)
		im=ax1.imshow(frame_data_new)
		fig1.colorbar(im)
		plt.show()
		
		# Stop streaming of data
		self.r.x4driver_set_fps(0)
		return frame_data,mean_data,frame_data_new,vital_detection.maxDoor,fI

if __name__ == '__main__':
	device = 'COM8'
	baseband = True
	fps = 10 #慢采样率
	dac_min = 900
	dac_max = 1150
	frame_area = [0.2,5]
	area_offset = 0.2
	baseband = True  # True-基带信号 False-频带信号
	xep_operate = Connect_xep(fps,dac_min,dac_max,frame_area,area_offset,device,baseband)
	frame_data,mean_data,frame_data_new,maxDoor,fI= xep_operate.simple_xep_operate()
	data_process1 = data_process(fps)
	r_signal,h_signal = data_process1.signal_divide(frame_data_new[maxDoor,:])