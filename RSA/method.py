from ctypes import *
import os
import numpy as np
#import seaborn
###################
#引入包缩减
import json
import pandas
from pandas import DataFrame
###################
import datetime
import time
import pymysql as mdb
#import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from decimal import Decimal
import serial 



#读取系统默认参数
def read_config():
	with open(os.getcwd()+'\\wxpython.json','r') as data:
		config_data = json.load(data)
	return config_data
	
def read_direction_cache():
	with open(os.getcwd()+'\\direction.json','r') as data:
		direction_cache = json.load(data)
	return direction_cache
	
def read_reflect_inf():
	with open(os.getcwd()+'\\reflect_inf.json','r') as data:
		reflect_inf = json.load(data)
	return reflect_inf



config_data=read_config()
direction_cache=read_direction_cache()
reflect_inf=read_reflect_inf()
print ('reflect_inf:',reflect_inf)

def compute_distance(p1,p2):
	R=6378137;
	dx=(p2[0]-p1[0])*R*np.cos(p1[1])
	dy=(p2[1]-p1[1])*R*np
	return np.sqrt(dx*dx+dy*dy)

def read_mysql_config():
	with open(os.getcwd()+'\\mysql.json','r') as data:
		mysql_config=json.load(data)
	return mysql_config

mysql_config=read_mysql_config()
###########################################################################
## draw pictures
###########################################################################
def draw_picture(x,y,title='',xlabel='',ylabel='',height=4,width=6,face_color='k',gridcolor='y',figure=None):
	if figure==None:
		figure_score = Figure()
	else:
		figure_score=figure
	figure_score.set_figheight(height)
	figure_score.set_figwidth(width)
	axes_score = figure_score.add_subplot(111,facecolor=face_color)

	#if len(x)!=0 and len(y)!=0:
	l_user,=axes_score.plot(x, y, 'b')
	axes_score.set_title(title)
	axes_score.grid(True,color=gridcolor)
	axes_score.set_xlabel(xlabel)
	axes_score.set_ylabel(ylabel)
	# axes_score.set_xlim(xlim[0],xlim[1])
	# axes_score.set_ylim(ylim[0],ylim[1])
	#FigureCanvas(self.scorePanel, -1, self.figure_score)
	return figure_score,axes_score,l_user
	

# search/connect variables
def instrument_connect():
	os.chdir(os.getcwd())
	rsa = cdll.LoadLibrary("RSA_API.dll")
	numFound = c_int(0)
	intArray = c_int * 10
	deviceIDs = intArray()
	deviceSerial = create_string_buffer(8)
	deviceType = create_string_buffer(8)
	rsa.DEVICE_Search(byref(numFound), deviceIDs, deviceSerial, deviceType)
	message=[]  #return the device_type and Serial_number
	if numFound.value < 1:
		print('No instruments found . Exiting script.')
		return [0,rsa,message]
	elif numFound.value == 1:
		print('One device found.')
		print('Device type:{}'.format(deviceSerial.value))
		rsa.DEVICE_Connect(deviceIDs[0])
		print('The device has connected')
		message.append(deviceType.value)
		message.append(deviceSerial.value)
		return [1,rsa,message]
	else:
		print('Unexpected number of devices found, exiting script.')
		return [2,rsa,message]
    # return deviceSerial.value

#disconnect variables
def instrument_disconnect(rsa):
    rsa.Disconnect()

'''
# 噪声监测
# def detectNoise(rsa300):

   # # create Spectrum_Settings data structure
   # class Spectrum_Settings(Structure):
      # _fields_ = [('span', c_double),
                  # ('rbw', c_double),
                  # ('enableVBW', c_bool),
                  # ('vbw', c_double),
                  # ('traceLength', c_int),
                  # ('window', c_int),
                  # ('verticalUnit', c_int),
                  # ('actualStartFreq', c_double),
                  # ('actualStopFreq', c_double),
                  # ('actualFreqStepSize', c_double),
                  # ('actualRBW', c_double),
                  # ('actualVBW', c_double),
                  # ('actualNumIQSamples', c_double)]

   # # initialize variables
   # specSet = Spectrum_Settings()
   # longArray = c_long * 10
   # deviceIDs = longArray()
   # deviceSerial = c_wchar_p('')
   # numFound = c_int(0)
   # enable = c_bool(True)  # spectrum enable
   # cf = c_double(20e6)  # center freq
   # refLevel = c_double(0)  # ref level
   # ready = c_bool(False)  # ready
   # timeoutMsec = c_int(500)  # timeout
   # trace = c_int(0)  # select Trace 1
   # detector = c_int(0)  # set detector type to max

   # # preset the RSA306 and configure spectrum settings
   # rsa300.Preset()
   # rsa300.SetCenterFreq(cf)
   # rsa300.SetReferenceLevel(refLevel)
   # rsa300.SPECTRUM_SetEnable(enable)
   # rsa300.SPECTRUM_SetDefault()
   # rsa300.SPECTRUM_GetSettings(byref(specSet))

   # # configure desired spectrum settings
   # # some fields are left blank because the default
   # # values set by SPECTRUM_SetDefault() are acceptable
   # specSet.span = c_double(40e6)
   # specSet.rbw = c_double(300e3)
   # specSet.enableVBW = c_bool(True)
   # specSet.vbw = c_double(300e3)
   # specSet.traceLength = c_int(801)
   # specSet.detector = detector
   # # specSet.window =
   # # specSet.verticalUnit =
   # specSet.actualStartFreq = c_double(0)
   # specSet.actualStopFreq = c_double(40e6)
   # # specSet.actualFreqStepSize =c_double(50000.0)
   # # specSet.actualRBW =
   # # specSet.actualVBW =
   # # specSet.actualNumIQSamples =

   # # set desired spectrum settings
   # rsa300.SPECTRUM_SetSettings(specSet)
   # rsa300.SPECTRUM_GetSettings(byref(specSet))

   # # uncomment this if you want to print out the spectrum settings


   # # print out spectrum settings for a sanity check
   # # print('Span: ' + str(specSet.span))
   # # print('RBW: ' + str(specSet.rbw))
   # # print('VBW Enabled: ' + str(specSet.enableVBW))
   # # print('VBW: ' + str(specSet.vbw))
   # # print('Trace Length: ' + str(specSet.traceLength))
   # # print('Window: ' + str(specSet.window))
   # # print('Vertical Unit: ' + str(specSet.verticalUnit))
   # # print('Actual Start Freq: ' + str(specSet.actualStartFreq))
   # # print('Actual End Freq: ' + str(specSet.actualStopFreq))
   # # print('Actual Freq Step Size: ' + str(specSet.actualFreqStepSize))
   # # print('Actual RBW: ' + str(specSet.actualRBW))
   # # print('Actual VBW: ' + str(specSet.actualVBW))

   # # initialize variables for GetTrace
   # traceArray = c_float * specSet.traceLength
   # traceData = traceArray()
   # outTracePoints = c_int()

   # # generate frequency array for plotting the spectrum
   # freq = np.arange(specSet.actualStartFreq,
                    # specSet.actualStartFreq + specSet.actualFreqStepSize * specSet.traceLength,
                    # specSet.actualFreqStepSize)

   # # start acquisition
   # rsa300.Run()
   # while ready.value == False:
      # rsa300.SPECTRUM_WaitForDataReady(timeoutMsec, byref(ready))

   # rsa300.SPECTRUM_GetTrace(c_int(0), specSet.traceLength,
                            # byref(traceData), byref(outTracePoints))
   # # print('Got trace data.')

   # # convert trace data from a ctypes array to a numpy array
   # trace = np.ctypeslib.as_array(traceData)

   # # Peak power and frequency calculations
   # average = sum(trace) / len(trace)
   # # print('Disconnecting.')
   # # rsa300.Disconnect()
   # return average
'''
def detectNoise(rsa300,startFreq,endFreq,rbw,vbw):


   # create Spectrum_Settings data structure
   class Spectrum_Settings(Structure):
      _fields_ = [('span', c_double),
                  ('rbw', c_double),
                  ('enableVBW', c_bool),
                  ('vbw', c_double),
                  ('traceLength', c_int),
                  ('window', c_int),
                  ('verticalUnit', c_int),
                  ('actualStartFreq', c_double),
                  ('actualStopFreq', c_double),
                  ('actualFreqStepSize', c_double),
                  ('actualRBW', c_double),
                  ('actualVBW', c_double),
                  ('actualNumIQSamples', c_double)]

   # initialize variables
   specSet = Spectrum_Settings()
   longArray = c_long * 10
   deviceIDs = longArray()
   deviceSerial = c_wchar_p('')
   numFound = c_int(0)
   enable = c_bool(True)  # spectrum enable
   cf = c_double(startFreq+(endFreq-startFreq)/2.0)  # center freq
   refLevel = c_double(0)  # ref level
   ready = c_bool(False)  # ready
   timeoutMsec = c_int(500)  # timeout
   trace = c_int(0)  # select Trace 1
   detector = c_int(1)  # set detector type to max

   # preset the RSA306 and configure spectrum settings
   rsa300.Preset()
   rsa300.SetCenterFreq(cf)
   rsa300.SetReferenceLevel(refLevel)
   rsa300.SPECTRUM_SetEnable(enable)
   rsa300.SPECTRUM_SetDefault()
   rsa300.SPECTRUM_GetSettings(byref(specSet))

   # configure desired spectrum settings
   # some fields are left blank because the default
   # values set by SPECTRUM_SetDefault() are acceptable
   specSet.span = c_double(endFreq-startFreq)
   specSet.rbw = c_double(rbw)
   specSet.enableVBW = c_bool(True)
   specSet.vbw = c_double(vbw)
   specSet.traceLength = c_int(801)
   #specSet.SpectrumVerticalUnits=c_int(4)
   # specSet.window =
   specSet.verticalUnit =c_int(4)
   specSet.actualStartFreq = c_double(0)
   specSet.actualStopFreq = c_double(40e6)
   # specSet.actualFreqStepSize =c_double(50000.0)
   # specSet.actualRBW =
   # specSet.actualVBW =
   # specSet.actualNumIQSamples =

   # set desired spectrum settings
   rsa300.SPECTRUM_SetSettings(specSet)
   rsa300.SPECTRUM_GetSettings(byref(specSet))

   # uncomment this if you want to print out the spectrum settings


   # print out spectrum settings for a sanity check
   # print('Span: ' + str(specSet.span))
   # print('RBW: ' + str(specSet.rbw))
   # print('VBW Enabled: ' + str(specSet.enableVBW))
   # print('VBW: ' + str(specSet.vbw))
   # print('Trace Length: ' + str(specSet.traceLength))
   # print('Window: ' + str(specSet.window))
   # print('Vertical Unit: ' + str(specSet.verticalUnit))
   # print('Actual Start Freq: ' + str(specSet.actualStartFreq))
   # print('Actual End Freq: ' + str(specSet.actualStopFreq))
   # print('Actual Freq Step Size: ' + str(specSet.actualFreqStepSize))
   # print('Actual RBW: ' + str(specSet.actualRBW))
   # print('Actual VBW: ' + str(specSet.actualVBW))

   # initialize variables for GetTrace
   traceArray = c_float * specSet.traceLength
   traceData = traceArray()
   outTracePoints = c_int()

   # generate frequency array for plotting the spectrum
   freq = np.arange(specSet.actualStartFreq,
                    specSet.actualStartFreq + specSet.actualFreqStepSize * specSet.traceLength,
                    specSet.actualFreqStepSize)

   # start acquisition
   rsa300.Run()
   while ready.value == False:
      rsa300.SPECTRUM_WaitForDataReady(timeoutMsec, byref(ready))

   rsa300.SPECTRUM_GetTrace(c_int(0), specSet.traceLength,
                            byref(traceData), byref(outTracePoints))
   # print('Got trace data.')

   # convert trace data from a ctypes array to a numpy array
   trace = np.ctypeslib.as_array(traceData)

   # Peak power and frequency calculations
   min_peak = min(trace)
   threshold = min_peak + 5
   # Peak power and frequency calculations
   trace1 = [data for data in trace if data < threshold]
   ave = np.mean(trace1)
   return ave

#预测带宽
def bandwidth(peakPower,peakFreq,trace,freq):
	peak_index1 = np.argmax(trace)
	peak_index2 = np.argmax(trace) 
	a1 = 0
	a2 = len(trace) - 1
	#print ('1:',a1,a2)
	while peak_index1>0:
		if trace[peak_index1]>peakPower-3:
			pass
		else:
			a2 = peak_index1
			break
		peak_index1 -= 1
	while peak_index2<len(trace):
		if trace[peak_index2]>peakPower-3:
			pass
		else:
			a1 = peak_index2
			break
		peak_index2 += 1
	
	# print ('2:',a1,a2)
	# peak_index3=a2
	# peak_index4=a1
	# while peak_index4<a2:
		# if trace[peak_index4]>trace[a1]+3:
			# break;
		# peak_index4+=1
	# while peak_index3>a1:
		# if trace[peak_index3]>trace[a2]+3:
			# break;
		# peak_index3-=1

	# a2=peak_index3
	# a1=peak_index4
	# print ('3:',a1,a2)

	bandWidth = freq[a1] - freq[a2]
	freq_cf = freq[a1]+bandWidth/2
	return freq_cf, bandWidth

#预测带宽2
def bandwidth2(peakPower,peakNum,trace,freq):
	peak_index1 = peakNum
	peak_index2 = peakNum
	#print (trace,freq)
	a1 = 0
	a2 = len(trace) - 1
	while peak_index1>0:
		if trace[peak_index1]>peakPower-6:
			pass
		else:
			a1 = peak_index1
			break
		peak_index1 -= 1
	#print ('peak_index1:',peak_index1)
	while peak_index2<len(trace):
		if trace[peak_index2]>peakPower-6:
			pass
		else:
			a2 = peak_index2
			break
		peak_index2 += 1

	peak_index3=a2
	peak_index4=a1
	while peak_index4<=a2:
		if trace[peak_index4]>trace[a1]+3:
			break;
		peak_index4+=1
	while peak_index3>=a1:
		if trace[peak_index3]>trace[a2]+3:
			break;
		peak_index3-=1

	a2=peak_index3
	a1=peak_index4
	#print ('peak_index2:',peak_index2)
	bandWidth = freq[a2] - freq[a1]
	freq_cf = freq[a1]+bandWidth/2
	return freq_cf, bandWidth

#预测带宽3
def bandwidth3(peakPower,peakFreq,trace,freq):
	trace_max=max(trace)
	a1 = 0
	a2 = len(trace) - 1
	peak_index1 = a2
	peak_index2 = a1
	#print ('1:',a1,a2)
	while peak_index1>0:
		if trace[peak_index1]<trace_max-6:
			pass
		else:
			a2 = peak_index1
			break
		peak_index1 -= 1
	while peak_index2<len(trace):
		if trace[peak_index2]<trace_max-6:
			pass
		else:
			a1 = peak_index2
			break
		peak_index2 += 1

	bandWidth = freq[a2] - freq[a1]
	freq_cf = freq[a1]+bandWidth/2
	return freq_cf, bandWidth


'''
# # 频谱监测（总） 输入参数是监测设备信息，天线信息，起始频率，终止频率，频率跨度，rbw，持续时间
# def spectrum2(rsa300,deviceSerial, span , t ,anteid, startFreq, stopFreq,rbw):
	# # 进行连接获得检测设备信息
	# # print (average)
	# # 参数设置
	# # 获取并设置设置起始频率和终止频率
	# startFreq = c_double(float(startFreq))
	# stopFreq = c_double(float(stopFreq))
	# # set span
	# span = c_double(float(span))
	# # 设置rbw
	# rbw = c_double(float(rbw))
	# # 从别的数据库读取出所选用的天线类型
	# # 设置step_size
	# # step_size = c_double(float(input()))
	# t = float(t)  # 持续时间
	# time_ref = time.time()
	# # 本次扫描的全部数据存储
	# trace = DataFrame({})
	# restf = []
	# restp = []
	# count = 0
	# 参数配置完之后就可以开始进行数据库的创建，之后具体信号的检测得到的数据在
	# 之后的程序中写入表中即可
	# # 先建立主表即测试任务，起始时间，持续时间
	# str_time = str(datetime.datetime.now().strftime('%F-%H-%M-%S'))  # 整个测试的起始的准确时间用于存储原始文件路径的因为路径中不能有：
	# start_time = str(datetime.datetime.now().strftime('%F %H:%M:%S')) # 总表的起始时间
	# while time.time() - time_ref < t:
		# average = detectNoise(rsa300)
		# count += 1
		# str_tt1 = str(datetime.datetime.now().strftime('%F %H:%M:%S'))  # 内部扫频的时刻
		# str_tt2 = str(datetime.datetime.now().strftime('%F-%H-%M-%S'))  # 作为内部细扫的文件名
		# z1, z2, z3, draw_Spectrum_total,Sub_Spectrum= spectrum1(rsa300,average, startFreq, stopFreq, span, rbw, str_time, count, str_tt1, str_tt2)
		# trace = pandas.concat([trace, z1])
		# restf.append(z2)
		# restp.append(z3)
		# # 主检测页面显示无人机频谱监测
		# #uav00() #无人机的先放在一边

	# # 获取测试时间,保存原始频谱数据
	# path = os.getcwd()+"\\data111\\" + str_time + "spectrum%ds.csv" % t  # 频谱数据粗扫描数据存储路径
	# trace.to_csv(
		# path,
		# index=False
	# )
	# str_time1 = str(datetime.datetime.now().strftime('%F %H:%M:%S'))  # 结束的准确时间，直接传到数据库中自动转化成datetime格式
	# s_c = spectrum_occ(start_time, str_time1, str_time, startFreq.value, stopFreq.value)
	# print('12121212')
	# print(s_c)
	# # 数据库构建
	# con = mdb.connect('localhost', 'root', 'cdk120803', 'ceshi1')
	# with con:
		# # 获取连接的cursor，只有获取了cursor，我们才能进行各种操作
		# cur = con.cursor()  # 一条游标操作就是一条数据插入，第二条游标操作就是第二条记录，所以最好一次性插入或者日后更新也行
		# print ([str_time, start_time, str_time1, float(t), float(s_c), path, deviceSerial, anteid, count])
		# cur.execute("INSERT INTO Minitor_Task VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",[str_time, start_time, str_time1, float(t), float(s_c), path, deviceSerial, anteid, count])
		# cur.close()
	# con.commit()
	# con.close()
	# return draw_Spectrum_total,Sub_Spectrum
'''
# 一次扫频参数：噪声均值、起始频率、终止频率、频率跨度、rbw、任务名、计数、某一次扫频的具体时间
def spectrum1(rsa300,average, startFreq, stopFreq, span, rbw,vbw, str_time, count, str_tt1, str_tt2,longitude,latitude,num_signal,Sub_cf_all):
	# create Spectrum_Settings data structure
	class Spectrum_Settings(Structure):
		_fields_ = [('span', c_double),
					('rbw', c_double),
					('enableVBW', c_bool),
					('vbw', c_double),
					('traceLength', c_int),
					('window', c_int),
					('verticalUnit', c_int),
					('actualStartFreq', c_double),
					('actualStopFreq', c_double),
					('actualFreqStepSize', c_double),
					('actualRBW', c_double),
					('actualVBW', c_double),
					('actualNumIQSamples', c_double)]

	# initialize variables
	specSet = Spectrum_Settings()
	longArray = c_long * 10
	deviceIDs = longArray()
	deviceSerial = c_wchar_p('')
	numFound = c_int(0)
	enable = c_bool(True)  # spectrum enable
	# cf = c_double(9e8)            #center freq
	refLevel = c_double(0)  # ref level
	ready = c_bool(False)  # ready
	timeoutMsec = c_int(500)  # timeout
	trace = c_int(0)  # select Trace 1
	detector = c_int(0)  # set detector type to max
	# 由起始频率和终止频率直接可以得到中心频率
	# set cf
	cf = c_double((startFreq.value + stopFreq.value) / 2)
	'''
	# search the USB 3.0 bus for an RSA306
	ret = rsa300.Search(deviceIDs, byref(deviceSerial), byref(numFound))
	if ret != 0:
		print('Error in Search: ' + str(ret))
	if numFound.value < 1:
		print('No instruments found. Exiting script.')
		exit()
	elif numFound.value == 1:
		print('One device found.')
		print('Device Serial Number: ' + deviceSerial.value)
	else:
		print('2 or more instruments found.')
		# note: the API can only currently access one at a time

	# connect to the first RSA306
	ret = rsa300.Connect(deviceIDs[0])
	if ret != 0:
		print('Error in Connect: ' + str(ret))
	'''
	# preset the RSA306 and configure spectrum settings
	rsa300.Preset()
	rsa300.SetCenterFreq(cf)
	rsa300.SetReferenceLevel(refLevel)
	rsa300.SPECTRUM_SetEnable(enable)
	rsa300.SPECTRUM_SetDefault()
	rsa300.SPECTRUM_GetSettings(byref(specSet))

	# configure desired spectrum settings
	# some fields are left blank because the default
	# values set by SPECTRUM_SetDefault() are acceptable
	specSet.span = span
	specSet.rbw = c_double(float(rbw))
	specSet.enableVBW = c_bool(True)
	specSet.vbw = c_double(float(vbw))
	specSet.traceLength = c_int(801)#c_int(int(span.value/step_size.value))#c_int(801)
	specSet.detector = detector
	# specSet.window =
	specSet.verticalUnit = c_int(4)
	specSet.actualStartFreq = startFreq
	specSet.actualStopFreq = stopFreq
	specSet.actualFreqStepSize = c_double(span.value/801)#step_size c_double(span.value/801)   # c_double(50000.0)
	# specSet.actualRBW =
	# specSet.actualVBW =
	# specSet.actualNumIQSamples =
	# set desired spectrum settings
	rsa300.SPECTRUM_SetSettings(specSet)
	rsa300.SPECTRUM_GetSettings(byref(specSet))

	# initialize variables for GetTrace
	traceArray = c_float * specSet.traceLength
	traceData = traceArray()
	outTracePoints = c_int()
	#print (span.value)
	#print (specSet.actualFreqStepSize)
	# generate frequency array for plotting the spectrum
	freq = np.arange(specSet.actualStartFreq,
					 specSet.actualStartFreq + specSet.actualFreqStepSize * specSet.traceLength,
					 specSet.actualFreqStepSize)

	# start acquisition
	rsa300.Run()
	while ready.value == False:
		rsa300.SPECTRUM_WaitForDataReady(timeoutMsec, byref(ready))

	rsa300.SPECTRUM_GetTrace(c_int(0), specSet.traceLength,
							 byref(traceData), byref(outTracePoints))
	#print('Got trace data.')

	# convert trace data from a ctypes array to a numpy array
	trace = np.ctypeslib.as_array(traceData)

	# Peak power and frequency calculations
	peakPower = np.amax(trace)
	peakPowerFreq = freq[np.argmax(trace)]
	#print('Peak power in spectrum: %4.3f dBmV @ %d Hz' % (peakPower, peakPowerFreq))

	# plot the spectrum trace (optional)
	#axes_score.plot(freq, traceData)  #图换1
	
	#draw_Spectrum_total,axes_score,l_user1=draw_picture(freq,traceData,'Spectrum',"Frequency/MHz","Amplitude/dBmV")
	#axes_score.set_ylim(-100,-60)
	
	

	# 绘制方框
	a = []
	b = []  # 存储频率
	c = []
	d = []  # 存储电平
	freq_list=[]
	freq_all=[]
	peakPower=[] #记录峰值
	peakNum=[]  #记录峰值对应的横坐标
	# 得到局部数据
	for i in range(801):
		if traceData[i] > average + 6:
			a.append(i)
			c.append(trace[i])
			freq_list.append(freq[i])
			#print (a)
		elif traceData[i]<average+4:
			if a:
				#print (a)
				b.append(a)
				d.append(c)
				peakNum.append(a[np.argmax(c)])
				peakPower.append(np.max(c))
				freq_all.append(freq_list)
				a = []
				c = []
				freq_list=[]
	#print (b)
	#print (d)
	
	#print ('the length of b is:',len(b))
	#print (average,d,b)

	# 获取局部框架的数量,用来绘制局部的子图
	# rest_freq = []
	# rest_power = []
	# for i in range(len(b)):
		# if len(b[i]) != 0:
			# rest_freq.append(b[i])
			# rest_power.append(d[i])

	j1 = 0
	point=[] #画框在整个界面的坐标信息，用于鼠标坐标相应
	point_xy=[]  #画框的坐标信息 ，用于画框
	print('average:',average)
	for i in range(len(b)):
		# 跳过空数据
		if len(b[i]) > 1:
			j1 += 1
			s1_x = freq[b[i][0]]
			s1_y = average
			s2_x = freq[b[i][-1]]
			s2_y = average + 6
			# s3_x = b[i][0]
			s3_y = np.amax(d[i])
			# s4_x = b[i][-1]
			# s4_y = np.amax(b[i])
			# 画出四条线
			# plt.plot([s1_x, s1_x], [s1_y, s3_y])
			# plt.plot([s1_x, s2_x], [s3_y, s3_y])
			# plt.plot([s2_x, s2_x], [s3_y, s1_y])
			# plt.plot([s1_x, s2_x], [s1_y, s2_y])
			# plt.text((b[i][0]+b[i][-1])/2,s3_y,'%s'%j1)
			
			freq_len=freq[-1]-freq[0]
			if freq_len>20e6:
				#print (average,d[i])
				break;
			#print ('changdu:',freq_len)
			u_x=200000
			u_y=20   #画图与实际的偏置
			point_x1=540*(s1_x-u_x-freq[0])/freq_len+88
			point_x2=540*(s2_x+u_x-freq[0])/freq_len+88
			point_y1=308*(-s1_y+20)/70+48
			point_y2=308*(-s3_y+20)/70+48
			point.append([point_x1,point_x2,point_y2,point_y1])
			point_xy.append([s1_x-u_x,s2_x+u_x,s1_y,s3_y])
			# axes_score.plot([s1_x, s1_x], [s1_y, s3_y])
			# axes_score.plot([s1_x, s2_x], [s3_y, s3_y])
			# axes_score.plot([s2_x, s2_x], [s3_y, s1_y])
			# axes_score.plot([s1_x, s2_x], [s1_y, s1_y])
			# axes_score.text((b[i][0]+b[i][-1])/2,s3_y,'%s'%j1)
	# 分别绘制局部的子图
	j = 0
	Sub_Spectrum=[]     #存当前扫描频率
	Sub_Spectrum2=[]    #存当前扫描电平
	Sub_cf=[]           #存中心频率
	Sub_span=[]         #存扫描频宽（无用）
	Sub_cf_channel=[]   #存峰值 （无用，和Sub_peak重复）
	Sub_band=[]         #存带宽
	Sub_peak=[]         #存峰值
	Sub_illegal=[]      #存当前信号段合法与否
	#print (len(b))
	for i in range(len(b)):
		if len(b[i]) > 1:
		
			#确定signal_No
			#j += 1
			b_f = float(freq[b[i][0]])
			e_f = float(freq[b[i][-1]])
			
			#print ((float(b_f), float(e_f)))
			sql1 = "select SERVICEDID from  RMBT_SERVICE_FREQDETAIL where STARTFREQ <= %s and ENDFREQ >= %s" % (float(b_f/1e6), float(e_f/1e6))
			con1 = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],'110000_rmdsd')
			c = pandas.read_sql(sql1, con1)
			#print (c)
			if len(c) > 0:
				sevice_name = c['SERVICEDID'][0]
			else:
				sevice_name = 'No'
			#print ('sevice_name:',sevice_name)
			

			#print ('have signals:',i+1)
			#band_t, peak_t, peak_tf ,draw_Sub_Spectrum,draw_Sub_Spectrum2= spectrum0(rsa300,b_f, e_f, str_time, j, count, str_tt1, str_tt2,longitude,latitude,rbw,vbw)
			''' 
			plt.plot(b[i], d[i])
			plt.xlabel('Frequency (Hz)')
			plt.ylabel('Amplitude (dBmV)')
			plt.title('Spectrum')
			plt.show()
			'''
			band_t=e_f-b_f
			peak_t=np.amax(d[i])
			peak_tf=freq[b[i][np.argmax(d[i])]]
			
			
			#Sub_Spectrum.append(draw_Sub_Spectrum)
			#Sub_Spectrum2.append(draw_Sub_Spectrum2)
			Sub_cf_channel.append((freq[-1]+freq[0])/2)
			#Sub_span.append(e_f-b_f)
			Sub_span.append(freq[-1]-freq[0])
			Sub_peak.append(peak_t)
			# 输出监测到的信号的真实信号带宽，峰值信息，中心频率
			#print(band_t, peak_t, peak_tf)
			#freq_cf, band = bandwidth2(peakPower[i], peakNum[i], trace, freq)  # 求带宽
			#print (peakPower[i], freq[peakNum[i]], d[i], freq_all[i])
			freq_cf, band = bandwidth3(peakPower[i], freq[peakNum[i]], d[i], freq_all[i])  # 求带宽
			Sub_band.append(band)
			Sub_cf.append(freq_cf)
			
			#判断信号是否合法
			illegal=0  #0表示非法
			reflect_inf1={'0': [116.43, 39.9, 940*1e6, 950*1e6]}
			#print ('reflect_inf:',reflect_inf1)
			for i in reflect_inf1:
				#print (i)
				if freq_cf>=reflect_inf1[i][2] and freq_cf<=reflect_inf1[i][3]:
					illegal=1
					break
			Sub_illegal.append(illegal)

			#判断是否有重复频段,有则序号加1
			if not Sub_cf_all:
				num_signal=num_signal+1
			else:
				divide_freq=np.array(Sub_cf_all)-freq_cf
				if sum(abs(divide_freq)<=0.5*1e6)==0:
					num_signal=num_signal+1
			
			con=mdb.connect(mysql_config['host'],mysql_config['user'],mysql_config['password'],mysql_config['database'])
			with con:
				# 获取连接的cursor，只有获取了cursor，我们才能进行各种操作
				#print ('#'*30)
				#print ('ceshi1')
				cur = con.cursor()
				cur.execute("INSERT INTO SPECTRUM_IDENTIFIED VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s)", [str_time, sevice_name, str_tt1, float(b_f), float(e_f),float(freq_cf),float(band), int(count),float(longitude),float(latitude),num_signal,illegal,float(peak_t)])
				cur.close()
			con.commit()
			con.close()
			
			
	Sub_Spectrum=freq  #频率
	Sub_Spectrum2=trace  #信号
	# 后台自动保存这一次扫频频谱数据
	# df1 = DataFrame({
		# 'datetime':str_tt1,
		# 'frequency': freq,
		# 'power': trace
	# })
	head=['datetime']+['longitude']+['latitude']+list(freq)
	data1=[str_tt1]+[longitude,latitude]+list(trace)
	#print (Sub_band,Sub_cf_channel)

	#draw_Spectrum_total=1
	return head,data1,Sub_cf_channel,Sub_span,Sub_cf,Sub_band,Sub_Spectrum,Sub_Spectrum2,freq, traceData,point,point_xy,Sub_peak,num_signal,Sub_illegal
# 返回原始的频谱数据

# 一次细扫，扫每一个方框的信号；输入参数：起始频率、终止频率、任务名称，方框编号，计数、细扫的时间
def spectrum0(rsa300,startFreq, stopFreq, str_time, k, count, str_tt1, str_tt2,longitude,latitude,rbw,vbw):
    # create Spectrum_Settings data structure

    class Spectrum_Settings(Structure):
        _fields_ = [('span', c_double),
                    ('rbw', c_double),
                    ('enableVBW', c_bool),
                    ('vbw', c_double),
                    ('traceLength', c_int),
                    ('window', c_int),
                    ('verticalUnit', c_int),
                    ('actualStartFreq', c_double),
                    ('actualStopFreq', c_double),
                    ('actualFreqStepSize', c_double),
                    ('actualRBW', c_double),
                    ('actualVBW', c_double),
                    ('actualNumIQSamples', c_double)]

    # initialize variables
    specSet = Spectrum_Settings()
    longArray = c_long * 10
    deviceIDs = longArray()
    deviceSerial = c_wchar_p('')
    numFound = c_int(0)
    enable = c_bool(True)  # spectrum enable
    # cf = c_double(9e8)            #center freq
    refLevel = c_double(0)  # ref level
    ready = c_bool(False)  # ready
    timeoutMsec = c_int(500)  # timeout
    trace = c_int(0)  # select Trace 1
    detector = c_int(0)  # set detector type to max
    # 由起始频率和终止频率直接可以得到中心频率
    # set cf
    cf = c_double((startFreq + stopFreq) / 2)
    '''
    # search the USB 3.0 bus for an RSA306
    ret = rsa300.Search(deviceIDs, byref(deviceSerial), byref(numFound))
    if ret != 0:
        print('Error in Search: ' + str(ret))
    if numFound.value < 1:
        print('No instruments found. Exiting script.')
        exit()
    elif numFound.value == 1:
        print('One device found.')
        print('Device Serial Number: ' + deviceSerial.value)
    else:
        print('2 or more instruments found.')
        # note: the API can only currently access one at a time

    # connect to the first RSA306
    ret = rsa300.Connect(deviceIDs[0])
    if ret != 0:
        print('Error in Connect: ' + str(ret))
'''
    # preset the RSA306 and configure spectrum settings
    rsa300.Preset()
    rsa300.SetCenterFreq(cf)
    rsa300.SetReferenceLevel(refLevel)
    rsa300.SPECTRUM_SetEnable(enable)
    rsa300.SPECTRUM_SetDefault()
    rsa300.SPECTRUM_GetSettings(byref(specSet))

    # configure desired spectrum settings
    # some fields are left blank because the default
    # values set by SPECTRUM_SetDefault() are acceptable
    span = c_double(stopFreq - startFreq)
    specSet.span = span
    specSet.rbw = c_double(rbw)
	# specSet.vbw = c_double(vbw)
    specSet.enableVBW = c_bool(True)
    specSet.vbw = c_double(vbw)
    specSet.traceLength = c_int(801)  # c_int(int(span.value/step_size.value))#c_int(801)
    # specSet.window =
    specSet.verticalUnit = c_int(4)
    specSet.actualStartFreq = startFreq
    specSet.actualStopFreq = stopFreq
    specSet.actualFreqStepSize = c_double(
        span.value / 801)  # step_size c_double(span.value/801)   # c_double(50000.0)
    specSet.detector = detector
    # specSet.actualRBW =
    # specSet.actualVBW =
    # specSet.actualNumIQSamples =

    # set desired spectrum settings
    rsa300.SPECTRUM_SetSettings(specSet)
    rsa300.SPECTRUM_GetSettings(byref(specSet))

    # uncomment this if you want to print out the spectrum settings

    # print out spectrum settings for a sanity check
    # print('Span: ' + str(specSet.span))
    # print('RBW: ' + str(specSet.rbw))
    # print('VBW Enabled: ' + str(specSet.enableVBW))
    # print('VBW: ' + str(specSet.vbw))
    # print('Trace Length: ' + str(specSet.traceLength))
    # print('Window: ' + str(specSet.window))
    # print('Vertical Unit: ' + str(specSet.verticalUnit))
    # print('Actual Start Freq: ' + str(specSet.actualStartFreq))
    # print('Actual End Freq: ' + str(specSet.actualStopFreq))
    # print('Actual Freq Step Size: ' + str(specSet.actualFreqStepSize))
    # print('Actual RBW: ' + str(specSet.actualRBW))
    # print('Actual VBW: ' + str(specSet.actualVBW))

    # initialize variables for GetTrace
    traceArray = c_float * specSet.traceLength
    traceData = traceArray()
    outTracePoints = c_int()

    # generate frequency array for plotting the spectrum
    freq = np.arange(specSet.actualStartFreq,
                     specSet.actualStartFreq + specSet.actualFreqStepSize * specSet.traceLength,
                     specSet.actualFreqStepSize)

    # start acquisition
    rsa300.Run()
    while ready.value == False:
        rsa300.SPECTRUM_WaitForDataReady(timeoutMsec, byref(ready))

    rsa300.SPECTRUM_GetTrace(c_int(0), specSet.traceLength,
                             byref(traceData), byref(outTracePoints))
    # print('Got trace data.')

    # convert trace data from a ctypes array to a numpy array
    trace = np.ctypeslib.as_array(traceData)

    # Peak power and frequency calculations
    peakPower = np.amax(trace)
    peakPowerFreq = freq[np.argmax(trace)]
    # print('Peak power in spectrum: %4.3f dBmV @ %d Hz' % (peakPower, peakPowerFreq))

    # plot the spectrum trace (optional)
    #draw_Sub_Spectrum,axes_score1=draw_picture(freq, traceData,height=2.5,width=3)
    #draw_Sub_Spectrum2,axes_score2=draw_picture(freq, traceData,height=4,width=6) #画大图
    draw_Sub_Spectrum=freq
    draw_Sub_Spectrum2=trace
    # axes_score1.set_xlabel('Frequency (Hz)')
    # axes_score1.set_ylabel('Amplitude (dBmV)')
    # axes_score1.set_title('SubSpectrum-%s'%k)
    # plt.show()
    freq_cf, band = bandwidth(peakPower, peakPowerFreq, trace, freq)  # 带宽
    # 存储细扫描数据
    # 将时间改成合法文件名形式
    #print (str_time)
    if not os.path.exists(os.getcwd()+"\\data1\\%s\\"%str_time):
        os.mkdir(os.getcwd()+"\\data1\\%s\\"%str_time)
    path = os.getcwd()+"\\data1\\%s\\" % str_time + str_tt2 + "spectrum%d.csv" % k
    df0 = DataFrame({
        'datetime': str_tt1,
        'frequency': freq,
        'power': trace
    })
    df0.to_csv(
        path,
        index=False
    )
    '''
    需要将return的基本数据存在数据库中
    '''
    # 获取业务类型写入表中
    # 需要连接对方数据库
    #con1 = mdb.connect('localhost', 'root', '17704882970', 'ceshi1')
    #sql = "select ObjectID from RFBT_Allocation where Spectrum_Start <= %f and Spectrum_Stop >= %f" % (peakPowerFreq,peakPowerFreq)
    #objectId = pandas.read_sql(sql,con1)  # 获取信号业务类型的种类编号

    #con = mdb.connect('localhost', 'root', 'cdk120803', 'ceshi1')
    con=mdb.connect(mysql_config['host'],mysql_config['user'],mysql_config['password'],mysql_config['database'])
    with con:
        # 获取连接的cursor，只有获取了cursor，我们才能进行各种操作
        cur = con.cursor()
        cur.execute("INSERT INTO SPECTRUM_IDENTIFIED VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", [str_time, int(k), str_tt1, float(startFreq), float(stopFreq),float(freq_cf),float(band), int(count),'haha',float(longitude),float(latitude)])
        cur.close()
    con.commit()
    con.close()
    return band, peakPower, freq_cf,draw_Sub_Spectrum,draw_Sub_Spectrum2
# 返回某一方框信号的带宽，中心频率，中心频率峰值,子频谱图

# 仅仅绘制无人机频谱图像
def uav00(rsa300):

    class Spectrum_Settings(Structure):
        _fields_ = [('span', c_double),
                    ('rbw', c_double),
                    ('enableVBW', c_bool),
                    ('vbw', c_double),
                    ('traceLength', c_int),
                    ('window', c_int),
                    ('verticalUnit', c_int),
                    ('actualStartFreq', c_double),
                    ('actualStopFreq', c_double),
                    ('actualFreqStepSize', c_double),
                    ('actualRBW', c_double),
                    ('actualVBW', c_double),
                    ('actualNumIQSamples', c_double)]

    # initialize variables
    specSet = Spectrum_Settings()
    longArray = c_long * 10
    deviceIDs = longArray()
    deviceSerial = c_wchar_p('')
    numFound = c_int(0)
    enable = c_bool(True)  # spectrum enable
    # cf = c_double(9e8)            #center freq
    refLevel = c_double(0)  # ref level
    ready = c_bool(False)  # ready
    timeoutMsec = c_int(500)  # timeout
    trace = c_int(0)  # select Trace 1
    detector = c_int(0)  # set detector type to max
    # 由起始频率和终止频率直接可以得到中心频率
    # set cf
    startFreq = c_double(840.5e6)
    stopFreq = c_double(845e6)
    cf = c_double((startFreq.value + stopFreq.value) / 2)
    '''
    # search the USB 3.0 bus for an RSA306
    ret = rsa300.Search(deviceIDs, byref(deviceSerial), byref(numFound))
    if ret != 0:
        print('Error in Search: ' + str(ret))
    if numFound.value < 1:
        print('No instruments found. Exiting script.')
        exit()
    elif numFound.value == 1:
        print('One device found.')
        print('Device Serial Number: ' + deviceSerial.value)
    else:
        print('2 or more instruments found.')
        # note: the API can only currently access one at a time

    # connect to the first RSA306
    ret = rsa300.Connect(deviceIDs[0])
    if ret != 0:
        print('Error in Connect: ' + str(ret))
'''
    # preset the RSA306 and configure spectrum settings
    rsa300.Preset()
    rsa300.SetCenterFreq(cf)
    rsa300.SetReferenceLevel(refLevel)
    rsa300.SPECTRUM_SetEnable(enable)
    rsa300.SPECTRUM_SetDefault()
    rsa300.SPECTRUM_GetSettings(byref(specSet))

    # configure desired spectrum settings
    # some fields are left blank because the default
    # values set by SPECTRUM_SetDefault() are acceptable
    span = c_double(stopFreq.value - startFreq.value)
    specSet.span = span
    specSet.rbw = c_double(300e3)
    specSet.enableVBW = c_bool(True)
    specSet.vbw = c_double(300e3)
    specSet.traceLength = c_int(801)  # c_int(int(span.value/step_size.value))#c_int(801)
    specSet.detector = detector
    # specSet.window =
    specSet.verticalUnit = c_int(4)
    specSet.actualStartFreq = startFreq
    specSet.actualStopFreq = stopFreq
    specSet.actualFreqStepSize = c_double(
        span.value / 801)  # step_size c_double(span.value/801)   # c_double(50000.0)
    # specSet.actualRBW =
    # specSet.actualVBW =
    # specSet.actualNumIQSamples =

    # set desired spectrum settings
    rsa300.SPECTRUM_SetSettings(specSet)
    rsa300.SPECTRUM_GetSettings(byref(specSet))

    # uncomment this if you want to print out the spectrum settings

    # print out spectrum settings for a sanity check
    # print('Span: ' + str(specSet.span))
    # print('RBW: ' + str(specSet.rbw))
    # print('VBW Enabled: ' + str(specSet.enableVBW))
    # print('VBW: ' + str(specSet.vbw))
    # print('Trace Length: ' + str(specSet.traceLength))
    # print('Window: ' + str(specSet.window))
    # print('Vertical Unit: ' + str(specSet.verticalUnit))
    # print('Actual Start Freq: ' + str(specSet.actualStartFreq))
    # print('Actual End Freq: ' + str(specSet.actualStopFreq))
    # print('Actual Freq Step Size: ' + str(specSet.actualFreqStepSize))
    # print('Actual RBW: ' + str(specSet.actualRBW))
    # print('Actual VBW: ' + str(specSet.actualVBW))

    # initialize variables for GetTrace
    traceArray = c_float * specSet.traceLength
    traceData = traceArray()
    outTracePoints = c_int()

    # generate frequency array for plotting the spectrum
    freq = np.arange(specSet.actualStartFreq,
                     specSet.actualStartFreq + specSet.actualFreqStepSize * specSet.traceLength,
                     specSet.actualFreqStepSize)

    # start acquisition
    rsa300.Run()
    while ready.value == False:
        rsa300.SPECTRUM_WaitForDataReady(timeoutMsec, byref(ready))

    rsa300.SPECTRUM_GetTrace(c_int(0), specSet.traceLength,
                             byref(traceData), byref(outTracePoints))
    # print('Got trace data.')

    # convert trace data from a ctypes array to a numpy array
    trace = np.ctypeslib.as_array(traceData)

    # Peak power and frequency calculations
    peakPower = np.amax(trace)
    peakPowerFreq = freq[np.argmax(trace)]
    # print('Peak power in spectrum: %4.3f dBmV @ %d Hz' % (peakPower, peakPowerFreq))

    # plot the spectrum trace (optional)
    figure_score_uav,axes_score_uav,l_user_uav=draw_picture(freq,traceData,'UAV-Spectrum',"Frequency/MHz","Amplitude/dBmV",2.5,3)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude (dBmV)')
    # plt.title('uav-Spectrum')
    # plt.show()
    return figure_score_uav,traceData
	
# 绘制实时的无人机频谱图，顺便计算出带宽
def uav0(rsa300,startFreq,endFreq,average,rbw,vbw):


	class Spectrum_Settings(Structure):
		_fields_ = [('span', c_double),
					('rbw', c_double),
					('enableVBW', c_bool),
					('vbw', c_double),
					('traceLength', c_int),
					('window', c_int),
					('verticalUnit', c_int),
					('actualStartFreq', c_double),
					('actualStopFreq', c_double),
					('actualFreqStepSize', c_double),
					('actualRBW', c_double),
					('actualVBW', c_double),
					('actualNumIQSamples', c_double)]

	# initialize variables
	specSet = Spectrum_Settings()
	longArray = c_long * 10
	deviceIDs = longArray()
	deviceSerial = c_wchar_p('')
	numFound = c_int(0)
	enable = c_bool(True)  # spectrum enable
	# cf = c_double(9e8)            #center freq
	refLevel = c_double(0)  # ref level
	ready = c_bool(False)  # ready
	timeoutMsec = c_int(500)  # timeout
	trace = c_int(0)  # select Trace 1
	detector = c_int(0)  # set detector type to max
	# 由起始频率和终止频率直接可以得到中心频率
	# set cf
	startFreq = c_double(startFreq)
	stopFreq = c_double(endFreq)
	cf = c_double((startFreq.value + stopFreq.value) / 2)
	specSet.detector = detector
	'''
	# search the USB 3.0 bus for an RSA306
	ret = rsa300.Search(deviceIDs, byref(deviceSerial), byref(numFound))
	if ret != 0:
		print('Error in Search: ' + str(ret))
	if numFound.value < 1:
		print('No instruments found. Exiting script.')
		exit()
	elif numFound.value == 1:
		print('One device found.')
		print('Device Serial Number: ' + deviceSerial.value)
	else:
		print('2 or more instruments found.')
		# note: the API can only currently access one at a time

	# connect to the first RSA306
	ret = rsa300.Connect(deviceIDs[0])
	if ret != 0:
		print('Error in Connect: ' + str(ret))
	'''
	# preset the RSA306 and configure spectrum settings
	rsa300.Preset()
	rsa300.SetCenterFreq(cf)
	rsa300.SetReferenceLevel(refLevel)
	rsa300.SPECTRUM_SetEnable(enable)
	rsa300.SPECTRUM_SetDefault()
	rsa300.SPECTRUM_GetSettings(byref(specSet))

	# configure desired spectrum settings
	# some fields are left blank because the default
	# values set by SPECTRUM_SetDefault() are acceptable
	span = c_double(stopFreq.value - startFreq.value)
	specSet.span = span
	specSet.rbw = c_double(float(rbw))
	specSet.enableVBW = c_bool(True)
	specSet.vbw = c_double(float(vbw))
	specSet.traceLength = c_int(801)  # c_int(int(span.value/step_size.value))#c_int(801)
	# specSet.window =
	specSet.verticalUnit = c_int(4)
	specSet.actualStartFreq = startFreq
	specSet.actualStopFreq = stopFreq
	specSet.actualFreqStepSize = c_double(
		span.value / 801)  # step_size c_double(span.value/801)   # c_double(50000.0)
	specSet.detector = detector
	# specSet.actualRBW =
	# specSet.actualVBW =
	# specSet.actualNumIQSamples =

	# set desired spectrum settings
	rsa300.SPECTRUM_SetSettings(specSet)
	rsa300.SPECTRUM_GetSettings(byref(specSet))

	# uncomment this if you want to print out the spectrum settings

	# print out spectrum settings for a sanity check
	# print('Span: ' + str(specSet.span))
	# print('RBW: ' + str(specSet.rbw))
	# print('VBW Enabled: ' + str(specSet.enableVBW))
	# print('VBW: ' + str(specSet.vbw))
	# print('Trace Length: ' + str(specSet.traceLength))
	# print('Window: ' + str(specSet.window))
	# print('Vertical Unit: ' + str(specSet.verticalUnit))
	# print('Actual Start Freq: ' + str(specSet.actualStartFreq))
	# print('Actual End Freq: ' + str(specSet.actualStopFreq))
	# print('Actual Freq Step Size: ' + str(specSet.actualFreqStepSize))
	# print('Actual RBW: ' + str(specSet.actualRBW))
	# print('Actual VBW: ' + str(specSet.actualVBW))

	# initialize variables for GetTrace
	traceArray = c_float * specSet.traceLength
	traceData = traceArray()
	outTracePoints = c_int()

	# generate frequency array for plotting the spectrum
	freq = np.arange(specSet.actualStartFreq,
					 specSet.actualStartFreq + specSet.actualFreqStepSize * specSet.traceLength,
					 specSet.actualFreqStepSize)

	# start acquisition
	rsa300.Run()
	while ready.value == False:
		rsa300.SPECTRUM_WaitForDataReady(timeoutMsec, byref(ready))

	rsa300.SPECTRUM_GetTrace(c_int(0), specSet.traceLength,
							 byref(traceData), byref(outTracePoints))
	# print('Got trace data.')

	# convert trace data from a ctypes array to a numpy array
	trace = np.ctypeslib.as_array(traceData)

	# Peak power and frequency calculations
	peakPower = np.amax(trace)
	peakPowerFreq = freq[np.argmax(trace)]
	# print('Peak power in spectrum: %4.3f dBmV @ %d Hz' % (peakPower, peakPowerFreq))

	# plot the spectrum trace (optional)
	##draw_UAV_Spectrum,axes_score_uav1,l_user_uav1=draw_picture(freq,traceData,'Spectrum',"Frequency/MHz","Amplitude/dBmV")
	# plt.plot(freq, traceData)
	# plt.xlabel('Frequency (Hz)')
	# plt.ylabel('Amplitude (dBmV)')
	# plt.title('uav-Spectrum')
	# plt.show()
	# 显示无人机实时监测出来的信号频谱中心频点等
	
	# 得到局部峰值数据
	a=[]  #有信号频谱时的横坐标编号
	c=[]  #有信号频谱时的信号强度
	peakPower=[]
	peakNum=[]
	freq_cfs=[] #保存有信号频谱时的中心频率
	bands=[]    #保存有信号频谱时的带宽
	end=0
	for i in range(801):
		if traceData[i] > average + 6:
			a.append(i)
			c.append(trace[i])
		elif traceData[i]<average-4:
			if len(a)>1:
				peakPower.append(np.max(c))
				peakNum.append(a[np.argmax(c)])
				a = []
				c = []
	for i in range(len(peakNum)):
		freq_cf,band=bandwidth2(peakPower[i],peakNum[i],trace,freq)
		freq_cfs.append(freq_cf)
		bands.append(band)

	return freq_cfs, bands, peakPower,freq,traceData
	
# 无人机IQ数据
def uav1(rsa300,band,peakFreq):

    class IQHeader(Structure):
        _fields_ = [('acqDataStatus', c_uint16),
                    ('acquisitionTimestamp', c_uint64),
                    ('frameID', c_uint32), ('trigger1Index', c_uint16),
                    ('trigger2Index', c_uint16), ('timeSyncIndex', c_uint16)]

    # initialize/assign variables
    longArray = c_long * 10
    deviceIDs = longArray()
    deviceSerial = c_wchar_p('')
    numFound = c_int(0)
    serialNum = c_char_p(b'')
    nomenclature = c_char_p(b'')
    header = IQHeader()

    refLevel = c_double(0)
    cf = c_double(peakFreq)
    iqBandwidth = c_double(band)
    recordLength = c_long(1024)
    mode = c_int(0)
    level = c_double(-10)
    iqSampleRate = c_double(0)
    runMode = c_int(0)
    timeoutMsec = c_int(1000)
    ready = c_bool(False)

    # initialize
    rsa300.GetDeviceSerialNumber(serialNum)
    #print('Serial Number: ' + str(serialNum.value))
    rsa300.GetDeviceNomenclature(nomenclature)
    #print('Device Nomenclature: ' + str(nomenclature.value))

    # configure instrument
    rsa300.Preset()
    rsa300.SetReferenceLevel(refLevel)
    rsa300.SetCenterFreq(cf)
    rsa300.SetIQBandwidth(iqBandwidth)
    rsa300.SetIQRecordLength(recordLength)
    rsa300.SetTriggerMode(mode)
    rsa300.SetIFPowerTriggerLevel(level)

    # begin acquisition
    rsa300.Run()

    # get relevant settings values
    rsa300.GetReferenceLevel(byref(refLevel))
    rsa300.GetCenterFreq(byref(cf))
    rsa300.GetIQBandwidth(byref(iqBandwidth))
    rsa300.GetIQRecordLength(byref(recordLength))
    rsa300.GetTriggerMode(byref(mode))
    rsa300.GetIFPowerTriggerLevel(byref(level))
    rsa300.GetRunState(byref(runMode))
    rsa300.GetIQSampleRate(byref(iqSampleRate))

    # for kicks and giggles
    # print('Run Mode:' + str(runMode.value))
    # print('Reference level: ' + str(refLevel.value) + 'dBmV')
    # print('Center frequency: ' + str(cf.value) + 'Hz')
    # print('Bandwidth: ' + str(iqBandwidth.value) + 'Hz')
    # print('Record length: ' + str(recordLength.value))
    # print('Trigger mode: ' + str(mode.value))
    # print('Trigger level: ' + str(level.value) + 'dBmV')
    # print('Sample rate: ' + str(iqSampleRate.value) + 'Samples/sec')

    # check for data ready
    while ready.value == False:
        ret = rsa300.WaitForIQDataReady(timeoutMsec, byref(ready))
    # print('IQ Data is Ready')

    # as a bonus, get the IQ header even though it's not used
    ret = rsa300.GetIQHeader(byref(header))
    if ret != 0:
        print('Error in GetIQHeader: ' + str(ret))
    # print('Got IQ Header')

    # initialize data transfer variables
    iqArray = c_float * recordLength.value
    iData = iqArray()
    qData = iqArray()
    startIndex = c_int(0)

    # query I and Q data
    rsa300.GetIQDataDeinterleaved(byref(iData), byref(qData), startIndex, recordLength)
    # print('Got IQ data')

    # convert ctypes array to numpy array for ease of use
    I = np.ctypeslib.as_array(iData)
    Q = np.ctypeslib.as_array(qData)

    # create time array for plotting
    time = np.linspace(0, recordLength.value / iqSampleRate.value, recordLength.value)
    '''
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.title('I and Q vs Time')
    plt.plot(time, I)
    plt.ylabel('I (V)')
    plt.subplot(2, 1, 2)
    plt.plot(time, Q)
    plt.ylabel('Q (V)')
    plt.xlabel('Time (sec)')
    plt.show()
    '''
    str_time = str(datetime.datetime.now().strftime('%F-%H-%M-%S'))
    df1 = DataFrame({
        'datetime': str_time,
        'I': I,
        'Q': Q
    })
    return df1,I,Q

# 监测无人机信号，单独服务于无人机界面调用uav0和uav1
def uav(rsa,t_r, test_No, deviceSerial, anteid,rbw,vbw):  # 参数是持续时间、测试编号、监测设备信息、天线信息
    #deviceSerial = connect.connect()  # 随设备进行连接然后获取设备信息
    # os.chdir("E:/项目/洪（私）/pro/RSA_API/lib/x64")
    # rsa = WinDLL("RSA_API.dll")
    trace_IQ = DataFrame({})
    count = 0
    average = detectNoise(rsa300,startFreq,endFreq,rbw,vbw)
    a = []  # 存储无人机出现的时间段
    b = []  # 存储无人机信号出现的带宽
    c = []  # 存储无人机信号出现的中心频点
    str_time = str(datetime.datetime.now().strftime('%F-%H-%M-%S'))  # 无人机检测起始时间
    t_ref = time.time()
    longitude = float(116.41)
    latitude = float(39.85)
    while time.time() - t_ref < t_r:
       peakFreq, band, peak,draw_UAV_Spectrum= uav0(rsa,rbw,vbw)
       if peak > average + 6 and count == 0:
           t1 = time.time()
           count = 1
           b.append(band)
           c.append(peakFreq)
       elif peak > average + 6 and count == 1:
           pass
       else:
           t2 = time.time()
           count = 0
           a.append(t2-t1)
       df1 = uav1(rsa,band,peakFreq)
       trace_IQ = pandas.concat([trace_IQ, df1])
    os.mkdir(os.getcwd()+"\\IQ\\%s\\" % str_time)
    path = os.getcwd()+"\\IQ\\%s\\" % str_time + str_time + "IQ%s.csv" % t_r  # IQ数据存储路径
    trace_IQ.to_csv(
        path,
        index=False
    )
    #con = mdb.connect('localhost', 'root', 'cdk120803', 'ceshi1')
    con=mdb.connect(mysql_config['host'],mysql_config['user'],mysql_config['password'],mysql_config['database'])
    with con:
        # 操作数据库需要一次性的进行，一条代码就是写入一行所以一次就把表的一行全部写入
        cur = con.cursor()
        cur.execute("INSERT INTO uav VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s)", [int(test_No), str_time, str_time, float(c[0]-b[0]/2), float(c[0] + b[0] / 2), float(b[0]), path, deviceSerial, anteid,float(longitude),float(latitude)])
        cur.close()
    con.commit()
    con.close()
    if a==[]:
        a.append(t_r)
    #print(a,b,c)
    return a,b,c
	

# 计算频谱占用度必须用到原始格式的数据库
# 参数：起始时间、终止时间、任务名称、其实频率、终止频率
def spectrum_occ(start_time,stop_time,task_name,freq_start,freq_stop):
	# 输入就是起始时间、终止时间、任务名称、起始频率、终止频率
	spectrum_span = freq_stop - freq_start
	# 以一个小时为单位
	sql3 = "select FreQ_BW, COUNT1 from SPECTRUM_IDENTIFIED where Task_Name='%s' && FREQ_CF between %f and %f" % (task_name, float(freq_start), float(freq_stop)) + "&& Start_time between DATE_FORMAT('%s'," % (start_time) + "'%Y-%m-%d %H:%i:%S')" + "and DATE_FORMAT('%s'," % (stop_time) + "'%Y-%m-%d %H:%i:%S')"
	con=mdb.connect(mysql_config['host'],mysql_config['user'],mysql_config['password'],mysql_config['database'])
	#con = mdb.connect('localhost', 'root', 'cdk120803', 'ceshi1')
	c = pandas.read_sql(sql3, con)
	con.commit()
	con.close()
	spectrum_occ1 = sum(c['FreQ_BW'])
	if len(c['COUNT1'])>0:
		c1 = np.array(c['COUNT1'])
		num = max(c['COUNT1']) - min(c['COUNT1'])+1
		spectrum_occ = spectrum_occ1 / float(spectrum_span)
	else:
		print ('count:',len(c['COUNT1']))
		spectrum_occ = 0
	return spectrum_occ  # 返回频谱占用度

# 绘制频谱占用度图像
# 参数：起始时间、终止时间、任务名称、其实频率、终止频率
def plot_spectrum_occ(start_time,stop_time,task_name,freq_start,freq_stop):
    starttime = datetime.datetime.strptime(str(start_time), "%Y-%m-%d %H:%M:%S")
    stoptime = datetime.datetime.strptime(str(stop_time), "%Y-%m-%d %H:%M:%S")
    #print (starttime)
    #print (stoptime)
    delta = int((stoptime - starttime).seconds)  #先以s为单位
    #print (delta)
    occ1 = []
    figure_score = Figure((9,2),100)
    axes_score = figure_score.add_subplot(111,facecolor='w')
    axes_score.set_title("The Spectrum occupancy")
    axes_score.set_xlabel("freq/MHz")
    axes_score.set_ylabel("percentage/%")
    for i in range(delta):
        s_t = starttime + datetime.timedelta(seconds = (i))
        e_t = starttime + datetime.timedelta(seconds = (i+1))
        occ1_1 = spectrum_occ(s_t,e_t,task_name,freq_start,freq_stop)
        occ1.append(occ1_1)
    #print('occ1',occ1)
    time_slot=np.linspace(1,delta,delta)
    dates=[]
    for i in range(delta):
        dates.append(str(starttime+datetime.timedelta(seconds=time_slot[i])))
    a=[datetime.datetime.strptime(d,"%Y-%m-%d %H:%M:%S") for d in dates]
    #print (a)
    axes_score.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
    axes_score.xaxis.set_major_locator(mdates.SecondLocator())
    if len(a)>=2:
        axes_score.set_xlim(a[0],a[-1])
    # for label in axes_score.get_xticklabels():
        # label.set_rotation(0)
    axes_score.plot(a,occ1)
    return figure_score,axes_score


# 计算信道占用度
# 参数：起始时间、终止时间、任务名称、其实频率、终止频率
def channel_occ(start_time,stop_time,task_name,freq_start,freq_stop):
	# 输入就是起始时间、终止时间、任务名称、起始频率、终止频率
	step = float(freq_stop - freq_start) / 40
	sql2 = "select COUNT from minitor_task where Task_Name='%s'" % (task_name)
	con=mdb.connect(mysql_config['host'],mysql_config['user'],mysql_config['password'],mysql_config['database'])
	#con = mdb.connect('localhost', 'root', 'cdk120803', 'ceshi1')
	b = pandas.read_sql(sql2, con)
	# 绘制柱状图
	figure_score = Figure((9,2.1),100)
	axes_score = figure_score.add_subplot(111,facecolor='w')
	axes_score.set_title("The Signal_Channel occupancy")
	axes_score.set_xlabel("freq/MHz",fontsize=12)
	axes_score.set_ylabel("percentage/%")
	if not b.empty:
		print('not empty')
		channel_occupied = []
		for i in range(40):
			start_f = freq_start + i * step
			stop_f = freq_start + (i + 1) * step
			sql1 = "select count1 from SPECTRUM_IDENTIFIED where Task_Name='%s' && FREQ_CF between %f and %f "%(task_name, float(start_f), float(stop_f))+"&& Start_time between DATE_FORMAT('%s',"%(start_time)+"'%Y-%m-%d %H:%i:%S')"+"and DATE_FORMAT('%s'," % (stop_time)+"'%Y-%m-%d %H:%i:%S')"
			a = pandas.read_sql(sql1, con)
			a = a.drop_duplicates()  # 去电重复项
			channel_occupied1 = len(a) / float(max(b['COUNT']) - min(b['COUNT']))
			channel_occupied.append(channel_occupied1)


		axis_x = np.arange(freq_start, freq_stop, 40)
		axis_y = channel_occupied
		axes_score.bar(axis_x, axis_y, 2)
	else:
		print('empty')
	return figure_score,axes_score



def read_file(file):
    path = os.getcwd()+"\\data1\\"+file
    file1 = file[:10]+' '+file[11:13]+':'+file[14:16]+':'+file[17:19]
    file2 = file[21:31]+' '+file[32:34]+':'+file[35:37]+':'+file[38:40]
    #print (file1)
    retain_time=int(file[48:]) #持续时间
    start_time = datetime.datetime.strptime(file1, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.datetime.strptime(file2, "%Y-%m-%d %H:%M:%S")
    sql = "select startFreq,endFreq from minitor_task where Task_Name='%s'"%(file[0:19])
    #print (sql)
    con = mdb.connect('localhost', 'root', 'cdk120803', 'ceshi1')
    c = pandas.read_sql(sql, con)
    #print (c)
    con.commit()
    con.close()
    start_freq = c['startFreq'][0]
    end_freq = c['endFreq'][0]
    return start_time,end_time,retain_time,float(start_freq),float(end_freq),path


# 导入导出数据
# 粗扫描，返回某一次任务的初试频率和终止频率以及频率和功率矩阵,起始时间和终止时间
def importData_cu(task_name,file_name,raw_path):#task_name就是文件夹的名字，filename就是粗扫的csv文件名字，raw_path就是文件存储的路径
    sql = "select startFreq,endFreq,Task_B,Task_E from minitor_task where Task_Name='%s'"%(task_name[0:19])
    con=mdb.connect(mysql_config['host'],mysql_config['user'],mysql_config['password'],mysql_config['database'])
    #con = mdb.connect('localhost', 'root', 'cdk120803', 'ceshi1')
    c_cu = pandas.read_sql(sql, con)
    con.commit()
    con.close()
    start_freq_cu = c_cu['startFreq'][0]
    end_freq_cu = c_cu['endFreq'][0]
    start_time_cu = c_cu['Task_B'][0]
    end_time_cu=c_cu['Task_E'][0]
    path = raw_path+"//"+file_name+'.csv'
    df_cu = pandas.read_csv(path)
    num = int(len(df_cu)/801)
    x_mat = np.array(df_cu['frequency']).reshape(num,801)
    y_mat = np.array(df_cu['power']).reshape(num, 801)
    return start_freq_cu,end_freq_cu,x_mat,y_mat,start_time_cu,end_time_cu

# 细扫描，返回某一次细扫描的起始频率，终止频率，带宽，中心频率，频率数据，功率数据
def importData_xi(task_name,file_name,raw_path):
    start_time = file_name[:10]+' '+file_name[11:13]+':'+file_name[14:16]+':'+file_name[17:19]
    start_time=datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    numb = file_name[27:]
    sql1 = "select FreQ_B, FreQ_E,FreQ_BW，Freq_CF from SPECTRUM_IDENTIFIED where Start_time = DATE_FORMAT('%s'," % (start_time) + "'%Y-%m-%d %H:%i:%S')"+" and Signal_No=%d" % int(numb)
    con=mdb.connect(mysql_config['host'],mysql_config['user'],mysql_config['password'],mysql_config['database'])
    #con = mdb.connect('localhost', 'root', 'cdk120803', 'ceshi1')
    c_xi = pandas.read_sql(sql1, con)
    con.commit()
    con.close()
    start_freq_xi = c_xi['FreQ_B'][0]
    end_freq_xi = c_xi['FreQ_E'][0]
    bandwith_xi = c_xi['FreQ_BW'][0]
    cf_xi = c_xi['FreQ_CF'][0]
    path = raw_path+"//"+file_name+'.csv'
    df_xi = pandas.read_csv(path)
    x_freq_xi = np.array(df_xi['frequency'])
    y_power_xi = np.array(df_xi['power'])
    return start_freq_xi,end_freq_xi,bandwith_xi,cf_xi,x_freq_xi,y_power_xi

# 无人机IQ数据导入，输入是文件名和该文件所在的地址
def importData_uav(file_name,raw_path):
    path = raw_path + "//" + file_name + '.csv'
    df_cu = pandas.read_csv(path)
    num = int(len(df_cu) / 1024)
    I_mat = np.array(df_cu['I']).reshape(num, 1024)
    Q_mat = np.array(df_cu['Q']).reshape(num, 1024)
    return I_mat,Q_mat

# # 输入就是大小为36的信号强度的list
# def ce_xiang_plot(a):
    # N = 36
    # theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    # radii = 10 * a
    # width = np.pi / 50 * np.ones(N)
    # ax = plt.subplot(111, projection='polar')
    # bars = ax.bar(theta, radii, width=width, bottom=0.0)

    # # Use custom colors and opacity
    # for r, bar in zip(radii, bars):
        # bar.set_facecolor(plt.cm.viridis(r / 10.))
        # bar.set_alpha(0.5)

    # plt.show()

# 测向输入就是一个频率输出就是对应的强度
def find_direction(rsa300,freq):

	# os.chdir(os.getcwd())
	# rsa300 = cdll.LoadLibrary("RSA_API.dll")

	# create Spectrum_Settings data structure
	class Spectrum_Settings(Structure):
		_fields_ = [('span', c_double),
					('rbw', c_double),
					('enableVBW', c_bool),
					('vbw', c_double),
					('traceLength', c_int),
					('window', c_int),
					('verticalUnit', c_int),
					('actualStartFreq', c_double),
					('actualStopFreq', c_double),
					('actualFreqStepSize', c_double),
					('actualRBW', c_double),
					('actualVBW', c_double),
					('actualNumIQSamples', c_double)]

	# initialize variables
	specSet = Spectrum_Settings()
	longArray = c_long * 10
	deviceIDs = longArray()
	deviceSerial = c_wchar_p('')
	numFound = c_int(0)
	enable = c_bool(True)  # spectrum enable
	cf = c_double(freq)  # center freq
	refLevel = c_double(0)  # ref level
	ready = c_bool(False)  # ready
	timeoutMsec = c_int(500)  # timeout
	trace = c_int(0)  # select Trace 1
	detector = c_int(1)  # set detector type to max

	# preset the RSA306 and configure spectrum settings
	rsa300.Preset()
	rsa300.SetCenterFreq(cf)
	rsa300.SetReferenceLevel(refLevel)
	rsa300.SPECTRUM_SetEnable(enable)
	rsa300.SPECTRUM_SetDefault()
	rsa300.SPECTRUM_GetSettings(byref(specSet))

	# configure desired spectrum settings
	# some fields are left blank because the default
	# values set by SPECTRUM_SetDefault() are acceptable
	specSet.span = c_double(1e6)
	specSet.rbw = c_double(300e3)
	# specSet.enableVBW =
	# specSet.vbw =
	specSet.traceLength = c_int(801)
	# specSet.window =
	specSet.verticalUnit = c_int(4)
	specSet.actualStartFreq = c_double(freq-0.1e6)
	specSet.actualStopFreq = c_double(freq+0.1e6)
	# specSet.actualFreqStepSize =c_double(50000.0)
	# specSet.actualRBW =
	# specSet.actualVBW =
	# specSet.actualNumIQSamples =

	# set desired spectrum settings
	rsa300.SPECTRUM_SetSettings(specSet)
	rsa300.SPECTRUM_GetSettings(byref(specSet))

	# uncomment this if you want to print out the spectrum settings

	# print out spectrum settings for a sanity check
	# print('Span: ' + str(specSet.span))
	# print('RBW: ' + str(specSet.rbw))
	# print('VBW Enabled: ' + str(specSet.enableVBW))
	# print('VBW: ' + str(specSet.vbw))
	# print('Trace Length: ' + str(specSet.traceLength))
	# print('Window: ' + str(specSet.window))
	# print('Vertical Unit: ' + str(specSet.verticalUnit))
	# print('Actual Start Freq: ' + str(specSet.actualStartFreq))
	# print('Actual End Freq: ' + str(specSet.actualStopFreq))
	# print('Actual Freq Step Size: ' + str(specSet.actualFreqStepSize))
	# print('Actual RBW: ' + str(specSet.actualRBW))
	# print('Actual VBW: ' + str(specSet.actualVBW))

	# initialize variables for GetTrace
	traceArray = c_float * specSet.traceLength
	traceData = traceArray()
	outTracePoints = c_int()

	# generate frequency array for plotting the spectrum
	freq = np.arange(specSet.actualStartFreq,
					 specSet.actualStartFreq + specSet.actualFreqStepSize * specSet.traceLength,
					 specSet.actualFreqStepSize)

	# start acquisition
	rsa300.Run()
	while ready.value == False:
		rsa300.SPECTRUM_WaitForDataReady(timeoutMsec, byref(ready))

	rsa300.SPECTRUM_GetTrace(c_int(0), specSet.traceLength,
							 byref(traceData), byref(outTracePoints))
	print('Got trace data.')

	# convert trace data from a ctypes array to a numpy array
	trace = np.ctypeslib.as_array(traceData)

	# Peak power and frequency calculations
	peakPower = np.amax(trace)
	peakPowerFreq = freq[np.argmax(trace)]
	# print('Peak power in spectrum: %4.3f dBmV @ %d Hz' % (peakPower, peakPowerFreq))
	return peakPower

#################################################
####地图功能模块

#返回的result1是台站信息一个字典{编号：[经度，维度，[状态信息，带宽，起始频率，终止频率，调制方式]]}
#返回的result2是测试点信息是一个list[[测试点1的经纬度],[测试点2的经纬度]，.....[]]
def get_station_inf():
    sql1 = "select STAT_TYPE,FREQ_EFB,FREQ_LC,FREQ_UC,FREQ_MOD,STAT_LG,STAT_LA from rsbt_station "
    sql2 = "select LOGITUDE,LATITUDE from spectrum_identified"
    con1 = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],
                      mysql_config['database'])
    inf1 = pandas.read_sql(sql1, con1)
    inf2 = pandas.read_sql(sql2, con1)
    inf2 = inf2.drop_duplicates()
    # print(inf2)
    result1 = {}
    result2 = {}
    # print(len(inf2))
    for i in range(len(inf2)):
        result2[(i + 1)] = [inf2['LOGITUDE'][i], inf2['LATITUDE'][i]]
    for i in range(len(inf1)):
        result1[(i + 1)] = [inf1['STAT_LG'][i], inf1['STAT_LA'][i],
                            [inf1['STAT_TYPE'][i], inf1['FREQ_EFB'][i], inf1['FREQ_LC'][i], inf1['FREQ_UC'][i],
                             inf1['FREQ_MOD'][i]]]
    print(result1, result2)
    return result1, result2

# 输入经纬度返回[测试点经度，维度，起始频率，终止频率，中心频点，带宽]
def reflect_inf(freq_lc,freq_uc,longitude,latitude):
    sql2 = "select FreQ_B,FreQ_E,FREQ_CF,FreQ_BW from spectrum_identified where FreQ_B >= %s and FreQ_E<=%s and LOGITUDE=%s and LATITUDE=%s" % (
    float(freq_lc), float(freq_uc), float(longitude), float(latitude))
    con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],
                      mysql_config['database'])
    ref = pandas.read_sql(sql2, con)
    print(ref)
    result = []
    if len(ref) == 0:
        print('no match data')
    else:
        result = [ref['FreQ_B'][0], ref['FreQ_E'][0], ref['FREQ_CF'][0], ref['FreQ_BW'][0]]
    return result


#数据库存储
def rmbt_facility_freqband_emenv(task_name,span,start_time,end_time,longitude,latitude,mfid='11000001400001', statismode='04',serviceid='1',address='aasfasdfasfasdf',threshold=2.3,occ2=0,height=0):
    sql2 = "select FreQ_BW,COUNT1,legal from SPECTRUM_IDENTIFIED where Task_Name='%s'" % (
        task_name) + "&& Start_time between DATE_FORMAT('%s'," % (
        start_time) + "'%Y-%m-%d %H:%i:%S')" + "and DATE_FORMAT('%s'," % (end_time) + "'%Y-%m-%d %H:%i:%S')"
    con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],
                      mysql_config['database'])
    inf = pandas.read_sql(sql2, con)
    statisstartday = str(start_time)
    statisendday = str(end_time)
    latitude = Decimal(latitude).quantize(Decimal('0.000000000'))
    longitude = Decimal(longitude).quantize(Decimal('0.000000000'))
    frame_num = len(inf['COUNT1'].drop_duplicates())
    sig = np.sum(inf['FreQ_BW'].values)
    occ = (float(sig)*100) / (span*frame_num)
    bandoccupancy = Decimal(occ).quantize(Decimal('0.00'))
    threshold = Decimal(threshold).quantize(Decimal('0.00'))
    height = Decimal(height).quantize(Decimal('0.00'))
    ######legal=1是合法0是非法
    legal_sig = inf[inf['legal']==1]
    legal_frame_num = len(legal_sig['COUNT1'].drop_duplicates())
    legal_band = np.sum(legal_sig['FreQ_BW'].values)
    occ1 = float(legal_band*100)/(span*legal_frame_num)
    occ1 = Decimal(occ1).quantize(Decimal('0.00'))
    occ2 = Decimal(occ2).quantize(Decimal('0.00'))
    occ3  =100 - occ1
    occ3 = Decimal(occ3).quantize(Decimal('0.00'))
    con.close()
    con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],
                      '110000_rmdsd')
    with con:
        # 获取连接的cursor，只有获取了cursor，我们才能进行各种操作
        cur = con.cursor()  # 一条游标操作就是一条数据插入，第二条游标操作就是第二条记录，所以最好一次性插入或者日后更新也行
        cur.execute("INSERT INTO rmbt_facility_freqband_emenv VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [mfid, statisstartday, statisendday, statismode, serviceid, bandoccupancy, threshold, occ1, occ2, occ3,
                     latitude, longitude, height, address])
        cur.close()
    con.commit()
    con.close()

def rmbt_facility_freq_emenv(task_name,start_time,end_time,ssid,mfid='11000001400001',statismode='04',amplitudeunit='01',threshold=6):
    #sql2 = "select Signal_No, Start_time, FREQ_CF, FreQ_BW, COUNT1, peakpower, channel_no from spectrum_identified where Task_Name='%s'" % (task_name)
    sql2 = "select COUNT1 from SPECTRUM_IDENTIFIED where Task_Name='%s'" % (
        task_name) + "&& Start_time between DATE_FORMAT('%s'," % (
        start_time) + "'%Y-%m-%d %H:%i:%S')" + "and DATE_FORMAT('%s'," % (end_time) + "'%Y-%m-%d %H:%i:%S')"
    con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],
                      mysql_config['database'])
    inf = pandas.read_sql(sql2, con)
    df = inf
    con.close()
    list1 = df['Signal_No'].drop_duplicates().values
    df_r = []
    point_r = []
    for i in range(len(list1)):
        point = [0]
        df1 = df[df['Signal_No'] == list1[i]]
        df1["index"] = range(len(df1))
        df1 = df1.set_index(["index"])
        df_r.append(df1)
        for j in range(1,len(df1)):
            if df1['COUNT1'][j] - df1['COUNT1'][j-1] > 1:
                point.append(i)
        point_r.append(point)
    for sig in range(len(df_r)):
        occ = len(df_r[sig]['COUNT1'].drop_duplicates()) / float(len(df['COUNT1'].drop_duplicates()))
        if occ == 1:
            occ = Decimal(occ).quantize(Decimal('0.00'))
        else:
            occ = Decimal(occ * 100).quantize(Decimal('0.00'))
        if len(point_r[sig]) == 0:
            statisstartday = str(df_r[sig]['Start_time'][0])
            statisendday = str(df_r[sig]['Start_time'][len(df_r[sig])-1])
            servicedid = df_r[sig]['Signal_No'][0]
            cf = df_r[sig]['FREQ_CF'].values
            cf_avg = np.average(cf) / 1e6
            cf_avg = Decimal(cf_avg).quantize(Decimal('0.0000000'))
            bandwidth = df_r[sig]['FreQ_BW'].values
            band_avg = np.average(bandwidth) / 1e6
            band_avg = Decimal(band_avg).quantize(Decimal('0.0000000'))
            maxamplitude = np.max(df_r[sig]['peakpower'].values)
            maxamplitude = Decimal(maxamplitude).quantize(Decimal('0.00'))
            midamplitude = maxamplitude / 2
            threshold = Decimal(threshold).quantize(Decimal('0.00'))
            con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],
                              '110000_rmdsd')
            with con:
                # 获取连接的cursor，只有获取了cursor，我们才能进行各种操作
                cur = con.cursor()  # 一条游标操作就是一条数据插入，第二条游标操作就是第二条记录，所以最好一次性插入或者日后更新也行
                cur.execute("INSERT INTO rmbt_facility_freq_emenv VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                            [str(mfid), statisstartday, statisendday, str(statismode), str(servicedid), cf_avg, band_avg, str(ssid),str(amplitudeunit), maxamplitude,midamplitude, occ, threshold])
                cur.close()
            con.commit()
            con.close()
        else:
            point_r[sig].append(len(df_r[sig])-1)
            for i in range(1,len(point_r[sig])):
                start_index = point_r[sig][i-1]
                end_index = point_r[sig][i]-1
                statisstartday = str(df_r[sig]['Start_time'][start_index])
                statisendday = str(df_r[sig]['Start_time'][end_index])
                servicedid = df_r[sig]['Signal_No'][0]
                cf = df_r[sig]['FREQ_CF'][start_index:end_index].values
                cf_avg = np.average(cf)/1e6
                cf_avg = Decimal(cf_avg).quantize(Decimal('0.0000000'))
                bandwidth = df_r[sig]['FreQ_BW'][start_index:end_index].values
                band_avg = np.average(bandwidth)/1e6
                band_avg = Decimal(band_avg).quantize(Decimal('0.0000000'))
                maxamplitude = np.max(df_r[sig]['peakpower'][start_index:end_index].values)
                maxamplitude = Decimal(maxamplitude).quantize(Decimal('0.00'))
                midamplitude = maxamplitude / 2
                threshold = Decimal(threshold).quantize(Decimal('0.00'))
                con = mdb.connect('localhost', 'root', '17704882970', '110000_rmdsd')
                with con:
                    # 获取连接的cursor，只有获取了cursor，我们才能进行各种操作
                    cur = con.cursor()  # 一条游标操作就是一条数据插入，第二条游标操作就是第二条记录，所以最好一次性插入或者日后更新也行
                    cur.execute(
                        "INSERT INTO rmbt_facility_freq_emenv VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        [str(mfid), statisstartday, statisendday, str(statismode), str(servicedid), cf_avg, band_avg, str(ssid),str(amplitudeunit), maxamplitude,midamplitude, occ, threshold])
                    cur.close()
                con.commit()
                con.close()

#
def rmbt_facility_freq_emenv0(task_name,start_time,end_time,ssid,mfid='11000001400001',statismode='04',amplitudeunit='01',threshold=6):
    #sql2 = "select Signal_No, Start_time, FREQ_CF, FreQ_BW, COUNT1, peakpower, channel_no from spectrum_identified where Task_Name='%s'" % (task_name)
    sql2 = "select COUNT1,Signal_No,FREQ_CF,FreQ_BW from SPECTRUM_IDENTIFIED where Task_Name='%s'" % (
        task_name) + "&& Start_time between DATE_FORMAT('%s'," % (
        start_time) + "'%Y-%m-%d %H:%i:%S')" + "and DATE_FORMAT('%s'," % (end_time) + "'%Y-%m-%d %H:%i:%S')"
    con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],
                      mysql_config['database'])
    inf = pandas.read_sql(sql2, con)
    df = inf
    con.close()
    list1 = df['Signal_No'].drop_duplicates().values
    df_r = []
    for i in range(len(list1)):
        df1 = df[df['Signal_No'] == list1[i]]
        df1["index"] = range(len(df1))
        df1 = df1.set_index(["index"])
        df_r.append(df1)
    for sig in range(len(df_r)):
        occ = len(df_r[sig]['COUNT1'].drop_duplicates()) / float(len(df['COUNT1'].drop_duplicates()))
        if occ == 1:
            occ = Decimal(occ).quantize(Decimal('0.00'))
        else:
            occ = Decimal(occ * 100).quantize(Decimal('0.00'))
        statisstartday = str(start_time)
        statisendday = str(end_time)
        servicedid = df_r[sig]['Signal_No'][0]
        cf = df_r[sig]['FREQ_CF'].values
        cf_avg = np.average(cf) / 1e6
        cf_avg = Decimal(cf_avg).quantize(Decimal('0.0000000'))
        bandwidth = df_r[sig]['FreQ_BW'].values
        band_avg = np.average(bandwidth) / 1e6
        band_avg = Decimal(band_avg).quantize(Decimal('0.0000000'))
        #maxamplitude = np.max(df_r[sig]['peakpower'].values)
        maxamplitude = Decimal(-40).quantize(Decimal('0.00'))
        midamplitude = maxamplitude / 2
        threshold = Decimal(threshold).quantize(Decimal('0.00'))
        con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],
                          '110000_rmdsd')
        with con:
            # 获取连接的cursor，只有获取了cursor，我们才能进行各种操作
            cur = con.cursor()  # 一条游标操作就是一条数据插入，第二条游标操作就是第二条记录，所以最好一次性插入或者日后更新也行
            cur.execute("INSERT INTO rmbt_facility_freq_emenv VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        [str(mfid), statisstartday, statisendday, str(statismode), str(servicedid), cf_avg, band_avg,
                         str(ssid), str(amplitudeunit), maxamplitude, midamplitude, occ, threshold])
            cur.close()
        con.commit()

#
def rmbt_facility_freq_emenv1(task_name,start_time,end_time,ssid,mfid='11000001400001',statismode='04',amplitudeunit='01',threshold=6):
    sql1 = "select COUNT1 from SPECTRUM_IDENTIFIED where Task_Name='%s'" % (
    task_name) + "&& Start_time between DATE_FORMAT('%s'," % (
    start_time) + "'%Y-%m-%d %H:%i:%S')" + "and DATE_FORMAT('%s'," % (end_time) + "'%Y-%m-%d %H:%i:%S')"
    con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],
                      mysql_config['database'])
    inf = pandas.read_sql(sql1, con)
    df = inf
    con.close()
    list1 = df['channel_no'].drop_duplicates().values
    #for i in range(len(list1)):
    # def rmbt_freq_occupancy(span,start_time,end_time,startFreq,stopFreq,longitude,latitude,height,mfid='11000001400001',addr='aasfasdfasfasdf',amplitudeunit='01'):
    txt = str(0)
    step = span / float(801)
    startFreq1 = Decimal(startFreq/1e6).quantize(Decimal('0.0000000'))
    stopFreq1 = Decimal(stopFreq/1e6).quantize(Decimal('0.0000000'))
    step1 = Decimal(step).quantize(Decimal('0.0000000'))
    longitude1 = Decimal(longitude).quantize(Decimal('0.000000000'))
    latitude1 = Decimal(latitude).quantize(Decimal('0.000000000'))
    height1 = Decimal(height).quantize(Decimal('0.00'))
    con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],
                      '110000_rmdsd')
    # con = mdb.connect('localhost', 'root', 'cdk120803', 'ceshi1')
    with con:
        # 获取连接的cursor，只有获取了cursor，我们才能进行各种操作
        cur = con.cursor()  # 一条游标操作就是一条数据插入，第二条游标操作就是第二条记录，所以最好一次性插入或者日后更新也行
        # print([str_time, start_time, str_time1, float(t), float(s_c), path, deviceSerial, anteid, count])
        cur.execute("INSERT INTO RMBT_FREQ_OCCUPANCY VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [mfid,str(start_time), str(end_time), startFreq1, stopFreq1, step1, longitude1, latitude1, height1,
                     addr, amplitudeunit, txt])
        cur.close()
    con.commit()
    con.close()

def rmbt_facility_freq_emenv3(task_name,start_time,end_time,ssid,mfid='11000001400001',statismode='04',amplitudeunit='01',threshold=6):
    #sql2 = "select Signal_No, Start_time, FREQ_CF, FreQ_BW, COUNT1, peakpower, channel_no from spectrum_identified where Task_Name='%s'" % (task_name)
    sql2 = "select COUNT1,Signal_No,FREQ_CF,FreQ_BW,peakpower from SPECTRUM_IDENTIFIED where Task_Name='%s'" % (
        task_name) + "&& Start_time between DATE_FORMAT('%s'," % (
        start_time) + "'%Y-%m-%d %H:%i:%S')" + "and DATE_FORMAT('%s'," % (end_time) + "'%Y-%m-%d %H:%i:%S')"
    con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],
                      mysql_config['database'])
    inf = pandas.read_sql(sql2, con)
    df = inf
    con.close()
    list1 = df['Signal_No'].drop_duplicates().values
    df_r = []
    for i in range(len(list1)):
        df1 = df[df['Signal_No'] == list1[i]]
        df1["index"] = range(len(df1))
        df1 = df1.set_index(["index"])
        df_r.append(df1)
    for sig in range(len(df_r)):
        occ = len(df_r[sig]['COUNT1'].drop_duplicates()) / float(len(df['COUNT1'].drop_duplicates()))
        if occ == 1:
            occ = Decimal(occ).quantize(Decimal('0.00'))
        else:
            occ = Decimal(occ * 100).quantize(Decimal('0.00'))
        statisstartday = str(start_time)
        statisendday = str(end_time)
        servicedid = df_r[sig]['Signal_No'][0]
        cf = df_r[sig]['FREQ_CF'].values
        cf_avg = np.average(cf) / 10e6
        cf_avg = Decimal(cf_avg).quantize(Decimal('0.0000000'))
        bandwidth = df_r[sig]['FreQ_BW'].values
        band_avg = np.average(bandwidth) / 10e6
        band_avg = Decimal(band_avg).quantize(Decimal('0.0000000'))
        maxamplitude = np.max(df_r[sig]['peakpower'].values)
        maxamplitude = Decimal(maxamplitude).quantize(Decimal('0.00'))
        temp = np.sort(df_r[sig]['peakpower'])
        mid = temp[int(len(temp) / 2)]
        midamplitude = mid
        midamplitude = Decimal(midamplitude).quantize(Decimal('0.00'))
        threshold = Decimal(threshold).quantize(Decimal('0.00'))
        con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],
                          '110000_rmdsd')
        with con:
            # 获取连接的cursor，只有获取了cursor，我们才能进行各种操作
            cur = con.cursor()  # 一条游标操作就是一条数据插入，第二条游标操作就是第二条记录，所以最好一次性插入或者日后更新也行
            cur.execute("INSERT INTO rmbt_facility_freq_emenv VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        [str(mfid), statisstartday, statisendday, str(statismode), str(servicedid), cf_avg, band_avg,
                         str(ssid), str(amplitudeunit), maxamplitude, midamplitude, occ, threshold])
            cur.close()
        con.commit()



def rmbt_freq_occupancy(span,start_time,end_time,startFreq,stopFreq,longitude,latitude,height,trace1,average,mfid='11000001400001',addr='aasfasdfasfasdf',amplitudeunit='01'):
    step = span / float(801)
    startFreq1 = Decimal(startFreq/1e6).quantize(Decimal('0.0000000'))
    stopFreq1 = Decimal(stopFreq/1e6).quantize(Decimal('0.0000000'))
    step1 = Decimal(step).quantize(Decimal('0.0000000'))
    longitude1 = Decimal(longitude).quantize(Decimal('0.000000000'))
    latitude1 = Decimal(latitude).quantize(Decimal('0.000000000'))
    height1 = Decimal(height).quantize(Decimal('0.00'))
    con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],
                      '110000_rmdsd')
    # con = mdb.connect('localhost', 'root', 'cdk120803', 'ceshi1')
    z = pandas.DataFrame({})
    point = np.arange(0,801,1)
    z['freq_point'] = point
    z['freq_startFreq'] = trace1['frequency'][0:801]
    peak_list = []
    mid_list = []
    occ1 = []
    for i in z['freq_startFreq']:
        temp = trace1[trace1['frequency']==i]
        temp1 = np.sort(temp['power'])
        peak = np.max(temp['power'].values)
        mid = temp1[int(len(temp)/2)]
        occ = len(temp[temp['power']>average+6])/float(len(temp))
        occ = round(occ,2)
        occ1.append(occ*100)
        mid_list.append(mid)
        peak_list.append(peak)
    z['Max_peak'] = peak_list
    z['mid'] = mid_list
    z['occ'] = occ1
    z['threshold'] = 6
    #存入BLOB文件需要中间缓存文件
    file_path=os.getcwd()+"\\cache\\"
    #print (file_path)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    path1=file_path+'cache.csv'
    z.to_csv(path1) 
    file = open(path1)
    load_file = file.read()
    file.close()
    with con:
        # 获取连接的cursor，只有获取了cursor，我们才能进行各种操作
        cur = con.cursor()  # 一条游标操作就是一条数据插入，第二条游标操作就是第二条记录，所以最好一次性插入或者日后更新也行
        # print([str_time, start_time, str_time1, float(t), float(s_c), path, deviceSerial, anteid, count])
        cur.execute("INSERT INTO RMBT_FREQ_OCCUPANCY VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [mfid,str(start_time), str(end_time), startFreq1, stopFreq1, step1, longitude1, latitude1, height1,
                     addr, amplitudeunit, load_file])
        cur.close()
    con.commit()
    con.close()

#台站数据获取
def taizhan_out(task_name):
    sql = "select FREQ_CF,FreQ_B,FreQ_E,FreQ_BW,LOGITUDE,LATITUDE,peakpower,legal,channel_no from spectrum_identified where Task_Name='%s'" % (task_name)
    #print (sql)
    #print (mysql_config1)
    con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],
                      mysql_config['database'])
    inf = pandas.read_sql(sql, con)
    #print (inf)
    inf1 = inf.drop_duplicates(['channel_no'])
    cf = []
    bf = []
    ef = []
    band = []
    longitude = []
    latitude = []
    illegal = []
    peakPower = []
    for i in inf1.values:
        cf.append(i[0])
        bf.append(i[1])
        ef.append(i[2])
        band.append(i[3])
        longitude.append(i[4])
        latitude.append(i[5])
        peakPower.append(i[6])
        illegal.append(i[7])
    return cf,bf,ef,band,longitude,latitude,illegal,peakPower

def read_service():
    sql1 = "select SERVICEDID,FREQNAME,STARTFREQ,ENDFREQ from  RMBT_SERVICE_FREQDETAIL" 
    con1 = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'], '110000_rmdsd')
    c = pandas.read_sql(sql1, con1)
    servicedid = []
    freqname = []
    freqrange_list=[]
    for i in c.values:
        servicedid.append(i[0])
        freqname.append(i[1])
        freqrange_list.append(str(i[2])+'-'+str(i[3])+' MHz')
    return servicedid,freqname,freqrange_list


def get_GPS():
    longitude=-1
    latitude=-1
    highlight=-1
    try: 
        ser = serial.Serial('COM3', 9600) 
    except Exception as e: 
        #print ('open serial failed.')
        return longitude,latitude,highlight
    print ('A Serial Echo Is Running...')
    data=[]
    num_all=9
    num=0
    iscommon=1
    s = ser.read(1)
    print (s)
    print (s.decode('gb18030')=='$')
    while iscommon: 
        # echo 
        data1=''
        data1=data1.encode('utf-8')
        s = ser.read(1)
        while s.decode('gb18030')!='$':
            data1=data1+s
            s = ser.read(1)
  
        data.append(data1.decode('gb18030'))
        num += 1
        if num==num_all:
            iscommon=0

    print (data)

    data_list=[]
    for str_data in data:
        str_data=str_data.replace('\r\n','')
        data_list.append(str_data.split(','))
    print (data_list)
    print (type(data_list[0][0]))

    for data in data_list:
        if data[0]=='GNRMC':
            if data[2]=='A':
                longitude=float(str(data[3])[0:2])+float(str(data[3])[2:])/60
                latitude=float(str(data[5])[0:3])+float(str(data[5])[3:])/60
                print (longitude,latitude)
            else:
                print (None)
        if data[0]=='GNGGA':
            if data[9]!='':
                highlight=float(data[9])
    return longitude,latitude,highlight


def get_GPS2(rsa,GPS_q):
    eventID = c_int(1)  # 0:overrange, 1:ext trig, 2:1PPS
    eventOccurred = c_bool(False)
    eventTimestamp = c_uint64(0)
    hwInstalled = c_bool(False)

    """#################SEARCH/CONNECT#################"""
    hwInstalled = setup_gnss(rsa)

    """#################CONFIGURE INSTRUMENT#################"""
    rsa.CONFIG_Preset()

    rsa.DEVICE_Run()

    if hwInstalled == True:
        """#######USE THIS IF YOU HAVE AN RSA500/600 WITH GPS ANTENNA########"""
        print('Waiting for internal 1PPS.')
        while eventOccurred.value == False:
            rsa.GNSS_Get1PPSTimestamp(byref(eventOccurred), byref(eventTimestamp))
        nmeaMessage = get_gnss_message(rsa)
        GPS_q.put(nmeaMessage)
        return nmeaMessage
    else:
        """#######USE THIS IF YOU HAVE AN RSA306 W/1PPS INPUT########"""
        print('Waiting for external 1PPS.')
        # while eventOccurred.value == False:
            # rsa.DEVICE_GetEventStatus(eventID, byref(eventOccurred), byref(eventTimestamp))
        return [-1,-1,-1]



    rsa.DEVICE_Stop()

    print('Disconnecting.')
    ret = rsa.DEVICE_Disconnect()

def search_connect(rsa):
    # search/connect variables
    numFound = c_int(0)
    intArray = c_int * 10
    deviceIDs = intArray()
    # this is absolutely asinine, but it works
    deviceSerial = c_char_p(b'longer than the longest serial number')
    deviceType = c_char_p(b'longer than the longest device type')
    apiVersion = c_char_p(b'api')

    # get API version
    rsa.DEVICE_GetAPIVersion(apiVersion)
    print('API Version {}'.format(apiVersion.value))

    # search
    ret = rsa.DEVICE_Search(byref(numFound), deviceIDs,
                            deviceSerial, deviceType)

    if ret != 0:
        print('Error in Search: ' + str(ret))
        exit()
    if numFound.value < 1:
        print('No instruments found. Exiting script.')
        exit()
    elif numFound.value == 1:
        print('One device found.')
        print('Device type: {}'.format(deviceType.value))
        print('Device serial number: {}'.format(deviceSerial.value))
        ret = rsa.DEVICE_Connect(deviceIDs[0])
		
        if ret != 0:
            print('Error in Connect: ' + str(ret))
            exit()
    else:
        print('2 or more instruments found. Enumerating instruments, please wait.')
        for inst in range(numFound.value):
            rsa.DEVICE_Connect(deviceIDs[inst])
            rsa.DEVICE_GetSerialNumber(deviceSerial)
            rsa.DEVICE_GetNomenclature(deviceType)
            print('Device {}'.format(inst))
            print('Device Type: {}'.format(deviceType.value))
            print('Device serial number: {}'.format(deviceSerial.value))
            rsa.DEVICE_Disconnect()
        # note: the API can only currently access one at a time
        selection = 1024
        while (selection > numFound.value - 1) or (selection < 0):
            selection = int(input('Select device between 0 and {}\n> '.format(numFound.value - 1)))
        rsa.DEVICE_Connect(deviceIDs[selection])
        return selection


def setup_gnss(rsa, system=2):
    # setup variables
    enable = c_bool(True)
    powered = c_bool(True)
    installed = c_bool(False)
    locked = c_bool(False)#############
    msgLength = c_int(0)
    message = c_char_p(b'')
    # 1:GPS/GLONASS, 2:GPS/BEIDOU, 3:GPS, 4:GLONASS, 5:BEIDOU
    satSystem = c_int(system)

    # check for GNSS hardware
    rsa.GNSS_GetHwInstalled(byref(installed))
    if installed.value != True:
        print('No GNSS hardware installed, ensure there is a 1PPS signal present at the trigger/synch input.')
        input('Press ''ENTER'' to continue > ')
    else:
        # send setup commands to RSA
        rsa.GNSS_SetEnable(enable)
        rsa.GNSS_SetAntennaPower(powered)
        rsa.GNSS_SetSatSystem(satSystem)
        rsa.GNSS_GetEnable(byref(enable))
        rsa.GNSS_GetAntennaPower(byref(powered))
        rsa.GNSS_GetSatSystem(byref(satSystem))

    print('Waiting for GNSS lock.')
    waiti=0;
    while locked.value != True:
        waiti+=1;
        if waiti%100000==0:
            print (waiti)
        rsa.GNSS_GetStatusRxLock(byref(locked))
    print('GNSS locked.')

    return installed.value

def get_gnss_message(rsa):
    msgLength = c_int(0)
    message = c_char_p(b'')
    numMessages = 20
    #gnssMessage = []
    nmeaMessages = []
    data1=''
    #data1.encode('utf-8')
    print (111)

    # grab a certain number of GNSS message strings
    for i in range(numMessages):
        time.sleep(.25)
        rsa.GNSS_GetNavMessageData(byref(msgLength), byref(message))
        # concatenate the new string
        
        #gnssMessage += message.value
        print ('H'*20)
        print(type(message.value))
        try:
            data1=data1+message.value.decode('cp1252')
        except TypeError as e:
            print ('X'*30)
            
        #print (message.value)
    print (data1)
    dataString=''.join(map(str,data1));
    datalist=dataString.split('$')
    print ('start')
    [longitude,latitude,height]=[-1,-1,-1]
    for i in range(len(datalist)): 
        if 'GNGGA' in datalist[i]:
            data=datalist[i].split(',')
            print ('@'*20)
            print (data)
            print (len(data))
            if len(data)>=10:
                if data[2]!='' and data[4]!='' and data[9]!='':
                    longitude=float(str(data[2])[0:2])+float(str(data[2])[2:])/60
                    latitude=float(str(data[4])[0:3])+float(str(data[4])[3:])/60
                    height=float(data[9])
                    print (longitude,latitude,height)
                    break;
    return [longitude,latitude,height]

