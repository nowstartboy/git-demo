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
import cx_Oracle
#import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from decimal import Decimal
import serial 


class method1:

	def __init__(self):
		self.config_data=self.read_config()
		self.direction_cache=self.read_direction_cache()
		self.reflect_inf=self.read_reflect_inf()
		self.mysql_config=self.read_mysql_config()
		self.uav_config=self.read_uav_config()
		self.conn_str1 = self.mysql_config['user1']+'/'+self.mysql_config['password1']+'@'+self.mysql_config['host']+':'+self.mysql_config['interface']+'/'+self.mysql_config['SID']
		self.conn_str2 = self.mysql_config['user2']+'/'+self.mysql_config['password2']+'@'+self.mysql_config['host']+':'+self.mysql_config['interface']+'/'+self.mysql_config['SID']
		self.conn_str3 = self.mysql_config['user3']+'/'+self.mysql_config['password3']+'@'+self.mysql_config['host']+':'+self.mysql_config['interface']+'/'+self.mysql_config['SID']
		# conn_str1 测试数据库
		# conn_str2 月报数据库
		# conn_str3 台站数据库
		print(self.conn_str1,self.conn_str2)
		print ('reflect_inf:',self.reflect_inf)
		self.station_info = pandas.read_csv('taizhan.csv')
		# modified by vinson
		# self.freq_info = [[940,950]]
		self.freq_info = []
		# modified by vinson
		self.threshold = 200
		self.month_freq_list = []
		self.month_freq_str = []
		self.anteid = '0001'

	#读取系统默认参数
	def read_config(self):
		with open(os.getcwd()+'\\wxpython.json','r') as data:
			config_data = json.load(data)
		if config_data['Spec_dir']=='':
			config_data['Spec_dir']=os.getcwd()+"\\data1"
		return config_data
	
	#读取月报频率段
	def read_month_freq_list(self):
		#从本地数据文件中读取
		#month_freq_dataframe = pandas.read_csv('month_freq_list.csv')
		#从数据库CESHI1.MONTH_FREQ中读取
		connection_oracle = cx_Oracle.connect(self.conn_str1)
		sql_freq = "select START_FREQ, END_FREQ from MONTH_FREQ"
		month_freq_dataframe = pandas.read_sql(sql_freq,connection_oracle)
		month_freq_list = month_freq_dataframe.values.tolist()
		month_freq_str = []
		for freq_list in month_freq_list:
			month_freq_str.append(str(freq_list[0])+'-'+str(freq_list[1])+' MHz')
		return month_freq_list,month_freq_str
	
	#添加月报频率段
	def write_month_freq_list(self,new_freq):
		#存入本地csv文件中
		# month_freq_array = pandas.DataFrame(np.array(self.month_freq_list),columns=['START_FREQ','END_FREQ'])
		# month_freq_array.to_csv('month_freq_list.csv',index=False)
		#存入数据库CESHI1.MONTH_FREQ中
		connection_oracle = cx_Oracle.connect(self.conn_str1)
		cur=connection_oracle.cursor()
		sql_insert_freq = "INSERT INTO MONTH_FREQ(START_FREQ, END_FREQ) VALUES ('%s', '%s')"%(self.month_freq_list[-1][0],self.month_freq_list[-1][1])
		cur.execute(sql_insert_freq)
		connection_oracle.commit()
		cur.close()
		return 
		
	def read_direction_cache(self):
		with open(os.getcwd()+'\\direction.json','r') as data:
			direction_cache = json.load(data)
		return direction_cache
		
	def read_reflect_inf(self):
		with open(os.getcwd()+'\\reflect_inf.json','r') as data:
			reflect_inf = json.load(data)
		return reflect_inf

#	commented by vinson 0831
#	def get_oracle_connection(self):
#		conn_str1 = 'ceshi1/cdk120803@localhost:1521/SOUJIUSUBDB'
#		conn_str2 = 'A110000_RMDSD/cdk120803@localhost:1521/SOUJIUSUBDB'
#		return conn_str1,conn_str2
	
	def test_oracle_connection(self):
		connection_oracle = cx_Oracle.connect(self.conn_str1)
		connection_oracle = cx_Oracle.connect(self.conn_str2)
		return
	
	def compute_distance(self,gps1,gps2):
		R=float(6378137);
		dx = (gps2[0]*np.pi/180-gps1[0]*np.pi/180)*(R*np.cos(gps1[1]*np.pi/180));
		dy = (gps2[1]-gps1[1])*np.pi/180*R
		return np.sqrt(dx*dx+dy*dy)
	
	def station_info_update(self):   # 从RSBT_FREQ 和 RSBT_STATION中读出数据，去除空项和重复项，根据GUID一致的提出来合并
		connection_oracle2 = cx_Oracle.Connection(self.conn_str3)
		#读取台站频率，位置信息
		sql9 = 'select STATION_GUID, FREQ_EFB, FREQ_EFE from RSBT_FREQ'
		sql10 = 'select GUID,STAT_LG,STAT_LA from RSBT_STATION'
		#读取台站表数据，台站id,经纬度，然后去除空值
		station_guid_key = pandas.read_sql(sql10,connection_oracle2)
		station_guid_key_new = station_guid_key.dropna(axis=0,how='any')  #去除空值？
		print (station_guid_key_new.shape)
		station_guid_key_new = station_guid_key_new.rename(columns={'GUID':'STATION_GUID'}) 
		#读取频率表数据，体站id,其实频率，终止频率
		station_guid = pandas.read_sql(sql9,connection_oracle2)
		station_guid_one = station_guid.drop_duplicates()     #去除重复的
		station_guid_new=station_guid_one.dropna(axis=0,how='any')
		print (station_guid_new.shape)
		#将两张表进行链接
		station_info = pandas.merge(station_guid_key_new,station_guid_new,how='left',on='STATION_GUID')
		#存储数据
		station_info.to_csv('taizhan.csv',index=False,sep=',')
		return 
	
	def get_freq_range(self,gps,threshold):
		print ("start get_freq_range in method1")
		[m,n]=self.station_info.shape
		freq_info = []
		for i in range(m):
			lng = self.station_info.loc[i,'STAT_LG']
			lat = self.station_info.loc[i,'STAT_LA']
			distance = self.compute_distance(gps,[lng,lat])
			if distance<threshold:
				start_freq = self.station_info.loc[i,'FREQ_EFB']
				end_freq = self.station_info.loc[i,'FREQ_EFE']
				freq_info.append([start_freq,end_freq])
		print ("finish get frequency in database /get_freq_range")
		return freq_info

	def read_service(self,conn_str2):
		sql1 = "select SERVICEDID,FREQNAME,STARTFREQ,ENDFREQ from  RMBT_SERVICE_FREQDETAIL" 
		#con1 = mdb.connect(self.mysql_config['host'], self.mysql_config['user'], self.mysql_config['password'], '110000_rmdsd')
		con1 = cx_Oracle.connect(conn_str2)
		c = pandas.read_sql(sql1, con1)
		con1.close()
		servicedid = []
		freqname = []
		freqrange_list=[]
		for i in c.values:
			servicedid.append(i[0])
			freqname.append(i[1])
			freqrange_list.append(str(i[2])+'-'+str(i[3])+' MHz')
		return servicedid,freqname,freqrange_list

	def read_mysql_config(self):
		with open(os.getcwd()+'\\mysql.json','r') as data:
			mysql_config=json.load(data)
		return mysql_config
		
	def read_uav_config(self):
		with open(os.getcwd()+'\\uav.json','r') as data:
			uav_config=json.load(data)
		return uav_config

	def test_textCtrl_input(self, input_value, input_unit):
		'''
		   This method is set to get the input_freq_value and the unit is MHz
		   input: input_string, input_unit
		   output: {-1:invalid input, others:output_freq}
		'''
		if input_value.strip()=='':
			return -1
		else:
			try:
				input_freq=float(input_value)
				if input_freq < 0:
					return -1
				if input_unit=='GHz':
					input_freq=input_freq*(10**3)
				elif input_unit=='KHz':
					input_freq=input_freq/(10**3)
				return input_freq
			except (ValueError,TypeError) as e:
				return -1

	def file_to_list(self, file_name):
		new_l = []
		for i in file_name:
			temp = i[:40]
			str1=temp[:19]
			str2=temp[21:]
			time_start = i.index('m')+1
			str1_1 = "{year}{month}{day}".format(year=str1[:4],month=str1[5:7],day=str1[8:10])
			str2_1 = "{year}{month}{day}".format(year=str2[:4],month=str2[5:7],day=str2[8:10])
			result = str1_1+' '+ str1[11:].replace('-',':') + '--' + str2_1 + ' ' + str2[11:].replace('-',':')+'('+i[time_start:]+')'
			new_l.append(result)
		return new_l
		
	def list_to_file(self, str_file):
		str_file = str_file.replace(':','-')
		str_file = str_file.replace(' ','-')
		str_file = list(str_file)
		str_file.insert(4,'-')
		str_file.insert(7,'-')
		str_file.insert(25,'-')
		str_file.insert(28,'-')
		index1 = str_file.index('(')
		index2 = str_file.index(')')
		num = str_file[index1+1:index2]
		str1 = ''.join(str_file[:index1])+'spectrum' + ''.join(num)
		return str1
	###########################################################################
	## draw pictures
	###########################################################################
	def draw_picture(self,x,y,title='',xlabel='',ylabel='',height=4,width=6,face_color='k',gridcolor='y',figure=None):
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
	def instrument_connect(self):
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
	def instrument_disconnect(self,rsa):
		rsa.Disconnect()


	def detectNoise(self,rsa300,startFreq,endFreq,rbw,vbw):
		# 实现方法为：取频谱迹线后，取迹线上的最小值，
		# 然后取15dB以内的所有点取平均值，得到平均噪声电平
		# create Spectrum_Settings data structure
		try:
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
		except:
			return 0
		t1 = time.time()
		while ready.value == False:
			rsa300.SPECTRUM_WaitForDataReady(timeoutMsec, byref(ready))
			t2=time.time()
			if t2-t1>3:
				return 0

		rsa300.SPECTRUM_GetTrace(c_int(0), specSet.traceLength,
								byref(traceData), byref(outTracePoints))
		# print('Got trace data.')

		# convert trace data from a ctypes array to a numpy array
		trace = np.ctypeslib.as_array(traceData)

		# Peak power and frequency calculations
		min_peak = min(trace)
		threshold = min_peak + 15
		# Peak power and frequency calculations
		trace1 = [data for data in trace if data < threshold]
		ave = np.mean(trace1)
		return ave

	#预测带宽
	def bandwidth(self,peakPower,peakFreq,trace,freq):
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

		bandWidth_result = freq[a1] - freq[a2]
		freq_cf = freq[a1]+bandWidth_result/2
		return freq_cf, bandWidth_result

	#预测带宽2
	def bandwidth2(self,peakPower,peakNum,trace,freq):
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
	def bandwidth3(self,peakPower,peakFreq,trace,freq):
		#方法：从低频方向和高频方向分别向中心频率处搜索，找到两边比最大电平小6dB和10dB的点，如果两个带宽差1M以上，则选用-10dB的带宽
		# 否则，选用-6dB的带宽    经常碰到波形畸变严重的情况，-3dB没法正确测得带宽
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
				break
			peak_index1 -= 1
		while peak_index2<len(trace):
			if trace[peak_index2]<trace_max-6:
				pass
			else:
				break
			peak_index2 += 1
		
		#为了防止波形多变,多计算一种类型，两种进行综合分析
		peak_index_new1 = a2
		peak_index_new2 = a1
		while peak_index_new1>0:
			if trace[peak_index_new1]<trace_max-10:
				pass
			else:
				break
			peak_index_new1 -= 1
		while peak_index_new2<len(trace)-1:
			if trace[peak_index_new2]<trace_max-10:
				pass
			else:
				break
			peak_index_new2 += 1
		bandWidth1 = freq[peak_index1] - freq[peak_index2]
		bandWidth_2 = freq[peak_index_new1] - freq[peak_index_new2]
		#print (bandWidth1,bandWidth2)
		if abs(bandWidth1 - bandWidth_2)>1e6:
			bandWidth = bandWidth_2
		else:
			bandWidth = bandWidth1
		freq_cf = freq[a1]+bandWidth/2
		return freq_cf, bandWidth



	# 一次扫频参数：噪声均值、起始频率、终止频率、频率跨度、rbw、vbw,任务名,扫频计数,当前迹线对应时间，\
	# 当前迹线时间，精度，维度, 信号数，子信道数
	def spectrum1(self,rsa300,average, startFreq, stopFreq, span, rbw,vbw, str_time, count, str_tt1, 
				str_tt2,longitude,latitude,num_signal,Sub_cf_all, threshold):
		# create Spectrum_Settings data structure
		try:
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
		except:
			return 0
		
		t1 = time.time()
		while ready.value == False:
			rsa300.SPECTRUM_WaitForDataReady(timeoutMsec, byref(ready))
			t2=time.time()
			if t2-t1>3:
				return 0

		rsa300.SPECTRUM_GetTrace(c_int(0), specSet.traceLength,
								 byref(traceData), byref(outTracePoints))
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
		a = []  # 临时存储功率超过平均值8dB的频率点序列号
		b = []  # 存储信号段的位置，
		c = []  # 临时存储功率超过平均值8dB的频率点的功率值
		d = []  # 存储电平超过阈值的所有信号段的功率值
		freq_list=[] #记录功率超过阈值的数据点对应的频率值
		freq_all=[]
		peakPower=[] #记录峰值
		peakNum=[]  #记录峰值对应的横坐标位置
		# 得到局部数据
		for i in range(801):
			if traceData[i] > average + threshold:
				a.append(i)
				c.append(trace[i])
				freq_list.append(freq[i])
				#print (a)
			elif traceData[i]<average + threshold:   #信号幅度降低到阈值以下时
				if a:
					#print (a)
					b.append(a)    #把信号段放到b和d
					d.append(c)
					peakNum.append(a[np.argmax(c)])    # 添加本段信号峰值位置
					peakPower.append(np.max(c))     # 记录本段信号峰值功率
					freq_all.append(freq_list)     # 添加这一段信号带宽的所有频率点
					a = []		#清空a,c,freq_list准备记录下一段信号
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
		point=[] #画框在整个界面的坐标信息，用于鼠标坐标响应
		point_xy=[]  #画框的坐标信息，用于画框
		for i in range(len(b)):    # 信号序号点
			# 跳过空数据
			if len(b[i]) > 1:
				j1 += 1
				s1_x = freq[b[i][0]]   # 子频率段起始频率
				s1_y = average
				s2_x = freq[b[i][-1]]	# 子频率段终止频率
				s2_y = average + 6
				# s3_x = b[i][0]
				s3_y = np.amax(d[i])
				
				# ？？？ 为什么在这里加span是否大于30M的判断？
				freq_len=freq[-1]-freq[0]
				if freq_len>30e6:
					#print (average,d[i])
					break;
					
				u_x=200000
				#u_x=20000
				u_y=0   #画图与实际的偏置
				# point_x1=540*(s1_x-u_x-freq[0])/freq_len+88
				# point_x2=540*(s2_x+u_x-freq[0])/freq_len+88
				point_x1=570*(s1_x-u_x-freq[0])/freq_len+90
				point_x2=570*(s2_x+u_x-freq[0])/freq_len+90
				point_y1=360*(-s1_y+20)/70+56
				point_y2=360*(-s3_y+20)/70+56
				point.append([point_x1,point_x2,point_y2,point_y1])
				point_xy.append([s1_x-u_x,s2_x+u_x,s1_y,s3_y])

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
		Sub_type=[]         #信号业务类型
		service_name = -1
		#print (len(b))
		for i in range(len(b)):
			if len(b[i]) > 1:
			
				#确定signal_No
				#j += 1
				b_f = float(freq[b[i][0]])
				e_f = float(freq[b[i][-1]])
				
				#print ((float(b_f), float(e_f)))
				sql1 = "select SERVICEDID from RMBT_SERVICE_FREQDETAIL where STARTFREQ <= %s and ENDFREQ >= %s" % (float(b_f/1e6), float(e_f/1e6))
				#con1 = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],'110000_rmdsd')
				con1 = cx_Oracle.Connection(self.conn_str2)
				c = pandas.read_sql(sql1, con1)
				#print (c)
				# modified by vinson，获取业务名给Freq_Name显示
				if len(c) > 0: 
					service_name = c['SERVICEDID'][0]
					sql_FreqName = "select FREQNAME from RMBT_SERVICE_FREQDETAIL where STARTFREQ <= %s and ENDFREQ >= %s" % (float(b_f/1e6), float(e_f/1e6))
					data_sql_FreqName = pandas.read_sql(sql_FreqName, con1)
					#print (c)
					Freq_Name = data_sql_FreqName['FREQNAME'][0]
				else:
					service_name = -1
					Freq_Name = '  '
				#print ('have signals:',i+1)
				#band_t, peak_t, peak_tf ,draw_Sub_Spectrum,draw_Sub_Spectrum2= spectrum0(rsa300,b_f, e_f, str_time, j, count, str_tt1, str_tt2,longitude,latitude,rbw,vbw)

				band_t=e_f-b_f    # 带宽
				peak_t=np.amax(d[i])  # 峰值电平
				peak_tf=freq[b[i][np.argmax(d[i])]]		#峰值频率
				
				#Sub_Spectrum.append(draw_Sub_Spectrum)
				#Sub_Spectrum2.append(draw_Sub_Spectrum2)
				Sub_cf_channel.append((freq[-1]+freq[0])/2)   # 当前扫描帧中心频率
				#Sub_span.append(e_f-b_f)
				Sub_span.append(freq[-1]-freq[0])    # 当前扫描帧span
				Sub_peak.append(peak_t)
				# 输出监测到的信号的真实信号带宽，峰值信息，中心频率
				#print(band_t, peak_t, peak_tf)
				#freq_cf, band = bandwidth2(peakPower[i], peakNum[i], trace, freq)  # 求带宽
				#print (peakPower[i], freq[peakNum[i]], d[i], freq_all[i])
				freq_cf, band = self.bandwidth3(peakPower[i], freq[peakNum[i]], d[i], freq_all[i])  
				# 求带宽 参数：峰值功率，峰值频率，改频段所有功率值，该频段所有频率值
				Sub_band.append(band)    # 检测到信号段的带宽
				Sub_cf.append(freq_cf)   # 检测到信号段的中心频率
				
				#判断信号是否合法
				illegal=0  #0表示非法
				# business_type =0 #表示未知
				for station_freq in self.freq_info:
					if freq_cf>=station_freq[0]*1e6 and freq_cf<=station_freq[1]*1e6:
						illegal=1
						# business_type = '移动'
						break
				Sub_illegal.append(illegal)
				# Sub_type.append(business_type)
				Sub_type.append(Freq_Name)   # add by vinson, 把业务名称添加进sub_type

				#判断是否有重复频段,有则序号加1
				if not Sub_cf_all:
					num_signal=num_signal+1
				else:
					divide_freq=np.array(Sub_cf_all)-freq_cf
					if sum(abs(divide_freq)<=0.5*1e6)==0:
						num_signal=num_signal+1
				
				#con=mdb.connect(mysql_config['host'],mysql_config['user'],mysql_config['password'],mysql_config['database'])
				con = cx_Oracle.Connection(self.conn_str1)
				with con:
					# 获取连接的cursor，只有获取了cursor，我们才能进行各种操作
					#print ('ceshi1')
					cur = con.cursor()
					# 向本地测试数据库写入数据：测试任务名，serviceID，迹线时间，开始频率，结束频率，中心频率，带宽，第几条迹线，
					# 精度，维度，第几个信号，是否合法，峰值电平
					cur.execute("INSERT INTO SPECTRUM_IDENTIFIED VALUES('%s', %s, \
						to_date('%s','yyyy-MM-dd hh24:mi:ss'), %s, %s, %s, %s, %s, %s, %s, %s,%s,%s)"%(str_time, 
						service_name, str_tt1, float(b_f), float(e_f),float(freq_cf),float(band), int(count),
						float(longitude),float(latitude),num_signal,illegal,float(peak_t)))
					cur.close()
				con.commit()
				con.close()
				
		Sub_Spectrum=freq  # 完整扫频频率
		Sub_Spectrum2=trace  # 完整扫频信号
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
		# 返回信息：频率列表，电平列表，信号段中心频率数组，信号段span数组，信号段检测带宽数组，信号段峰值数组，完整扫频频率数据，
		# 完整扫频电平数据，完整迹线频率，完整迹线电平，红框四角，比红框宽一些的范围，检测到信号数量，信号段是否合法标识数组，业务名数组
		return head,data1,Sub_cf_channel,Sub_span,Sub_cf,Sub_band,Sub_peak,Sub_Spectrum,Sub_Spectrum2,freq,traceData,point,point_xy,num_signal,Sub_illegal,Sub_type, service_name
	# 返回原始的频谱数据

	def simple_spectrum(self,rsa300,startFreq,endFreq,rbw,vbw):
		# create Spectrum_Settings data structure
		try:
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
		except:
			return 0
		t1 = time.time()
		while ready.value == False:
			rsa300.SPECTRUM_WaitForDataReady(timeoutMsec, byref(ready))
			t2=time.time()
			if t2-t1>3:
				return 0

		rsa300.SPECTRUM_GetTrace(c_int(0), specSet.traceLength,
								byref(traceData), byref(outTracePoints))
		# print('Got trace data.')

		# convert trace data from a ctypes array to a numpy array
		trace = np.ctypeslib.as_array(traceData)

		# Peak power and frequency calculations
		return freq,trace

	# 绘制实时的无人机频谱图，顺便计算出带宽
	def uav0(self,rsa300,startFreq,endFreq,average,rbw,vbw):
		try:
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
		except:
			return 0
		
		t1 = time.time()
		while ready.value == False:
			rsa300.SPECTRUM_WaitForDataReady(timeoutMsec, byref(ready))
			t2=time.time()
			if t2-t1>3:
				return 0

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
		'''
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
			freq_cf,band=bandwidth3(peakPower[i],peakNum[i],trace,freq)
			freq_cfs.append(freq_cf)
			bands.append(band)
		'''
		freq_cfs=0
		bands=0
		peakPower=0
		return freq_cfs, bands, peakPower,freq,traceData


	def read_file(self,file):
		#path = os.getcwd()+"\\data1\\"+file
		path = self.config_data['Spec_dir']+"\\"+file
		file1 = file[:10]+' '+file[11:13]+':'+file[14:16]+':'+file[17:19]
		file2 = file[21:31]+' '+file[32:34]+':'+file[35:37]+':'+file[38:40]
		#print (file1)
		retain_time=int(file[48:]) #持续时间
		start_time = datetime.datetime.strptime(file1, "%Y-%m-%d %H:%M:%S")
		end_time = datetime.datetime.strptime(file2, "%Y-%m-%d %H:%M:%S")
		sql = "select startFreq,endFreq from minitor_task where Task_Name='%s'"%(file[0:19])
		#print (sql)
		#con = mdb.connect('localhost', 'root', 'cdk120803', 'ceshi1')
		con = cx_Oracle.Connection(self.conn_str1)
		c = pandas.read_sql(sql, con)
		#print (c)
		con.commit()
		con.close()
		start_freq = c['STARTFREQ'][0]
		end_freq = c['ENDFREQ'][0]
		return start_time,end_time,retain_time,float(start_freq),float(end_freq),path

	# 计算频谱占用度必须用到原始格式的数据库
	# 参数：起始时间、终止时间、任务名称、其实频率、终止频率
	def spectrum_occ(self,start_time,stop_time,task_name,freq_start,freq_stop):
		# 输入就是起始时间、终止时间、任务名称、起始频率、终止频率
		spectrum_span = freq_stop - freq_start
		# 以一个小时为单位
		sql3 = "select FREQ_BW, COUNT1 from SPECTRUM_IDENTIFIED where Task_Name='%s' and FREQ_CF between %f and %f" % (task_name, float(freq_start), float(freq_stop)) + " and START_TIME between to_date('%s'," %(start_time) + "'yyyy-MM-dd hh24:mi:ss')" + "and to_date('%s'," %(stop_time) + "'yyyy-MM-dd hh24:mi:ss')"
		#con=mdb.connect(mysql_config['host'],mysql_config['user'],mysql_config['password'],mysql_config['database'])
		con = cx_Oracle.Connection(self.conn_str1) 
		#con = mdb.connect('localhost', 'root', 'cdk120803', 'ceshi1')
		c = pandas.read_sql(sql3, con)
		#print ('spectrum success')
		con.commit()
		con.close()
		spectrum_occ1 = sum(c['FREQ_BW'])
		if len(c['COUNT1'])>0:
			c1 = np.array(c['COUNT1'])
			num = max(c['COUNT1']) - min(c['COUNT1'])+1
			spectrum_occ = spectrum_occ1 / (float(spectrum_span)*num)*100
		else:
			#print ('count:',len(c['COUNT1']))
			spectrum_occ = 0
		return spectrum_occ  # 返回频谱占用度

	# 导入导出数据
	# 粗扫描，返回某一次任务的初试频率和终止频率以及频率和功率矩阵,起始时间和终止时间
	def importData_cu(self,task_name,file_name,raw_path):
		#task_name就是文件夹的名字，filename就是粗扫的csv文件名字，raw_path就是文件存储的路径
		sql = "select startFreq,endFreq,Task_B,Task_E from minitor_task where Task_Name='%s'"%(task_name[0:19])
		#con=mdb.connect(mysql_config['host'],mysql_config['user'],mysql_config['password'],mysql_config['database'])
		con = cx_Oracle.Connection(self.conn_str1)
		#con = mdb.connect('localhost', 'root', 'cdk120803', 'ceshi1')
		c_cu = pandas.read_sql(sql, con)
		con.commit()
		con.close()
		start_freq_cu = c_cu['STARTFREQ'][0]
		end_freq_cu = c_cu['ENDFREQ'][0]
		start_time_cu = c_cu['TASK_B'][0]
		end_time_cu=c_cu['TASK_E'][0]
		path = raw_path+"//"+file_name
		df_cu = pandas.read_csv(path)
		columns=df_cu.columns
		#num = int(len(df_cu)/801)
		#x_mat = np.array(df_cu['frequency']).reshape(num,801)
		#y_mat = np.array(df_cu['power']).reshape(num, 801)
		detail_information = np.array(df_cu[columns[:6]].values)
		x_mat = np.array(columns[6:],np.float64)
		y_mat = np.array(df_cu[columns[6:]].values,np.float64)
		#记录每个峰段的坐标信息
		point_xy = []
		left_freq = x_mat[0]
		freq_len = x_mat[-1] - x_mat[0]
		# u_x = 20000
		u_x = 0    # 鼠标触发区域的频率轴范围扩展
		u_y = 10   # 鼠标触发提示的幅度轴扩展
		[m,n]=y_mat.shape
		for i in range(m-1):
			s1_x = detail_information[i][1]-detail_information[i][3]   # cf-band
			s2_x = detail_information[i][1]+detail_information[i][3]	# cf + band
			s1_y = detail_information[i][2]  	# band
			point_x1=570*(s1_x-u_x-left_freq)/freq_len+90
			point_x2=570*(s2_x+u_x-left_freq)/freq_len+90
			point_y1=360*(-s1_y+u_y)/80+56
			point_y2=360*(-s1_y-u_y)/80+56
			point_xy.append([point_x1,point_x2,point_y2,point_y1])
		print (point_xy)
		# 生成频谱图上的峰值提示信息
		label_information=[str(detail_information[u][0])+'\n'+'cf:'+'%.2f'%(detail_information[u][1]/1e6)+'MHz\n'+'peak:'+'%.2f'%(detail_information[u][2])+'dBmV\n'+'band:'+'%.3f'%(detail_information[u][3]/1e6)+'MHz' for u in range(m-1)]
		return start_freq_cu,end_freq_cu,x_mat,y_mat,start_time_cu,end_time_cu,detail_information,point_xy,label_information


	# 测向输入就是一个频率输出就是对应的强度
	def find_direction(self,rsa300,freq,span,reflevel,rbw):

		try:
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
			refLevel = c_double(reflevel)  # ref level
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
			specSet.span = c_double(span)
			specSet.rbw = c_double(rbw)
			# specSet.enableVBW =
			# specSet.vbw =
			specSet.traceLength = c_int(801)
			# specSet.window =
			specSet.verticalUnit = c_int(4)
			specSet.actualStartFreq = c_double(freq-span/2)
			specSet.actualStopFreq = c_double(freq+span/2)
			# specSet.actualFreqStepSize =c_double(50000.0)
			# specSet.actualRBW =
			# specSet.actualVBW =
			# specSet.actualNumIQSamples =

			# set desired spectrum settings
			rsa300.SPECTRUM_SetSettings(specSet)
			rsa300.SPECTRUM_GetSettings(byref(specSet))

			# uncomment this if you want to print out the spectrum settings

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
		except:
			t1 = time.time()
			while ready.value == False:
				rsa300.SPECTRUM_WaitForDataReady(timeoutMsec, byref(ready))
				t2=time.time()
				if t2-t1>3:
					return 0

		rsa300.SPECTRUM_GetTrace(c_int(0), specSet.traceLength,
								 byref(traceData), byref(outTracePoints))
		print('Got trace data.')

		# convert trace data from a ctypes array to a numpy array
		trace = np.ctypeslib.as_array(traceData)

		# Peak power and frequency calculations
		peakPower = np.amax(trace)
		peakPowerFreq = freq[np.argmax(trace)]
		# print('Peak power in spectrum: %4.3f dBmV @ %d Hz' % (peakPower, peakPowerFreq))
		return peakPower,freq,trace

	#################################################
	####地图功能模块

	#返回的result1是台站信息一个字典{编号：[经度，维度，[状态信息，带宽，起始频率，终止频率，调制方式]]}
	#返回的result2是测试点信息是一个list[[测试点1的经纬度],[测试点2的经纬度]，.....[]]
	def get_station_inf(self):
		sql1 = "select STAT_TYPE,FREQ_EFB,FREQ_LC,FREQ_UC,FREQ_MOD,STAT_LG,STAT_LA from rsbt_station "
		sql2 = "select LOGITUDE,LATITUDE from spectrum_identified"
		#con1 = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],mysql_config['database'])
		con1 = cx_Oracle.Connection(self.conn_str1)
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



	# 取测试数据库数据统计写入月报数据库 RMBT_FACILITY_FREQBAND_EMENV数据表
	def rmbt_facility_freqband_emenv(self,task_name,span,start_time,end_time,longitude,latitude, serviceid, threshold,
									mfid='11000001400001', statismode='04',
									address='aasfasdfasfasdf',occ2=0,height=1):
		sql2 = "select FreQ_BW,COUNT1,legal from SPECTRUM_IDENTIFIED where Task_Name='%s'" % (
			task_name) + "and Start_time between to_date('%s'," % (
			start_time) +"'yyyy-MM-dd hh24:mi:ss')" + "and to_date('%s'," % (end_time) +"'yyyy-MM-dd hh24:mi:ss')"
		#con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],mysql_config['database'])
		con = cx_Oracle.Connection(self.conn_str1)    # 取ceshi数据库中spectrum_identified数据进行统计写入月报数据库
		inf = pandas.read_sql(sql2, con)
		statisstartday = str(start_time)
		statisendday = str(end_time)
		latitude = Decimal(latitude).quantize(Decimal('0.000000000'))
		longitude = Decimal(longitude).quantize(Decimal('0.000000000'))
		frame_num = len(inf['COUNT1'].drop_duplicates())
		sig = np.sum(inf['FREQ_BW'].values)
		if frame_num != 0:
			occ = (float(sig)*100) / (span*frame_num)   #出现信号的总带宽占总监测span的比例
		else:
			occ = 0
		bandoccupancy = Decimal(occ).quantize(Decimal('0.00'))
		threshold = Decimal(threshold).quantize(Decimal('0.00'))
		height = Decimal(height).quantize(Decimal('0.00'))
		######legal=1是合法0是非法
		legal_sig = inf[inf['LEGAL']==0]   #合法台站数据集
		legal_frame_num = len(legal_sig['COUNT1'])   # 发现合法台站信号的帧数
		illegal_sig = inf[inf['LEGAL']==1]  #非法台站数据集
		illegal_frame_num = len(illegal_sig['COUNT1'])  # 非法台站帧数
		# print (legal_frame_num,legal_frame_num1)
		legal_band = np.sum(legal_sig['FREQ_BW'].values)
		illegal_band = np.sum(illegal_sig['FREQ_BW'].values)
		if legal_frame_num == 0:   
			# occ1 = 0.01    #为什么不是0？    合法台站频谱占用度
			occ1 = 0
		else:
			occ1 = float(legal_band*100)/(span*legal_frame_num)
		if illegal_frame_num == 0:
			occ2 = 0
		else:
			occ2 = float(illegal_band*100)/(span*illegal_frame_num)
		occ1 = Decimal(occ1).quantize(Decimal('0.00'))
		occ2 = Decimal(occ2).quantize(Decimal('0.00'))
		occ3  =100 - occ1-occ2
		occ3 = Decimal(occ3).quantize(Decimal('0.00'))
		# print (occ1,occ2,occ3)
		con.close()
		#con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],'110000_rmdsd')
		con = cx_Oracle.Connection(self.conn_str2)
		with con:
			# 获取连接的cursor，只有获取了cursor，我们才能进行各种操作
			cur = con.cursor()  # 一条游标操作就是一条数据插入，第二条游标操作就是第二条记录，所以最好一次性插入或者日后更新也行
			cur.execute("INSERT INTO rmbt_facility_freqband_emenv VALUES('%s',to_date('%s',\
						'yyyy-MM-dd hh24:mi:ss'),to_date('%s','yyyy-MM-dd hh24:mi:ss'),'%s','%s',\
						%s,%s,%s,%s,%s,%s,%s,%s,'%s')"
						%(mfid, statisstartday, statisendday, statismode, serviceid, bandoccupancy, 
						threshold, occ1, occ2, occ3,latitude, longitude, height, address))
			cur.close()
		con.commit()
		con.close()

	# 统计测试信号数据写入月报数据库 RMBT_FACILITY_FREQ_EMENV数据表
	def rmbt_facility_freq_emenv3(self,task_name,start_time,end_time,ssid,servicedid,threshold,mfid='11000001400001',
								statismode='04',amplitudeunit='01'):
		#sql2 = "select Signal_No, Start_time, FREQ_CF, FreQ_BW, COUNT1, peakpower, channel_no from spectrum_identified where Task_Name='%s'" % (task_name)
		sql2 = "select COUNT1,Signal_No,FREQ_CF,FreQ_BW,peakpower from SPECTRUM_IDENTIFIED where Task_Name='%s'" % (
			task_name) + "and Start_time between to_date('%s'," % (
			start_time) +"'yyyy-MM-dd hh24:mi:ss')" + "and to_date('%s'," % (end_time) +"'yyyy-MM-dd hh24:mi:ss')"
		#con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],mysql_config['database'])
		con = cx_Oracle.Connection(self.conn_str1)    # ceshi库
		inf = pandas.read_sql(sql2, con)
		df = inf
		con.close()
		list1 = df['SIGNAL_NO'].drop_duplicates().values
		df_r = []
		frame_num = len(df['COUNT1'].drop_duplicates())
		for i in range(len(list1)):
			df1 = df[df['SIGNAL_NO'] == list1[i]]
			df1["index"] = range(len(df1))
			df1 = df1.set_index(["index"])
			df_r.append(df1)
		for sig in range(len(df_r)):
			if frame_num != 0:
				occ = len(df_r[sig]['COUNT1'].drop_duplicates()) / float(frame_num)
			else:
				occ = 0
			occ = Decimal(occ * 100).quantize(Decimal('0.00'))
			'''
			if occ == 1:
				occ = Decimal(occ).quantize(Decimal('0.00'))
			else:
				occ = Decimal(occ * 100).quantize(Decimal('0.00'))
			'''
			statisstartday = str(start_time)
			statisendday = str(end_time)
			#servicedid = df_r[sig]['SIGNAL_NO'][0]
			cf = df_r[sig]['FREQ_CF'].values
			cf_avg = np.average(cf) / 1e6
			cf_avg = Decimal(cf_avg).quantize(Decimal('0.0000000'))
			bandwidth = df_r[sig]['FREQ_BW'].values
			band_avg = np.average(bandwidth) / 1e6
			band_avg = Decimal(band_avg).quantize(Decimal('0.0000000'))
			maxamplitude = np.max(df_r[sig]['PEAKPOWER'].values)
			maxamplitude = Decimal(maxamplitude).quantize(Decimal('0.00'))
			temp = np.sort(df_r[sig]['PEAKPOWER'])
			mid = temp[int(len(temp) / 2)]
			midamplitude = mid
			midamplitude = Decimal(midamplitude).quantize(Decimal('0.00'))
			threshold = Decimal(threshold).quantize(Decimal('0.00'))
			#con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],'110000_rmdsd')
			con = cx_Oracle.Connection(self.conn_str2)
			with con:
				# 获取连接的cursor，只有获取了cursor，我们才能进行各种操作
				cur = con.cursor()  # 一条游标操作就是一条数据插入，第二条游标操作就是第二条记录，所以最好一次性插入或者日后更新也行
				cur.execute("INSERT INTO rmbt_facility_freq_emenv VALUES('%s',to_date('%s','yyyy-MM-dd hh24:mi:ss'),\
							to_date('%s','yyyy-MM-dd hh24:mi:ss'),'%s','%s',%s,%s,'%s','%s',%s,%s,%s,%s)" %(str(mfid), statisstartday, statisendday, str(statismode), str(servicedid), cf_avg, band_avg,str(ssid),str(amplitudeunit), maxamplitude, midamplitude, occ, threshold))
				cur.close()
			con.commit()


	# 生成整个监测过程中的最大合成帧，并统计该帧的频谱占用度
	def rmbt_freq_occupancy(self,span,start_time,end_time,startFreq,stopFreq,longitude,latitude,height,trace1,head,average,threshold,mfid='11000001400001',addr='aasfasdfasfasdf',amplitudeunit='01'):
		step = span / float(801)
		startFreq1 = Decimal(startFreq/1e6).quantize(Decimal('0.0000000'))
		stopFreq1 = Decimal(stopFreq/1e6).quantize(Decimal('0.0000000'))
		step1 = Decimal(step).quantize(Decimal('0.0000000'))
		longitude1 = Decimal(longitude).quantize(Decimal('0.000000000'))
		latitude1 = Decimal(latitude).quantize(Decimal('0.000000000'))
		height1 = Decimal(height).quantize(Decimal('0.00'))
		#con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],'110000_rmdsd')
		con = cx_Oracle.Connection(self.conn_str2)
		# con = mdb.connect('localhost', 'root', 'cdk120803', 'ceshi1')
		z = pandas.DataFrame({})
		point = np.arange(0,801,1)
		z['freq_point'] = point
		#z['freq_startFreq'] = trace1['frequency'][0:801]
		z['freq_startFreq'] = head #频率
		peak_list = []
		mid_list = []
		occ1 = []
		for i in range(len(z['freq_startFreq'])):
			#temp = trace1[trace1['frequency']==i]
			temp = trace1[:,i]
			temp1 = np.sort(temp)
			#peak = np.max(temp)
			peak = temp1[-1]
			mid = temp1[int(len(temp)/2)]
			occ = len(temp[temp>average+threshold])/float(len(temp))
			occ = round(occ,2)
			occ1.append(occ*100)   #与数据库存储格式不符，应乘以10000
			mid_list.append(mid)
			peak_list.append(peak)
		z['Max_peak'] = peak_list
		z['mid'] = mid_list
		z['occ'] = occ1
		z['threshold'] = threshold
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
			cur.setinputsizes(blobData=cx_Oracle.BLOB)
			sql = "INSERT INTO RMBT_FREQ_OCCUPANCY VALUES('%s',to_date('%s','yyyy-MM-dd hh24:mi:ss'),to_date('%s','yyyy-MM-dd hh24:mi:ss'),\
														%s,%s,%s,%s,%s,%s,'%s','%s',:blobData)"%(mfid,start_time, end_time, 
														startFreq1, stopFreq1, step1, longitude1, latitude1, height1, addr, amplitudeunit)
			cur.execute(sql,{'blobData':load_file})
			cur.close()
		con.commit()
		con.close()

	#台站数据获取
	def taizhan_out(self,task_name):
		sql = "select FREQ_CF,FreQ_B,FreQ_E,FreQ_BW,LOGITUDE,LATITUDE,peakpower,legal,channel_no from spectrum_identified where Task_Name='%s'" % (task_name)
		#print (sql)
		#print (mysql_config1)
		#con = mdb.connect(mysql_config['host'], mysql_config['user'], mysql_config['password'],mysql_config['database'])
		con = cx_Oracle.Connection(self.conn_str1)
		inf = pandas.read_sql(sql, con)
		#print (inf)
		inf1 = inf.drop_duplicates(['CHANNEL_NO'])
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



	def get_GPS(self):
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


	def get_GPS2(self,rsa,GPS_q):
		eventID = c_int(1)  # 0:overrange, 1:ext trig, 2:1PPS
		eventOccurred = c_bool(False)
		eventTimestamp = c_uint64(0)
		hwInstalled = c_bool(False)

		"""#################SEARCH/CONNECT#################"""
		hwInstalled = self.setup_gnss(rsa)

		"""#################CONFIGURE INSTRUMENT#################"""
		rsa.CONFIG_Preset()

		rsa.DEVICE_Run()

		if hwInstalled == True:
			"""#######USE THIS IF YOU HAVE AN RSA500/600 WITH GPS ANTENNA########"""
			print('Waiting for internal 1PPS.')
			while eventOccurred.value == False:
				rsa.GNSS_Get1PPSTimestamp(byref(eventOccurred), byref(eventTimestamp))
			nmeaMessage = self.get_gnss_message(rsa)
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

	def search_connect(self,rsa):
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


	def setup_gnss(self,rsa, system=2):
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

	def get_gnss_message(self,rsa):
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

if __name__=='__main__':
	method = method1()
	input_value = method.test_textCtrl_input('23.0','KHz')
	print (input_value)
