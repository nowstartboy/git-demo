import wx
import wx.xrc
import wx.grid
import wx.lib.buttons as buttons
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import my_xep
import threading
import numpy as np
from pymoduleconnector import ModuleConnector

class StartDetectionThread(threading.Thread):     #运行
	"""
	This just simulates vital signs detection
	a message to the GUI thread.
	"""
	def __init__(self,window,fps,dac_min,dac_max,frame_area,area_offset,device,baseband,buffer_size):
		threading.Thread.__init__(self)
		self.window = window
		self.fps = fps
		self.dac_min = dac_min
		self.dac_max = dac_max
		self.frame_area = frame_area
		self.area_offset = area_offset
		self.device = device
		self.baseband = baseband
		self.buffer_size = buffer_size
		#self.q=q
		self.timeToQuit = threading.Event()
		self.timeToQuit.clear()
		self._running = True

	def stop(self):
		self._running = False


	def run(self):#运行一个线程
		xep_operate = my_xep.Connect_xep(self.fps,self.dac_min,self.dac_max,self.frame_area,self.area_offset,self.device,self.baseband)
		xep_operate.reset(self.device)
		self.mc = ModuleConnector(self.device)
		self.r=xep_operate.set_parameters(self.mc)
		
		#存储数据
		fI=np.linspace(0,self.fps-self.fps/self.buffer_size,self.buffer_size)
		start_freq,mid_freq,end_freq = 0.1,0.8,2 #体征信号频率范围为0.1Hz-2Hz
		start_point,mid_point,end_point = int(start_freq*self.buffer_size/self.fps),int(mid_freq*self.buffer_size/self.fps),int(end_freq*self.buffer_size/self.fps)
		print (start_point,end_point)
		
		myBuffer = my_xep.data_cache(self.buffer_size)
		vital_detection = my_xep.data_process(self.fps)
		self.show_list = [0]*self.buffer_size  #原始信号
		self.resp_list = [0]*(self.buffer_size*2) #呼吸信号
		self.heart_list = [0]*(self.buffer_size*2) #心跳信号
		
		iter = 1
		while self._running:
			new_frame = abs(xep_operate.read_frame(self.r))
			new_frame = list(new_frame)
			myBuffer.add(new_frame)
			if iter%50==0:
				print (len(myBuffer.buffer))
			if iter>=self.buffer_size:
				freq_rf,freq_hf = vital_detection.freq_get(np.transpose(np.array(myBuffer.buffer)),fI,start_point,mid_point,end_point,self.buffer_size)
				freq_rf_num = '%.2f' %(freq_rf*60)
				freq_hf_num = '%.2f' %(freq_hf*60)
				print ('Rf is :',freq_rf_num,freq_hf_num)
				self.window.m_textCtrl2.SetValue(freq_rf_num)
				self.window.m_textCtrl3.SetValue(freq_hf_num)
				#动态导入接收的信号
				if iter%self.buffer_size == 0:
					r_signal,h_signal = vital_detection.signal_divide(self.show_list)
					#r_signal = r_signal/max(abs(r_signal))
					r_signal = np.array(self.show_list)-np.mean(self.show_list)
					r_signal = r_signal/max(abs(r_signal))
					h_signal = h_signal/max(abs(h_signal))
					self.resp_list[self.buffer_size:]=r_signal
					self.heart_list[self.buffer_size:]=h_signal
				self.resp_list = self.resp_list[1:]+[0]
				self.heart_list = self.heart_list[1:]+[0]
				#动态显示呼吸信号
				self.window.axes_score.set_xlim(0*self.fps,self.buffer_size/self.fps)
				#self.window.axes_score.set_ylim(0,max(self.resp_list[0:self.buffer_size])*1.2)
				self.window.l_user.set_data(np.arange(200)/self.fps,self.resp_list[0:self.buffer_size])
				self.window.axes_score.draw_artist(self.window.l_user)
				#动态显示心跳信息
				self.window.axes_score_new.set_xlim(0*self.fps,self.buffer_size/self.fps)
				self.window.r_user.set_data(np.arange(200)/self.fps,self.heart_list[0:self.buffer_size])
				self.window.axes_score.draw_artist(self.window.r_user)
				self.window.canvas.draw()
			self.show_list = self.show_list[1:]+[new_frame[vital_detection.maxDoor]]
			iter += 1
		self.r.x4driver_set_fps(0)

class MyPanel1 ( wx.Panel ):
	"""
	This class is set as the main_interface including:
	1、axes display 
	2、result_data(RPM,HPM) display 
	3、parameters seting module
	4、starting and stopping button
	"""
	def __init__( self, parent ):
		wx.Panel.__init__ ( self, parent, id = wx.ID_ANY, pos = wx.DefaultPosition, size = wx.Size( 500,349 ), style = wx.TAB_TRAVERSAL )
		
		bSizer173 = wx.BoxSizer( wx.HORIZONTAL )
		
		################################################################添加体征信号显示图
		bSizer100 = wx.BoxSizer( wx.VERTICAL )
		self.m_panel9 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer100.Add( self.m_panel9, 1, wx.EXPAND |wx.ALL, 5 )
		
		self.figure_score = Figure((7.5,6.4),100)
		self.canvas = FigureCanvas(self.m_panel9, wx.ID_ANY, self.figure_score)
		# self.figure_score.set_figheight(4)
		# self.figure_score.set_figwidth(6)
		self.axes_score = self.figure_score.add_subplot(211,facecolor='k')
		#self.axes_score.set_autoscale_on(False) #关闭坐标轴自动缩放
		self.traceData=[-10]*801
		self.start_freq=float(2.4e9)
		self.end_freq=float(2.46e9)
		self.l_user,=self.axes_score.plot([],[],'y')
		self.axes_score.set_ylim(-1.5,1.5)
		self.axes_score.set_xlim(0,200)
		self.axes_score.set_title('Respiratory-signals')
		self.axes_score.grid(True,color='w')
		self.axes_score.set_ylabel('Amplitude/dBm')
		#self.canvas.draw()
		
		#######加入第二个子图
		self.axes_score_new = self.figure_score.add_subplot(212,facecolor='k')
		#self.axes_score_new.set_autoscale_on(False) #关闭坐标轴自动缩放
		#self.freq=arange(self.start_freq,self.end_freq,(self.end_freq-self.start_freq)/float(801))
		#self.r_user,=self.axes_score_new.plot(self.freq, self.traceData, 'b')
		self.r_user,=self.axes_score_new.plot([],[],'y')
		#self.axes_score_new.axhline(y=average, color='r')
		self.axes_score_new.set_ylim(-1.5,1.5)
		self.axes_score_new.set_xlim(0,200)
		self.axes_score_new.set_title('Heart-signals')
		self.axes_score_new.grid(True,color='w')
		self.axes_score_new.set_xlabel('t/s')
		self.axes_score_new.set_ylabel('Amplitude/dBm')
		
		# ymajorLocator= ticker.MultipleLocator(50) #将y轴主刻度标签设置为1的倍数
		# yminorLocator= ticker.MultipleLocator(5) #将y轴主刻度标签设置为1的倍数
		# self.axes_score_new.yaxis.set_minor_locator(yminorLocator)
		# self.axes_score_new.yaxis.set_major_locator(ymajorLocator)
		
		self.canvas.draw()
		
		bSizer173.Add( bSizer100, 6, wx.EXPAND, 5 )
		
		########################################################分隔线
		self.m_staticline15 = wx.StaticLine( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_VERTICAL )
		bSizer173.Add( self.m_staticline15, 0, wx.EXPAND |wx.ALL, 0 )
		
		########################################################添加数值显示模块
		bSizer176 = wx.BoxSizer( wx.VERTICAL )     #数值显示
		
		self.bx1=wx.StaticBox( self, wx.ID_ANY, u"体征信号结果显示" )
		#bx1.SetBackgroundColour("MEDIUM GOLDENROD")
		self.bx1.SetForegroundColour("SLATE BLUE")
		self.font = wx.Font(10, wx.DECORATIVE,wx.NORMAL,wx.BOLD)
		self.bx1.SetFont(self.font)
		self.sbSizer1 = wx.StaticBoxSizer(self.bx1, wx.VERTICAL )
		
		self.bSizer15 = wx.BoxSizer( wx.HORIZONTAL )
		self.m_staticText4 = wx.StaticText( self, wx.ID_ANY, u"RPM ：", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText4.Wrap( -1 )
		self.bSizer15.Add( self.m_staticText4, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.m_textCtrl2 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, (80,20), 0 )
		self.bSizer15.Add( self.m_textCtrl2, 0, wx.ALIGN_CENTER|wx.ALL, 0 )
		self.sbSizer1.Add( self.bSizer15, 1, wx.EXPAND, 5 )
		
		self.bSizer16 = wx.BoxSizer( wx.HORIZONTAL )
		self.m_staticText5 = wx.StaticText( self, wx.ID_ANY, u"HPM ：", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText5.Wrap( -1 )
		self.bSizer16.Add( self.m_staticText5, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.m_textCtrl3 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, (80,20), 0 )
		self.bSizer16.Add( self.m_textCtrl3, 0, wx.ALIGN_CENTER|wx.ALL, 0 )
		self.sbSizer1.Add( self.bSizer16, 1, wx.EXPAND, 5 )
		
		bSizer176.Add( self.sbSizer1, 2, wx.ALL, 0 )
		
		
		###########################################################参数配置模块
		self.bx2=wx.StaticBox( self, wx.ID_ANY, u"参数设置" )
		#bx1.SetBackgroundColour("MEDIUM GOLDENROD")
		self.bx2.SetForegroundColour("SLATE BLUE")
		self.font1 = wx.Font(10, wx.DECORATIVE,wx.NORMAL,wx.BOLD)
		self.bx2.SetFont(self.font1)
		self.sbSizer2 = wx.StaticBoxSizer(self.bx2, wx.VERTICAL )
		
		
		self.bSizer17 = wx.BoxSizer( wx.HORIZONTAL )
		self.m_staticText6 = wx.StaticText( self, wx.ID_ANY, u"area  :", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText6.Wrap( -1 )
		self.bSizer17.Add( self.m_staticText6, 1, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.m_textCtrl4 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, (60,20), 0 )
		self.bSizer17.Add( self.m_textCtrl4, 0, wx.ALIGN_CENTER|wx.ALL, 0 )
		self.m_staticText12 = wx.StaticText( self, wx.ID_ANY, u" -", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText12.Wrap( -1 )
		self.bSizer17.Add( self.m_staticText12, 1, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.m_textCtrl7 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, (60,20), 0 )
		self.bSizer17.Add( self.m_textCtrl7, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.m_staticText8 = wx.StaticText( self, wx.ID_ANY, u"m", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText8.Wrap( -1 )
		self.bSizer17.Add( self.m_staticText8, 1, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.sbSizer2.Add( self.bSizer17, 1, wx.EXPAND, 5 )
		
		self.bSizer18 = wx.BoxSizer( wx.HORIZONTAL )
		self.m_staticText7 = wx.StaticText( self, wx.ID_ANY, u"offset:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText7.Wrap( -1 )
		self.bSizer18.Add( self.m_staticText7, 1, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.m_textCtrl5 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, (80,20), 0 )
		self.bSizer18.Add( self.m_textCtrl5, 0, wx.ALIGN_CENTER|wx.ALL, 0 )
		self.m_staticText9 = wx.StaticText( self, wx.ID_ANY, u"m                ",wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText9.Wrap( -1 )
		self.bSizer18.Add( self.m_staticText9, 1, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.sbSizer2.Add( self.bSizer18, 1, wx.EXPAND, 5 )
		
		self.bSizer19 = wx.BoxSizer( wx.HORIZONTAL )
		self.m_staticText10 = wx.StaticText( self, wx.ID_ANY, u" fps  :", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText10.Wrap( -1 )
		self.bSizer19.Add( self.m_staticText10, 1, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.m_textCtrl6 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, (80,20), 0 )
		self.bSizer19.Add( self.m_textCtrl6, 0, wx.ALIGN_CENTER|wx.ALL, 0 )
		self.m_staticText11 = wx.StaticText( self, wx.ID_ANY, u"Hz                 ", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText11.Wrap( -1 )
		self.bSizer19.Add( self.m_staticText11, 1, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.sbSizer2.Add( self.bSizer19, 1, wx.EXPAND, 5 )
		
		#配置三个进度条
		'''
		self.bSizer17 = wx.BoxSizer( wx.HORIZONTAL )
		self.m_staticText6 = wx.StaticText( self, wx.ID_ANY, u"area ：", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText6.Wrap( -1 )
		self.bSizer17.Add( self.m_staticText6, 1, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.m_slider1 = wx.Slider( self, wx.ID_ANY, 50, 0, 100, wx.DefaultPosition, wx.DefaultSize, wx.SL_HORIZONTAL )
		self.bSizer17.Add( self.m_slider1, 4, wx.ALL|wx.ALIGN_CENTER, 0 )
		self.m_staticText8 = wx.StaticText( self, wx.ID_ANY, u"5 m", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText8.Wrap( -1 )
		self.bSizer17.Add( self.m_staticText8, 1, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.sbSizer2.Add( self.bSizer17, 1, wx.EXPAND, 5 )
		
		self.bSizer18 = wx.BoxSizer( wx.HORIZONTAL )
		self.m_staticText7 = wx.StaticText( self, wx.ID_ANY, u"offset:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText7.Wrap( -1 )
		self.bSizer18.Add( self.m_staticText7, 1, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.m_slider2 = wx.Slider( self, wx.ID_ANY, 50, 0, 100, wx.DefaultPosition, wx.DefaultSize, wx.SL_HORIZONTAL )
		self.bSizer18.Add( self.m_slider2, 4, wx.ALL|wx.ALIGN_CENTER, 0 )
		self.m_staticText9 = wx.StaticText( self, wx.ID_ANY, u"10 Hz", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText9.Wrap( -1 )
		self.bSizer18.Add( self.m_staticText9, 1, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.sbSizer2.Add( self.bSizer18, 1, wx.EXPAND, 5 )
		
		self.bSizer19 = wx.BoxSizer( wx.HORIZONTAL )
		self.m_staticText10 = wx.StaticText( self, wx.ID_ANY, u"fps :  ", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText10.Wrap( -1 )
		self.bSizer19.Add( self.m_staticText10, 1, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.m_slider3 = wx.Slider( self, wx.ID_ANY, 50, 0, 100, wx.DefaultPosition, wx.DefaultSize, wx.SL_HORIZONTAL )
		self.bSizer19.Add( self.m_slider3, 4, wx.ALL|wx.ALIGN_CENTER, 0 )
		self.m_staticText11 = wx.StaticText( self, wx.ID_ANY, u"Hz", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText11.Wrap( -1 )
		self.bSizer19.Add( self.m_staticText11, 1, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.sbSizer2.Add( self.bSizer19, 1, wx.EXPAND, 5 )
		'''
		
		bSizer176.Add( self.sbSizer2, 3, wx.ALL, 0 )
		
		
		##############################################################功能按钮模块
		self.bx3=wx.StaticBox( self, wx.ID_ANY, u"功能按钮" )
		#bx1.SetBackgroundColour("MEDIUM GOLDENROD")
		self.bx3.SetForegroundColour("SLATE BLUE")
		self.font2 = wx.Font(10, wx.DECORATIVE,wx.NORMAL,wx.BOLD)
		self.bx3.SetFont(self.font2)
		self.sbSizer3 = wx.StaticBoxSizer(self.bx3, wx.VERTICAL )
		
		
		bSizer174 = wx.BoxSizer(wx.HORIZONTAL)
		self.m_button2=wx.Button(self,wx.ID_ANY,u"开始检测",wx.DefaultPosition,(120,25),0)
		bSizer174.Add(self.m_button2,1,wx.ALL|wx.ALIGN_CENTER,0)
		self.m_button2.Bind(wx.EVT_BUTTON,self.start_detection)
		self.m_button3=wx.Button(self,wx.ID_ANY,u"停止检测",wx.DefaultPosition,(120,25),0)
		bSizer174.Add(self.m_button3,1,wx.ALL|wx.ALIGN_CENTER,0)
		self.m_button3.Bind(wx.EVT_BUTTON,self.stop_detection)
		self.sbSizer3.Add( bSizer174, 0, wx.ALL, 0 )
		
		bSizer176.Add( self.sbSizer3, 1, wx.ALL, 0 )
		
		
		############################################################## 占位
		self.m_panel1 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer176.Add( self.m_panel1, 6, wx.EXPAND |wx.ALL, 5 )

		
		bSizer173.Add( bSizer176, 1, wx.EXPAND, 5 )
		self.SetSizer( bSizer173 )
		
		############################################################## 定义变量
		self.start_threads = []

	def get_Input(self):
		"""
		This function is set to get the input information:
		detection_area: area_start,area_stop
		offset:area_offset
		fps:fps
		"""
		self.input_state = 0
		area_string1 = self.m_textCtrl4.GetValue()
		area_string2 = self.m_textCtrl7.GetValue()
		offset_string = self.m_textCtrl5.GetValue()
		fps_string = self.m_textCtrl6.GetValue()
		
		#如果area输入有一个为空，则默认为0.2-5m
		if area_string1=='' or area_string2=='':
			area_start = 0.2
			area_stop = 5.0
			self.m_textCtrl4.SetValue(str(area_start))
			self.m_textCtrl7.SetValue(str(area_stop))
		else:
			try:
				area_start = float(area_string1)
				area_stop = float(area_string2)
			except (ValueError,TypeError) as e:
				wx.MessageBox(u' area输入须为数值', "Message" ,wx.OK | wx.ICON_INFORMATION)
				return
		
		#获取area_offset
		if offset_string=='':
			area_offset = 0.2
			self.m_textCtrl5.SetValue(str(area_offset))
		else:
			try:
				area_offset = float(offset_string)
			except (ValueError,TypeError) as e:
				wx.MessageBox(u' area_offset输入须为数值', "Message" ,wx.OK | wx.ICON_INFORMATION)
				return
		
		#获取fps
		if fps_string=='':
			fps = 10
			self.m_textCtrl6.SetValue(str(fps))
		else:
			try:
				fps = float(fps_string)
				if fps<5:
					fps = 5
					self.m_textCtrl6.SetValue(str(fps))
				if fps>20:
					fps = 20
					self.m_textCtrl6.SetValue(str(fps))
			except (ValueError,TypeError) as e:
				wx.MessageBox(u' fps输入须为数值', "Message" ,wx.OK | wx.ICON_INFORMATION)
				return
		self.input_state = 1
		return [area_start,area_stop,area_offset,fps]
		
	def start_detection(self,event):
		"""
		This function is set to start vital_signs detection
		"""
		input_value = self.get_Input()
		device = 'COM8'
		baseband=True
		buffer_size = 200
		# 如果成功取到值
		if self.input_state:
			self.t1=StartDetectionThread(self,input_value[3],900,1150,[input_value[0],input_value[1]],input_value[2],device,baseband,buffer_size)
			self.start_threads.append(self.t1)
			self.t1.start()

	def stop_detection(self,event):
		while self.start_threads:
			thread=self.start_threads[0]
			thread.stop()
			self.start_threads.remove(thread)

class MyFrame1 ( wx.Frame ):
	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"基于雷达信号的体征检测", pos = wx.DefaultPosition, size = wx.Size( 1026,730 ), style = wx.MINIMIZE_BOX|wx.RESIZE_BORDER|wx.SYSTEM_MENU|wx.CAPTION|wx.CLOSE_BOX|wx.CLIP_CHILDREN|wx.TAB_TRAVERSAL )

		#self.SetBackgroundColour("WHITE")
		#self.Bind(wx.EVT_PAINT, self.OnPaint)
		self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )
		self.SetFont( wx.Font( 9, 70, 90, 92, False, "宋体" ) )
		
		#self.m_statusBar1 = self.CreateStatusBar( 1, wx.ST_SIZEGRIP, wx.ID_ANY )
		self.m_menubar1 = wx.MenuBar( 0 )
		self.m_menu1 = wx.Menu()
		Detection_main=self.m_menu1.Append(-1, "module1")
		Detection_selfDefined=self.m_menu1.Append(-1,'module2')
		self.m_menubar1.Append( self.m_menu1, u"功能选择" )
		
		self.m_menu3 = wx.Menu()
		config=self.m_menu3.Append(-1,u"默认参数设置")
		self.m_menubar1.Append( self.m_menu3, u"参数配置" )
		
		self.m_menu4 = wx.Menu()
		Connect=self.m_menu4.Append( -1, u"连接仪器" )
		disConnect=self.m_menu4.Append( -1, u"断开仪器" )
		self.m_menubar1.Append( self.m_menu4, u"仪器连接" )  

		self.m_menu5 = wx.Menu()
		self.m_menubar1.Append( self.m_menu5, u"帮助" ) 
		
		self.SetMenuBar( self.m_menubar1 )
		
		##############################################设置主界面
		self.mainSizer=wx.BoxSizer( wx.VERTICAL )
		
		self.buttom_color="SEA GREEN"
		
		
		#添加主界面
		self.bSizer3 = wx.BoxSizer( wx.VERTICAL )
		self.panelOne = MyPanel1(self)
		self.bSizer3.Add( self.panelOne, 1, wx.EXPAND , 5 )
		self.mainSizer.Add( self.bSizer3, 40, wx.EXPAND, 0 )
		
		# self.m_staticline11 = wx.StaticLine( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
		# self.mainSizer.Add( self.m_staticline11, 0, wx.EXPAND |wx.ALL, 0 )
		
		self.stateSizer1=wx.BoxSizer( wx.VERTICAL)
		self.m_panel13 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		self.m_panel13.SetBackgroundColour(self.buttom_color)
		self.stateSizer=wx.BoxSizer(wx.HORIZONTAL)
		
		
		self.m_staticText61 = wx.StaticText( self.m_panel13, wx.ID_ANY, u" XXXXXXXXXXXXX", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText61.Wrap( -1 )
		self.stateSizer.Add( self.m_staticText61, 3, wx.ALL, 5 )
		
		self.stateSizer.AddStretchSpacer(5) 
		
		self.m_staticText62 = wx.StaticText( self.m_panel13, wx.ID_ANY, u"测试仪器信息：", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText62.Wrap( -1 )
		self.stateSizer.Add( self.m_staticText62, 1, wx.ALL|wx.ALIGN_CENTER, 5 )
		self.m_staticText64 = wx.StaticText( self.m_panel13, wx.ID_ANY, u"X4M03 module", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText64.Wrap( -1 )
		self.stateSizer.Add( self.m_staticText64, 1, wx.ALL|wx.ALIGN_CENTER, 5 )
		
		self.m_panel13.SetSizer(self.stateSizer)
		self.m_panel13.Layout()
		self.stateSizer.Fit( self.m_panel13 )
		self.stateSizer1.Add( self.m_panel13, 1, wx.EXPAND, 5 )
		
		self.mainSizer.Add( self.stateSizer1, 1, wx.EXPAND, 5 )
		
		self.SetSizer( self.mainSizer )
		self.Layout()
		
		self.Centre( wx.BOTH )
		
		#关闭
		self.Bind(wx.EVT_CLOSE, self.OnClose)
		self.threads3=[]
		
		#设置定时器
		self.timer = wx.Timer(self)#创建定时器
		self.Bind(wx.EVT_TIMER, self.onTimer,self.timer)  #设置定时事件
		self.timer.Start(1000)

	def OnClose(self,event):
		self.Destroy()
	
	def onTimer(self, evt):
		pass;

def main():
	app = wx.App()
	win = MyFrame1(None)
	win.Show()
	app.MainLoop()

if __name__ == "__main__":
	main()