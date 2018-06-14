import sys
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pymoduleconnector import ModuleConnector

__version__ = 3

def reset(device_name):
    from time import sleep
    mc = ModuleConnector(device_name)
    r = mc.get_xep()
    r.module_reset()
    mc.close()
    sleep(3)

def simple_xep_plot(device_name, bb = False):
    
    FPS = 10
    
    reset(device_name)
    mc = ModuleConnector(device_name)

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
    r.x4driver_set_dac_min(900)
    r.x4driver_set_dac_max(1150)

    # Set integration
    r.x4driver_set_iterations(16)
    r.x4driver_set_pulses_per_step(26)
    
    r.x4driver_set_frame_area(0.2,9.8)
    r.x4driver_set_frame_area_offset(0.2)
    

    if bb:
        r.x4driver_set_downconversion(1)
    else:
        r.x4driver_set_downconversion(0)
        
    # Start streaming of data
    r.x4driver_set_fps(FPS)

    def clear_buffer():
        """Clears the frame buffer"""
        while r.peek_message_data_float():
            _=r.read_message_data_float()

    def read_frame():
        """Gets frame data from module"""
        d = r.read_message_data_float()
        frame = np.array(d.data)
        print (frame.shape)
        print (len(frame))
         # Convert the resulting frame to a complex array if downconversion is enabled
        if bb:
            n=len(frame)
            frame = frame[:n//2] + 1j*frame[n//2:]
            

        return frame
    
    def animate(i):
        if bb:
            line.set_ydata(abs(read_frame()))  # update the data
        else:
            line.set_ydata(read_frame())  # update the data
        return line,
        
    fig = plt.figure()
    fig.suptitle("simple_xep_plot version %d. Baseband = %r"%(__version__, bb))
    ax = fig.add_subplot(1,1,1)
    frame = read_frame()

    if bb:
        frame = abs(frame)

    line, = ax.plot(frame)
    
    clear_buffer()
    
    ani = FuncAnimation(fig, animate, interval=FPS)
    plt.show()
    
    # Stop streaming of data
    r.x4driver_set_fps(0)



def main():
    parser = OptionParser()
    parser.add_option(
        "-d",
        "--device",
        dest="device_name",
        help="device file to use",
        metavar="FILE")
    parser.add_option(
        "-b",
        "--baseband",
        action="store_true",
        default=False,
        dest="baseband",
        help="Enable baseband, rf data is default",
        metavar="FILE")


    (options, args) = parser.parse_args()
    print (options.device_name,options.baseband)

    if not options.device_name:
        print ("you have to specify device, e.g.: python record.py -d /dev/ttyACM0")
        sys.exit(1)

    simple_xep_plot(options.device_name, bb = options.baseband)

if __name__ == "__main__":
    main()


