import os
from pathlib import Path


class PrintLog:
    def __init__(self, temp_file_path, model, threshold, input_stream_w, input_stream_h, output_stream_w, 
                                                                                            output_stream_h) -> None:
        self.temp_file_path = temp_file_path
        self.model = model
        self.threshold = threshold
        self.input_stream_w = input_stream_w    
        self.input_stream_h = input_stream_h
        self.output_stream_w = output_stream_w
        self.output_stream_h = output_stream_h


    def print_log(self, results, elapsed_time=0):
        # ----->> Log prints <<-----
        os.system('clear')
        print(
            '___ [ RPI Object detection with TFLite ] _________________________')
        print()
        print("Used model:\t", Path(self.model).name)
        print()

        if elapsed_time:
            print('Process FPS:\t', '{0:.2f}'.format(1/(elapsed_time/1000)))
            with open(self.temp_file_path) as fp:
                line = fp.readline()
                line = int(line) / 1000

                print("CPU Temp:\t", line, "°C")
        else:
            print("Can't inform process fps.")
        
        print()
        print("Input size:")
        print("\twidth:\t", self.input_stream_w)
        print("\theight:\t", self.input_stream_h)
        print()
        print("Output size:")
        print("\twidth:\t", self.output_stream_w)
        print("\theight:\t", self.output_stream_h)
        print()
        if len(results):
            obj_count = len(results)
        else:
            obj_count = 0
        print("Detected objects:", obj_count)
        print("Threshold:\t", self.threshold)
        print()