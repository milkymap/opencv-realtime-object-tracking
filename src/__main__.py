import cv2 
import numpy as np 
import operator as op 

import click 
from typing import Tuple, List, Dict 
from typing import Optional, Any 

from beepy import beep 

import threading

from numpy.typing import NDArray
from .log import logger 

# ffmpeg -i input.mkv  -c:v libx264 -pix_fmt yuv420p  -profile:v baseline -level 3.0  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vb 1024k  -acodec aac -ar 44100 -ac 2 -minrate 1024k -maxrate 1024k -bufsize 1024k  -movflags +faststart output.mp4

def make_sound():
    beep(sound='ping')

@click.group(chain=False, invoke_without_command=True)
@click.pass_context
def handler(ctx:click.core.Context):
    ctx.ensure_object(dict)


@handler.command()
@click.option('--path2video', type=click.Path(exists=True, dir_okay=False))
@click.option('--tracker_name', type=click.Choice(['CSRT', 'KCF'], case_sensitive=False), default='KCF')
@click.option('--winname', type=str, default='main-screen')
@click.option('--winsize', type=click.Tuple([int, int]), default=(720, 720))
@click.option('--threshold', type=float, default=30)
@click.option('--memory', type=int, default=7)
@click.pass_context
def process_video(ctx:click.core.Context, path2video:str, tracker_name:str, winname:str, winsize:Tuple[int, int], threshold:float, memory:int):
    tracker_function_name = f'Tracker{tracker_name}_create'
    tracker:cv2.Tracker = op.attrgetter(tracker_function_name)(cv2)()

    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, *winsize)

    source:Optional[NDArray] = None 
    bounding_box:Optional[Tuple[int, int, int, int]] = None 
    accumulator:List[NDArray] = []
    MESSAGE = 'ABNORMAL EVENT'
    FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 2
    THICKNESS = 5
    threads_tracker:List[threading.Thread] = []
    (tw, th), tb = cv2.getTextSize(MESSAGE, FONT_FACE, FONT_SCALE, THICKNESS)
    capture = cv2.VideoCapture(path2video)
    while True:
        try:
            threads_tracker = [thread_ for thread_ in threads_tracker if thread_.is_alive()]
            accumulator = accumulator[-memory:]
            capture_val, bgr_image = capture.read()
            if not capture_val:
                break 
            
            key_code = cv2.waitKey(25) & 0xFF 
            if key_code == 27:
                break 

            bgr_image = cv2.resize(bgr_image, winsize)

            if key_code == 32:
                bounding_box = cv2.selectROI(winname, bgr_image, fromCenter=False, showCrosshair=True)   
                x, y, w, h = list(map(int, bounding_box))
                source = np.array([x + int(w / 2), y + int(h / 2)]) 
                tracker.init(bgr_image, bounding_box)
                accumulator = [np.asarray(bounding_box)]
            
            if bounding_box is None:
                cv2.imshow(winname, bgr_image)
                continue

            update_val, updated_coordinates = tracker.update(bgr_image)
            if not update_val:
                logger.warning('reduce tracker noise')
                updated_coordinates = np.mean(accumulator, axis=0).tolist()

            x, y, w, h = list(map(int, updated_coordinates))
            target = np.array([x + int(w / 2), y + int(h / 2)]) 
            
            cv2.line(bgr_image, (source), (target), (0, 255, 0), 3)
            cv2.circle(bgr_image, source, 7, (0, 0, 0), -1)
            cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
            
            accumulator.append(np.asarray(updated_coordinates))
            distance = np.sqrt(np.sum((target - source) ** 2) + 1e-8)
            if distance > threshold:
                logger.error(f"abnormal activity => distance to source : {distance:07.3f}")  # better algorithm
                H, W, _ = bgr_image.shape 
                cv2.putText(bgr_image, MESSAGE, (int(W / 2 - tw / 2), int(H / 2 + th / 2) - tb), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), THICKNESS)
                if len(threads_tracker) < 10:
                    thread_ = threading.Thread(target=make_sound, args=[])
                    thread_.daemon = True 
                    thread_.start()
                    threads_tracker.append(thread_)

            cv2.imshow(winname, bgr_image)
        except KeyboardInterrupt:
            break 
        except Exception as e:
            logger.error(e)
            break 
    
    del tracker 
    capture.release()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    handler()