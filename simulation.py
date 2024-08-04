import random
import time
import threading
import pygame
import sys
import google.generativeai as genai
from ultralytics import YOLOv10
import torch
import cv2
# import supervision as sv
# import matplotlib.pyplot as plt
import numpy as np

# Configure the API key
genai.configure(api_key="") # Provide you Gemini APi Key

# Initialize the Gemini client
model = genai.GenerativeModel('gemini-1.0-pro-latest')

# Default values of signal timers
defaultGreen = {0:10, 1:10, 2:10, 3:10}
defaultYellow = 2

signals = []
noOfSignals = 4
currentGreen = 0   
nextGreen = 1
currentYellow = 0    

speeds = {'car':2.25, 'bus':1.8, 'truck':1.8, 'bike':2.5}


x = {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]}    
y = {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]}

vehicles = {'right': {0:[], 1:[], 2:[], 'crossed':0}, 'down': {0:[], 1:[], 2:[], 'crossed':0}, 'left': {0:[], 1:[], 2:[], 'crossed':0}, 'up': {0:[], 1:[], 2:[], 'crossed':0}}
vehicleTypes = {0:'car', 1:'bus', 2:'truck', 3:'bike'}
directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}

signalCoods = [(530,230),(810,230),(810,570),(530,570)]
signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]

stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}

stoppingGap = 15 
movingGap = 15

# Ensure only specific parts use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assuming yolo is a PyTorch model
yolo = YOLOv10("T:\\Sus\\TrafficLight\\model\\yolov10x_tuned.pt").to(device)

print(next(yolo.parameters()).device)
temp = cv2.imread('T:\\Sus\\TrafficLight\\output_images\\13.png')
yolo(temp)[0]

video_writer = None
record_video = True
output_video_path = 'simulation_output.mp4'
running = True

pygame.init()
simulation = pygame.sprite.Group()

class TrafficSignal:
    def __init__(self, yellow, green):
        self.yellow = yellow
        self.green = green
        self.signalText = ""

class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction):
        pygame.sprite.Sprite.__init__(self)
        self.done = False
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.image = pygame.image.load(path)
        

        if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):    # if more than 1 vehicle in the lane of vehicle before it has crossed stop line
            if(direction=='right'):
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].image.get_rect().width - stoppingGap         # setting stop coordinate as: stop coordinate of next vehicle - width of next vehicle - gap
            elif(direction=='left'):
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].image.get_rect().width + stoppingGap
            elif(direction=='down'):
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].image.get_rect().height - stoppingGap
            elif(direction=='up'):
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].image.get_rect().height + stoppingGap
        else:
            self.stop = defaultStop[direction]
            
        # Set new starting and stopping coordinate
        if(direction=='right'):
            temp = self.image.get_rect().width + stoppingGap    
            x[direction][lane] -= temp
        elif(direction=='left'):
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] += temp
        elif(direction=='down'):
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] -= temp
        elif(direction=='up'):
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        if(self.direction=='down'):
            if(self.crossed==0 and self.y+self.image.get_rect().height>stopLines[self.direction]):
                self.crossed = 1
            if((self.y+self.image.get_rect().height<=self.stop or self.crossed == 1 or (currentGreen==1 and currentYellow==0)) and (self.index==0 or self.y+self.image.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - movingGap))):                
                # (if the image has not reached its stop coordinate or has crossed stop line or has green signal) and (it is either the first vehicle in that lane or it is has enough gap to the next vehicle in that lane)
                self.y += self.speed
        elif(self.direction=='up'):
            if(self.crossed==0 and self.y<stopLines[self.direction]):
                self.crossed = 1
            if((self.y>=self.stop or self.crossed == 1 or (currentGreen==3 and currentYellow==0)) and (self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].image.get_rect().height + movingGap))):                
                self.y -= self.speed
        elif(self.direction=='left'):
            if(self.crossed==0 and self.x<stopLines[self.direction]):
                self.crossed = 1
            if((self.x>=self.stop or self.crossed == 1 or (currentGreen==2 and currentYellow==0)) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].image.get_rect().width + movingGap))):                
                self.x -= self.speed   
        elif(self.direction=='right'):
            if(self.crossed==0 and self.x+self.image.get_rect().width>stopLines[self.direction]):   # if the image has crossed stop line now
                self.crossed = 1
            if((self.x+self.image.get_rect().width<=self.stop or self.crossed == 1 or (currentGreen==0 and currentYellow==0)) and (self.index==0 or self.x+self.image.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - movingGap))):                
                self.x += self.speed  # move the vehicle

        # Remove the vehicle if it has crossed the boundary
        if (self.x > 790 and self.y < 430) or (self.x < 595 and self.y > 430) or (self.y > 530 and self.x > 687) or (self.y < 340 and self.x < 687):
            if self.done == False:
                self.remove_from_lane()
            self.done = True

    def remove_from_lane(self):
        if self in vehicles[self.direction][self.lane]:
            vehicles[self.direction]['crossed'] += 1

# Initialization of signals with default values
def initialize():
    ts1 = TrafficSignal(defaultYellow, defaultGreen[0])
    signals.append(ts1)
    ts2 = TrafficSignal(defaultYellow, defaultGreen[1])
    signals.append(ts2)
    ts3 = TrafficSignal(defaultYellow, defaultGreen[2])
    signals.append(ts3)
    ts4 = TrafficSignal(defaultYellow, defaultGreen[3])
    signals.append(ts4)
    
    repeat()

def repeat():
    global currentGreen, currentYellow, nextGreen
    while(signals[currentGreen].green>0):   # while the timer of current green signal is not zero
        updateValues()
        time.sleep(1)
    currentYellow = 1   # set yellow signal on
    
    # reset stop coordinates of lanes and vehicles 
    for i in range(0,3):
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]

    while(signals[currentGreen].yellow>0):  # while the timer of current yellow signal is not zero
        updateValues()
        time.sleep(1)
    time.sleep(1)
    currentYellow = 0   # set yellow signal off

    currentGreen = nextGreen
    repeat()  

# Update values of the signal timers after every second
def updateValues():
    for i in range(0, noOfSignals):
        if(i==currentGreen):
            if(currentYellow==0):
                signals[i].green-=1
            else:
                signals[i].yellow-=1

# Generating vehicles in the simulation
def generateVehicles():
    global img_counter
    while(True):
           
        vehicle_type = random.randint(0,3)
        lane_number = random.randint(1,2)
        temp = random.randint(0,99)
        direction_number = 0
        dist = [25,50,75,100]
        
        if(temp<dist[0]):
            direction_number = 0
        
        elif(temp<dist[1]):
            direction_number = 1
        
        elif(temp<dist[2]):
            direction_number = 2
        
        elif(temp<dist[3]):
            direction_number = 3
        
        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number])
        
        time.sleep(1)

def get_car_count(img):
    global yolo, device

    # Perform a single inference on the entire image
    results = yolo(img)[0]

    # Get the image dimensions
    height, width = img.shape[:2]

    # Define the regions of interest (ROIs) for each direction
    roi_left = [(0, 352), (590, 435)]
    roi_right = [(int(0.5643*width), int(0.53125*height)), (width, int(0.6375*height))]
    roi_up = [(int(0.4857*width), 0), (int(0.55*width), int(0.43125*height))]
    roi_down = [(int(0.4286*width), int(0.64375*height)), (int(0.4929*width), height)]

    # Function to check if a box is within a ROI
    def is_in_roi(box, roi):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        roi_x1, roi_y1 = roi[0]
        roi_x2, roi_y2 = roi[1]
        return (x1 >= roi_x1 and y1 >= roi_y1 and x2 <= roi_x2 and y2 <= roi_y2)

    # Count vehicles in each direction
    left_count = sum(1 for box in results.boxes if is_in_roi(box, roi_left))
    right_count = sum(1 for box in results.boxes if is_in_roi(box, roi_right))
    up_count = sum(1 for box in results.boxes if is_in_roi(box, roi_up))
    down_count = sum(1 for box in results.boxes if is_in_roi(box, roi_down))

    return left_count, right_count, up_count, down_count


def get_state():
    screen_capture = pygame.surfarray.array3d(pygame.display.get_surface())
    screen_capture = screen_capture.transpose([1, 0, 2])  # Convert from (width, height, channel) to (height, width, channel)
    screen_capture = cv2.cvtColor(screen_capture, cv2.COLOR_RGB2BGR)
    
    left, right, up, down = get_car_count(screen_capture)
    
    state = {
        "currentGreen": currentGreen,
        "vehicles": {0: left, 1: up, 2: right, 3: down}
    }
    
    return state

def get_decision_from_ai():
    global currentGreen, currentYellow, nextGreen
    while True:    
        if signals[currentGreen].green == 2:
            print("hi")
            state = get_state()
            print(state)
            
            prompt = (
                f"Current green light is on road number: {state['currentGreen']}\n"
            )
            prompt += f"Vehicle Counts on Roads:\n"
            for i, count in enumerate(state['vehicles'].values()):
                prompt += f"  Road {i}: {count} vehicles\n"

            prompt += """Analyze the following data and decide what signal has to go green next based on the following factors: 
            1- Fist main factor is to see the count of car the one with the most cars road has to go green.
            2- Determine an appropriate green light duration based on the number of waiting vehicles (approximately 1 second per car).
            3- If there is ambulance. Priotize that roads.
                        """

            prompt += """
            Provide the next signal state in the format evrytime dont give any other output: 
            nextGreen: <road number>, defaultGreen: <green time>
            """
            
            try:
                analysis_results = model.generate_content(prompt)
                
                decision_text = analysis_results.candidates[0].content.parts[0].text.strip()
                print(decision_text)

                decision = {}
                for part in decision_text.split(','):
                    key, value = part.split(':')
                    decision[key.strip()] = int(value.strip())
                
                nextGreen = decision['nextGreen'] # set next signal as green signal
                defaultGreen[nextGreen] = decision['defaultGreen']
                
                gTime = defaultGreen[nextGreen]
                
                signals[nextGreen].green = gTime if gTime > 3 else 4
                signals[nextGreen].yellow = defaultYellow

            except Exception as e:
                print(f"Error in AI decision making: {e}")
                gTime = max(((len(vehicles['right'][0]) + len(vehicles['right'][1]) + len(vehicles['right'][2]))-vehicles['right']['crossed']), ((len(vehicles['left'][0]) + len(vehicles['left'][1]) + len(vehicles['left'][2]))-vehicles['left']['crossed']), ((len(vehicles['up'][0]) + len(vehicles['up'][1]) + len(vehicles['up'][2]))-vehicles['up']['crossed']), ((len(vehicles['down'][0]) + len(vehicles['down'][1]) + len(vehicles['down'][2]))-vehicles['down']['crossed']))
                if gTime > 10:
                    gTime -= 5
                elif gTime > 6:
                    gTime -= 3
                elif gTime > 4:
                    gTime -= 1
                elif gTime < 2:
                    gTime += 2
                    
                signals[nextGreen].green = gTime
                signals[nextGreen].yellow = defaultYellow
                
        time.sleep(1)

class Main:
    thread1 = threading.Thread(name="initialization",target=initialize, args=())    # initialization
    thread1.daemon = True
    thread1.start()

    # Colours 
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Screensize 
    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    # Setting background image i.e. image of intersection
    background = pygame.image.load('images/intersection.png')

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("SIMULATION")

    # Loading signal images and font
    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)

    thread2 = threading.Thread(name="generateVehicles",target=generateVehicles, args=())    # Generating vehicles
    thread2.daemon = True
    thread2.start()
    
    thread3 = threading.Thread(name="get_decision_from_ai", target=get_decision_from_ai, args=())
    thread3.daemon = True
    thread3.start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.blit(background,(0,0))   # display background in simulation
        for i in range(0,noOfSignals):  # display signal and set timer according to current status: green, yello, or red
            if(i==currentGreen):
                if(currentYellow==1):
                    signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])
        signalTexts = ["","","",""]

        # display signal timer
        for i in range(0,noOfSignals):  
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i],signalTimerCoods[i])

        # display the vehicles
        for vehicle in simulation:  
            screen.blit(vehicle.image, [vehicle.x, vehicle.y])
            vehicle.move()
        pygame.display.update()
