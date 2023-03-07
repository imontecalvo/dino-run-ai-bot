from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException
import time
import pyautogui
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mss import mss
from PIL import ImageGrab, Image

from joblib import Parallel, delayed
import joblib

options =  webdriver.ChromeOptions()
options.add_argument('--start-maximized')
options.add_experimental_option("useAutomationExtension", False)
options.add_experimental_option("excludeSwitches",["enable-automation"])

s=Service('chromedriver_linux64/chromedriver.exe')
driver = webdriver.Chrome(service=s,options=options)
url='chrome://dino'

# Deteccion de elementos
SCREEN_H, SCREEN_W = 320, 1290

OFFSET_X = 165
OFFSET_Y = 40

def invert_y_axis(value):
    return abs(value-SCREEN_H)

def detect_obstacles2(img):    
    # Ponemos en 255 todos los pixeles oscuros y en 0 todo lo que no nos sirve
    cond = img < 100
    img[cond] = 255
    img[~cond] = 0
    
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    # Iterar a través de los contornos encontrados
    for contour in contours:
        # Ignorar contornos demasiado pequeños
        if cv2.contourArea(contour) < 100:
            continue
        # Obtener las coordenadas (x, y) y el ancho y alto del rectángulo que encierra el contorno
        (x, y, w, h) = cv2.boundingRect(contour)
        if h < 30:
            continue
        rects.append({'d': OFFSET_X+x, 'y': invert_y_axis(OFFSET_Y+y), 'w': w, 'h': h})
    
    if len(rects):
        dist = [rect['d'] for rect in rects]
        return rects[dist.index(min(dist))]
    return False

def detect_dino2(img):
    dino_types = ['DinoStart','DinoDuck1']
    
    for dino_type in dino_types:
        template = cv2.imread(f"pics/elements/{dino_type}.png",0)
        w, h = template.shape[::-1]
        method = eval('cv2.TM_CCOEFF_NORMED')
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.7:
            top_left = (max_loc[0],max_loc[1])
            bottom_right = (top_left[0] + w, top_left[1] + h)
            
            return {"x":top_left[0]+w,
                "y":invert_y_axis(top_left[1]),
                "h":h,
                "top_left":top_left, 
                "bottom_right":bottom_right}
    return False

def detect_elements(img):
    dino_data = detect_dino2(img.copy()[:,:OFFSET_X])
    if dino_data:
        obstacle_data = detect_obstacles2(img.copy()[OFFSET_Y:,OFFSET_X:])
        return dino_data, obstacle_data
    print("Cannot detect dino")
    plt.imshow(img)
    plt.show()
    plt.imshow(img[:,:OFFSET_X])
    plt.show()
    return False

# Bot juego
def open_game(driver, url):
    try:
        driver.get(url)
        return True
    except WebDriverException as e:
        pass
    
def load_genome():
    best_genomes = joblib.load("trained_models/best_genomes_wo_speed1.pkl")
    return best_genomes[0]
    
def is_done(driver):
    return driver.execute_script('return Runner.instance_.crashed')
    
def normalize_data(data):
    return [
        data[0]/SCREEN_H*2-1,
        data[1]/SCREEN_W*2-1,
        data[2]/SCREEN_H*2-1,
        data[3]/SCREEN_W*2-1,
        data[4]/SCREEN_H*2-1,
        data[5]
    ]
    

def get_data(driver):
    # Data es una lista con:
    #   - coord Y de dinosaurio
    #   - distancia al proximo obstaculo
    #   - coord Y del obstaculo
    #   - ancho del obstaculo
    #   - alto del obstaculo
    #   - hay obstaculo o no
    image = mss().grab({'top': 212, 'left': 60, 'width': 1290, 'height': 320})
    image = np.array(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    dark_mode = driver.execute_script('return Runner.instance_.isDarkMode')
    if dark_mode:
        img = cv2.bitwise_not(img)
    
    dino_data, obstacle_data = detect_elements(img)
    data = [
        dino_data["y"],
        obstacle_data["d"] if obstacle_data else 0,
        obstacle_data["y"] if obstacle_data else 0,
        obstacle_data["w"] if obstacle_data else 0,
        obstacle_data["h"] if obstacle_data else 0,
        1 if obstacle_data else 0
    ]
    return normalize_data(data)
    
def make_decision(decision):
    if decision == 0:
        pyautogui.keyUp("down")
        pyautogui.press("space")
    elif decision == 1:
        pyautogui.keyDown("down")
    else:
        pyautogui.keyUp("down")
        pass


def main():
    open_game(driver, url)
    time.sleep(1)
    done = is_done(driver)
    player = load_genome()
    pyautogui.press("space")
    time.sleep(3)
        
    while not done:
        data = (get_data(driver))
        data = np.array(data)[np.newaxis,:]
        _,decision = player.evaluate(data)
        make_decision(decision)
        done = is_done(driver)

main()