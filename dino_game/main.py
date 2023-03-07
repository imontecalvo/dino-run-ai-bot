import pygame
import os
import random
import numpy as np
from joblib import Parallel, delayed
import joblib
import datetime

import sys
sys.path.append('../')
from genome import Genome

pygame.init()

# Global Constants
SCREEN_HEIGHT = 320
SCREEN_WIDTH = 1290
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Textures/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Textures/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Textures/Dino", "DinoStart.png"))
DUCKING = [pygame.image.load(os.path.join("Textures/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Textures/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Textures/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join(
                    "Textures/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Textures/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Textures/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join(
                    "Textures/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Textures/Cactus", "LargeCactus3.png"))]

BIRD = [pygame.image.load(os.path.join("Textures/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Textures/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Textures/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join("Textures/Other", "Track.png"))

GENOME_ARCH = [6,10,10,3]

def invert_y_axis(value):
    return abs(value-SCREEN_HEIGHT)

class Dinosaur:
    X_POS = 42
    Y_POS = SCREEN_HEIGHT-113
    Y_POS_DUCK = SCREEN_HEIGHT-75
    JUMP_VEL = 8.5

    # best_players: lista de mejores genomas, los cuales se usaran para generar uno nuevo con combinacion y mutacion
    # trained_genome: genoma entrenado, se lo pasamos en caso de que querramos usar este genoma y no generar uno nuevo
    def __init__(self, best_players, trained_genome=False):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.dead = False

        if trained_genome:
            self.genome = best_players[0]
        else:
            self.genome = Genome(GENOME_ARCH, best_players)

    def update(self, action):
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

        if action==0 and not self.dino_jump and self.dino_rect.y == self.Y_POS:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif action==1:
            self.jump_vel = self.JUMP_VEL
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif  not (self.dino_jump or action==1):
            self.jump_vel = self.JUMP_VEL
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel < - self.JUMP_VEL:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))
        # pygame.draw.rect(SCREEN, (255,0,0), self.dino_rect,2)

    # Data es una lista con:
    #   - coord Y de dinosaurio
    #   - distancia al proximo obstaculo
    #   - coord Y del obstaculo
    #   - ancho del obstaculo
    #   - alto del obstaculo
    #   - hay obstaculo o no
    def make_decision(self, data):
        data = data[np.newaxis,:]
        _, action = self.genome.evaluate(data)
        return action
    

class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1290)
        self.y = random.randint(30, 50)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)
        # pygame.draw.rect(SCREEN, (255,0,0), self.rect, 2)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 230


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 200


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)
        self.type = random.randint(0,2)
        if self.type == 0:
            self.rect.y = 223
        elif self.type == 1:
            self.rect.y = 168
        elif self.type == 2:
            self.rect.y = 114
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index//5], self.rect)
        self.index += 1



def main(players, n_generation):
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True
    clock = pygame.time.Clock()

    cloud = Cloud()
    game_speed = 20
    x_pos_bg = 0
    y_pos_bg = 282
    points = 0
    font = pygame.font.Font('freesansbold.ttf', 20)
    obstacles = []

    best_players = []
    best_scores = []

    def score():
        global points, game_speed
        points += 1
        if points % 100 == 0:
            game_speed += 1

        text = font.render("Points: " + str(points), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)

        alive = font.render("Alive: " + str(len(players)), True, (0, 0, 0))
        aliveRect = text.get_rect()
        aliveRect.center = (1000, 65)

        generation = font.render("Generation: " + str(n_generation), True, (0, 0, 0))
        generationRect = text.get_rect()
        generationRect.center = (1000, 90)

        SCREEN.blit(text, textRect)
        SCREEN.blit(alive, aliveRect)
        SCREEN.blit(generation, generationRect)

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    def draw_players():
        for player in players:
            player.draw(SCREEN)

    def check_collision(obstacle):
        for player in players:
            if player.dino_rect.colliderect(obstacle.rect):
                player.dead = True
                if len(players) <= 10:
                    best_players.insert(0,player)
                    best_scores.insert(0,points)
                players.remove(player)
                del player

    def update_players():
        for player in players:
            data = [
                invert_y_axis(player.dino_rect.top),
                obstacles[0].rect.left if len(obstacles)>0 else 0,
                invert_y_axis(obstacles[0].rect.top) if len(obstacles)>0 else 0,
                abs(obstacles[0].rect.left-obstacles[0].rect.right) if len(obstacles)>0 else 0,
                abs(obstacles[0].rect.bottom-obstacles[0].rect.top) if len(obstacles)>0 else 0,
                1 if len(obstacles)>0 else 0
            ]
            data = normalize_data(data)
            action = player.make_decision(np.array(data))
            player.update(action)

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        SCREEN.fill((255, 255, 255))

        draw_players()

        if len(obstacles) == 0:
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 2) == 2:
                obstacles.append(Bird(BIRD))
        
        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            check_collision(obstacle)
            obstacle.update()


        cloud.draw(SCREEN)
        cloud.update()

        background()
        score()

        clock.tick(30)
        pygame.display.update()
        update_players()
        if(len(players) == 0):
            pygame.time.delay(500)
            return best_players, best_scores, points


def create_individuals(n_players, best_genomes, use_fixed_genome=False):
    individuals = []
    for i in range(n_players):
        ind = Dinosaur(best_genomes, use_fixed_genome)
        individuals.append(ind)
    return individuals

# Cargar mejores genomas
def load_genomes(genome_file):
    return joblib.load(genome_file)

def load_scores(scores_file):
    return joblib.load(scores_file)


# Funcion menu
# use_fixed_genome: Si queremos usar un genoma fijo en lugar de crear nuevos combinando los mejores guardados
def menu(n_players,use_pretrained_model, use_fixed_genome=False):
    #f = open("scores3.txt", "a")
    global points
    points=0
    run = True

    load_models = use_pretrained_model or use_fixed_genome
    best_genomes = load_genomes("../trained_models/best_genomes_wo_speed2.pkl") if load_models else []
    best_scores = load_scores("../trained_models/best_scores_wo_speed2.pkl") if load_models else [0]*10

    generation = 1
    while run:
        print(f"Start gen: {generation} -> {datetime.datetime.now()}")
        SCREEN.fill((255, 255, 255))
        font = pygame.font.Font('freesansbold.ttf', 30)

        # Pantalla inicio
        text = font.render("Press any Key to Start", True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        text_score = font.render("Your Score: " + str(points), True, (0, 0, 0))
        text_scoreRect = text.get_rect()
        text_scoreRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2+45)
        SCREEN.blit(text, textRect)
        SCREEN.blit(text_score, text_scoreRect)
        SCREEN.blit(RUNNING[0], (SCREEN_WIDTH // 2 - 20, SCREEN_HEIGHT // 2 - 140))
        pygame.display.update()

        # Creacion individuos/agentes
        players = create_individuals(n_players, best_genomes,use_fixed_genome)

        # Proxima generacion
        best_players, player_scores, points = main(players, generation)
        #f.write(f"{generation},{','.join(map(str, player_scores))}\n")
        #f.flush()
        

        # Actualizacion de mejores puntajes y mejores genomas
        for i in range(len(player_scores)):
            for j in range(len(best_scores)):
                if player_scores[i] > best_scores[j]:
                    best_scores.insert(j, player_scores[i])
                    best_genomes.insert(j, best_players[i].genome)
                    break
        best_scores = best_scores[:10]
        best_genomes = best_genomes[:10]

        print(f"\nSCORE: {points} - BEST SCORES:{best_scores} - G: {generation}")
        del best_players
        pygame.time.delay(500)

        # Se guarda lista de mejores genomas
        #joblib.dump(best_genomes, '../trained_models/best_genomes_wo_speed2.pkl')
        #joblib.dump(best_scores, '../trained_models/best_scores_wo_speed2.pkl')

        generation += 1
        # Cerrar juego
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False                

def normalize_data(data):
    return [
        data[0]/SCREEN_HEIGHT*2-1,
        data[1]/SCREEN_WIDTH*2-1,
        data[2]/SCREEN_HEIGHT*2-1,
        data[3]/SCREEN_WIDTH*2-1,
        data[4]/SCREEN_HEIGHT*2-1,
        data[5]
    ]
    

menu(n_players=1000, use_pretrained_model=True, use_fixed_genome=False)