
import numpy as np

from Tkinter import *

import time

import threading

import os
import neat
import pickle

import sys

SEED = 5            # critical
NUM_FOOD = 25       # critical
FOOD_SCORE = 10    # critical for training

RANGE = 3           # critical
GAME_SIZE = 50      # critical
DISPLAY_SIZE = 500
STEP_SPEED = 0.1

BLANK_REP = 0
SNAKE_REP = 1
WALL_REP = 2
FOOD_REP = 3

SNAKE_INIT = [(5, 5), (4, 5),(3, 5),(2,5)]  # critical
SNAKE2_INIT = [(5, 5), (4, 5),(3, 5),(2,5)]  # critical

SIM_STEPS = GAME_SIZE*GAME_SIZE             # critical
MAX_GENERATIONS = 25
DISPLAY_ALL = FALSE



class Python(object):
    """ A snake game object!

    Attributes:
    _snake - list of snake x,y tuples
    _step - the current time step
    _alive - the current state of the snake
    width - width of game
    height - height of game

    Methods:
    full_state - returns a vector of the entire board state
    0 - blank
    1 - snake
    2 - walls
    """
    def __init__(self, width, height, snake, snake2):
        self.width = width
        self.height = height
        self._snake = snake[:] # for deep copy, otherwise init keeps changing and game never restarts
        self._snake2 = snake[:] # for deep copy, otherwise init keeps changing and game never restarts
        self._alive = True
        self._step = 0
        self._score = 0SNAKE_INIT
        self._walls = self.generate_walls()
        self._food = []
        # self.generate_food()

    def full_state(self):
        state = np.zeros((self.width, self.height))
        for cell in self._snake:
            state[cell[0], cell[1]] = SNAKE_REP
        for cell in self._walls:
            state[cell[0], cell[1]] = WALL_REP
        for cell in self._food:
            state[cell[0], cell[1]] = FOOD_REP
        return state

    def half_state(self):
        # cant handle if snake off screen
        position = [self._snake[0][0], self._snake[0][1]]
        state = np.asarray(self.full_state())

        if position[0]-RANGE < 0:
            position[0] = RANGE
        elif position[0] >= state.shape[0]-RANGE:
            position[0] = state.shape[0]-RANGE-1
        if position[1]-RANGE < 0:
            position[1] = RANGE
        elif position[1] >= state.shape[1]-RANGE:
            position[1] = state.shape[1]-RANGE-1

        field = state[position[0]-RANGE:position[0]+RANGE+1, position[1]-RANGE:position[1]+RANGE+1]
        return field

    def generate_walls(self):
        walls = []
        for x in range(self.width):
            walls.append((x, 0))
            walls.append((x, self.height-1))
        for y in range(1, self.height-1):
            walls.append((0, y))
            walls.append((self.width-1, y))
        return walls

    def generate_food(self):
        np.random.seed(SEED)
        while len(self._food) < NUM_FOOD:
            bite = (np.random.randint(1,self.width-1,1)[0], np.random.randint(1,self.height-1,1)[0])
            if bite not in self._food and bite not in self._snake and bite not in self._walls:
                self._food.append(bite)
        #self._food = [(1,1),(5,6),(8,2),(2,8),(8,8)]

    def vec_add(self, (x1, y1), (x2, y2)):
        return (x1 + x2, y1 + y2)

    def move_snake(self, snake_dir):
        self._snake.insert(0, self.vec_add(self._snake[0], snake_dir))
        self._snake.pop()

    def check_food(self):
        if self._snake[0] in self._food:
            return True
        else:
            return False

    def check_collision(self):
        if self._snake[0] in self._walls:
            return True
        elif self._snake[0] in self._snake[1:]:
            return True
        else:
            return False


    def step(self, snake_dir):
        if self._alive:
            self.move_snake(snake_dir)

        if self.check_collision():
            self._alive = False

        if self.check_food():
            self._score += FOOD_SCORE
            self._food.remove(self._snake[0])

        self._step += 1


















class Graphics(object):
    """ Graphics object for drawing snake

    """

    ID = 0

    def __init__(self, width, height, newwindow=False):
        Graphics.ID += 1
        if Graphics.ID == 1 or newwindow == True:
            self.setup(width, height)

        self.width = Graphics.width
        self.height = Graphics.height
        self.master = Graphics.master
        self.window = Graphics.window


    def setup(self, width, height):
        Graphics.width = width
        Graphics.height = height
        Graphics.master = Tk()
        Graphics.window = Canvas(Graphics.master, width=Graphics.width, height=Graphics.height)
        Graphics.window.pack()

    def draw(self, full_state):
        shape = np.asarray(full_state).shape
        #print(full_state)
        #print(shape)
        box_size = np.min((self.width / shape[0], self.height / shape[1]))
        #print(box_size)
        self.window.delete(ALL)
        for y in range(shape[1]):
            for x in range(shape[0]):
                if full_state[x][y] == BLANK_REP:
                    self.window.create_rectangle(x*box_size, y*box_size, box_size+x*box_size, box_size+y*box_size, width=0, fill='#000')
                elif full_state[x][y] == SNAKE_REP:
                    self.window.create_rectangle(x*box_size, y*box_size, box_size+x*box_size, box_size+y*box_size, width=0, fill='#fff')
                elif full_state[x][y] == WALL_REP:
                    self.window.create_rectangle(x*box_size, y*box_size, box_size+x*box_size, box_size+y*box_size, width=0, fill='#fff444444')
                elif full_state[x][y] == FOOD_REP:
                    self.window.create_rectangle(x*box_size, y*box_size, box_size+x*box_size, box_size+y*box_size, width=0, fill='#77f')
                elif full_state[x][y] == SNAKE2_REP:
                    self.window.create_rectangle(x*box_size, y*box_size, box_size+x*box_size, box_size+y*box_size, width=0, fill='#0f0')
        self.master.update()

    def end(self):
        #self.master.destroy()
        pass























def action_to_dir(action):
    neuron = np.argmax(action)

    if neuron == 0:
        return (0, 1)
    if neuron == 1:
        return (-1, 0)
    if neuron == 2:
        return (0, -1)
    if neuron == 3:
        return (1, 0)







genome_count = 0

def eval_genome(genome, config, display=DISPLAY_ALL):
    net = neat.nn.FeedForwardNetwork.create(genome, config)


    sim = Python(GAME_SIZE, GAME_SIZE, SNAKE_INIT, SNAKE2_INIT)
    if display:
        graph = Graphics(DISPLAY_SIZE, DISPLAY_SIZE)

    fitness = 0.0
    while sim._step < SIM_STEPS:
        # inputs = sim.full_state()
        # inputs = np.asarray(inputs).flatten()
        inputs = sim.half_state().flatten()
        # print(sim.full_state)
        action = net.activate(inputs)

        # force = snake.convert(action)
        sim.step(action_to_dir(action))
        if display:
            graph.draw(sim.full_state())
            print(sim.half_state().transpose())
            time.sleep(STEP_SPEED)


        #print('halfstate', inputs)
        #print('actions', action)
        #print('dir', action_to_dir(action))
        #print('fitness', fitness)

        #time.sleep(1)
        if not sim._alive:
            break

        fitness = sim._score #+ sim._step
    return fitness

    #time.sleep(1)
    #graph.end()
    #master.quit()






def eval_genomes(genomes, config):
    # print('Evaluating Genomes!')

    # np.random.seed(5)
    global SEED
    SEED = np.random.randint(0,255,1)

    for genome_id, genome in genomes:
        # print('Genome_ID:', genome_id)
        genome.fitness = eval_genome(genome, config)



def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    # pe = neat.ParallelEvaluator(4, eval_genome)
    # winner = pop.run(pe.evaluate, MAX_GENERATIONS)
    winner = pop.run(eval_genomes, MAX_GENERATIONS)

    eval_genome(winner, config, display=True)

    with open('winner', 'wb') as f:
        pickle.dump([winner,[SEED, NUM_FOOD, RANGE, GAME_SIZE]], f)

    print(winner)




def run_winner(name):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    data = pickle.load(open(name,'rb'))
    winner = data[0]

    global SEED, NUM_FOOD, RANGE, GAME_SIZE
    SEED = data[1][0]
    NUM_FOOD = data[1][1]
    RANGE = data[1][2]
    GAME_SIZE = data[1][3]

    eval_genome(winner, config, display=True)



















if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_winner(sys.argv[1])
    else:
        run()
