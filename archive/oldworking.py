
import numpy as np

from Tkinter import *

import time

import threading

import os
import neat
import pickle

NUM_FOOD = 5
GAME_SIZE = 10
simulation_steps = 100

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
    def __init__(self, width, height, snake):
        self.width = width
        self.height = height
        self._snake = snake
        self._alive = True
        self._step = 0
        self._score = 0
        self._walls = self.generate_walls()
        self._food = []
        self.generate_food()

    def step(self, snake_dir):
        self.move_snake(snake_dir)

        if self.check_collision():
            self._alive = False

        if self.check_food():
            self._score += 10
            #print('SCORE UP BECAUSE FOOD')
            self._food.remove(self._snake[0])
            #self.generate_food()


        self._step += 1


    def vec_add(self, (x1, y1), (x2, y2)):
        return (x1 + x2, y1 + y2)

    def move_snake(self, snake_dir):
        self._snake.insert(0, self.vec_add(self._snake[0], snake_dir))
        self._snake.pop()

    def check_food(self):
        if self._snake[0] in self._food:
            #print('FOOOOOOOOOOOOOOOOOOOOOD')
            return True
        return False

    def check_collision(self):
        if self._snake[0] in self._walls:
            return True
        elif self._snake[0] in self._snake[1:]:
            return True
        else:
            return False


    def half_state(self):
        range = 1
        position = [self._snake[0][0], self._snake[0][1]]
        state = np.asarray(self.full_state())

        if position[0]-1 < 0:
            position[0] += 1
        elif position[0] >= state.shape[0]:
            position[0] -= 1
        if position[1]-1 < 0:
            position[1] += 1
        elif position[1] >= state.shape[1]:
            position[1] -= 1

        field = state[position[0]-range:position[0]+range+1, position[1]-range:position[1]+range+1]
        return field


    # def smart_cast(self):
    #     if self._snake[0][0]:
    #     pass

    def cheap_cast(self):
        trace = []
        trace.append(self.height - self._snake[0][1]-1) #maybe remove -1's?
        trace.append(self._snake[0][1])
        trace.append(self._snake[0][0])
        trace.append(self.width - self._snake[0][0]-1)
        if self._snake[1][0] < self._snake[0][0]:
            trace[2] = 1
        elif self._snake[1][0] > self._snake[0][0]:
            trace[3] = 1
        elif self._snake[1][1] < self._snake[0][1]:
            trace[1] = 1
        else:
            trace[0] = 1
        return trace

    def cast(self, pos):
        fs = self.full_state()
        trace = []
        dirs = [(0,1),(0,-1),(-1,0),(1,0)]
        for d in dirs:
            x = pos[0]
            y = pos[1]
            dist = 0
            while True:
                #will crash if not hit a block and goes outside of array
                #is also wrong and one off (is at 2 when on edge of map)
                x += d[0]
                y += d[1]
                dist += 1
                if fs[x][y] != 0:
                    break
            trace.append(dist)
        return trace

    def cast2(self, pos):
        fs = self.full_state()
        fs = np.asarray(fs)

        # segments on each side of pos
        left = fs[0:pos[0],pos[1]]
        right = fs[pos[0]+1:,pos[1]]
        up = fs[pos[0],0:pos[1]]
        down = fs[pos[0],pos[1]+1:]

        trace = []
        trace.append(np.argmax(down > 0))
        trace.append(np.argmax(up[::-1] > 0))
        trace.append(np.argmax(left[::-1] > 0))
        trace.append(np.argmax(right > 0))
        return trace

    def full_state(self):
        state = []
        for x in range(self.width):
            col = []
            for y in range(self.height):
                if (x, y) in self._snake:
                    col.append(1)
                elif (x, y) in self._walls:
                    col.append(2)
                elif (x, y) in self._food:
                    col.append(3)
                else:
                    col.append(0)
            state.append(col)
        return state

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
        np.random.seed(2)
        while len(self._food) < NUM_FOOD:
            bite = (np.random.randint(1,self.width-1,1)[0], np.random.randint(1,self.height-1,1)[0])
            if bite not in self._food and bite not in self._snake and bite not in self._walls:
                self._food.append(bite)
        #self._food = [(1,1),(5,6),(8,2),(2,8),(8,8)]














class Graphics(object):
    """ Graphics object for drawing snake

    """

    count = 0

    def __init__(self, width, height, newwindow=False):
        Graphics.count += 1
        if Graphics.count == 1 or newwindow == True:
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
                if full_state[x][y] == 0:
                    #self.window.create_rectangle(x*box_size, y*box_size, box_size, box_size, fill='black')
                    pass
                elif full_state[x][y] == 1:
                    self.window.create_rectangle(x*box_size, y*box_size, box_size+x*box_size, box_size+y*box_size, fill='blue')
                elif full_state[x][y] == 2:
                    self.window.create_rectangle(x*box_size, y*box_size, box_size+x*box_size, box_size+y*box_size, fill='red')
                elif full_state[x][y] == 3:
                    self.window.create_rectangle(x*box_size, y*box_size, box_size+x*box_size, box_size+y*box_size, fill='green')
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

def eval_genome(genome, config, display=False):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    #print(genome)
    #master = Tk()

    #global genome_count
    #genome_count += 1
    #display = True
    #if genome_count%10:
    #display = False

    sim = Python(GAME_SIZE, GAME_SIZE, [(5, 5), (4, 5),(3, 5),(2,5)])
    if display:
        graph = Graphics(500, 500)

    fitness = 0.0
    while sim._step < simulation_steps:
        # inputs = sim.full_state()
        # inputs = np.asarray(inputs).flatten()
        inputs = sim.half_state().flatten()
        action = net.activate(inputs)

        # force = snake.convert(action)
        sim.step(action_to_dir(action))
        if display:
            graph.draw(sim.full_state())
            print(sim.half_state().transpose())
            time.sleep(0.1)


        #print('halfstate', inputs)
        #print('actions', action)
        #print('dir', action_to_dir(action))
        #print('fitness', fitness)

        #time.sleep(1)
        if not sim._alive:
            break

        fitness = sim._score + sim._step
    return fitness

    #time.sleep(1)
    #graph.end()
    #master.quit()


def eval_genomes(genomes, config):
    print('Evaluating Genomes!')
    for genome_id, genome in genomes:
        print('Genome_ID:', genome_id)
        genome.fitness = eval_genome(genome, config, display=False)

def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(4, eval_genome)
    winner = pop.run(pe.evaluate, 20)
    #winner = pop.run(eval_genomes, 300)

    eval_genome(winner, config, display=True)

    with open('winner', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)


def run_winner(name):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    winner = pickle.load(open(name,'rb'))
    eval_genome(winner, config, display=True)



















if __name__ == '__main__':
    run()
    # run_winner('winner')
