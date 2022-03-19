import pygame
import math
from queue import PriorityQueue
import random

WIDTH = 950
ROWS = 50
WIN = pygame.display.set_mode((WIDTH, WIDTH))

pygame.display.set_caption("Path Finding Algorithm and Visualizer: M-Mazes, D-Djikstra's, G-Greedy, A-A*, B-BFS")

RED = (255, 0, 0)
OFF_RED = (254, 0, 0)
GREEN = (0, 255, 0)
OFF_GREEN = (0, 254, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 100)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)
RED_GREY = (192, 64, 64)
GREEN_GREY = (64, 192, 64)


class Node:
  def __init__(self, row, col, width, total_rows):
    self.row = row
    self.col = col
    self.x = row * width
    self.y = col * width
    self.width = width
    self.total_rows = total_rows
    self.color = WHITE 
    self.neighbors = []
    self.maze_neighbors = []
    self.visited = False
  def get_pos(self):
    return self.row, self.col
  def is_closed(self):
    return self.color == RED
  def is_open(self):
    return self.color == GREEN
  def is_barrier(self):
    return self.color == BLACK
  def is_start(self):
    return self.color == ORANGE
  def is_end(self):
    return self.color == TURQUOISE
  def is_weight(self):
    return self.color == GREY
  def is_path(self):
    return self.color == PURPLE
  def is_half_closed(self):
    return self.color == OFF_RED
  def is_half_open(self):
    return self.color == OFF_GREEN
  def is_visited(self):
    return self.visited
  def reset(self):
    self.color = WHITE
  def make_closed(self):
    self.color = RED
  def make_open(self):
    self.color = GREEN
  def make_barrier(self):
    self.color = BLACK
  def make_start(self):
    self.color = ORANGE
  def make_end(self):
    self.color = TURQUOISE
  def make_path(self):
    self.color = PURPLE
  def make_weight(self):
    self.color = GREY
  def make_half_closed(self):
    self.color = OFF_RED
  def make_half_open(self):
    self.color = OFF_GREEN
  def make_visited(self):
    self.visited = True
  def draw(self, win):
    pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))
  def update_neighbors(self, grid):
    self.neighbors = []
    if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # down
      self.neighbors.append(grid[self.row + 1][self.col])

    if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # up
      self.neighbors.append(grid[self.row - 1][self.col])

    if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # right
      self.neighbors.append(grid[self.row][self.col + 1])

    if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # left
      self.neighbors.append(grid[self.row][self.col - 1])
  def maze_update_neighbors(self, grid):
    self.maze_neighbors = []
    if self.row < self.total_rows - 2 and not grid[self.row + 2][self.col].is_visited(): # down
      self.maze_neighbors.append(grid[self.row + 2][self.col])

    if self.row > 0 and not grid[self.row - 2][self.col].is_visited(): # up
      self.maze_neighbors.append(grid[self.row - 2][self.col])

    if self.col < self.total_rows - 2 and not grid[self.row][self.col + 2].is_visited(): # right
      self.maze_neighbors.append(grid[self.row][self.col + 2])

    if self.col > 0 and not grid[self.row][self.col - 2].is_visited(): # left
      self.maze_neighbors.append(grid[self.row][self.col - 2])
  def __lt__(self, other):
    return False

    
def heuristic(p1, p2):
  x1, y1 = p1
  x2, y2 = p2
  return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw):
  while current in came_from:
    current = came_from[current]
    current.make_path()
    draw()

def algorithm(draw, grid, start, end, is_greedy, h_score_multiplier, g_score_multiplier, is_weighted): # Algorithm that can be A*, Djikstra's, BFS, Swarm, Convergent Swarm, or Greedy Best-First Search with different multipliers passed in.

  if not is_weighted: # If the algorithm we have passed in does not account for weights, remove all the weights.
    for row in grid:
      for node in row:
        if node.is_weight():
          node.reset()

  count = 0
  open_set = PriorityQueue()
  open_set.put((0, count, start))
  came_from = {} # stores the path the pathfinder takes
  g_score = {node: float("inf") for row in grid for node in row}
  g_score[start] = 0 # the start position has a a g_score of 0
  f_score = {node: float("inf") for row in grid for node in row}
  f_score[start] = h_score_multiplier * heuristic(start.get_pos(), end.get_pos()) # distance from start to end, we don't need to add the g_score, because it is 0 anyways
  
  open_set_hash = {start}

  while not open_set.empty(): #While the open set has not run out of nodes to check
    for event in pygame.event.get(): #Allow the user to close the visualizer during the algorithm
      if event.type == pygame.QUIT:
        pygame.quit()

    current = open_set.get()[2] #Gets the node with lowest value f-score in the open-set - sorted by f-score, then count
    open_set_hash.remove(current) #Synchronizes with the open_set

    if current == end: # If we found the path
      reconstruct_path(came_from, end, draw) # Reconstructs the path
      end.make_end() #Makes the start and end nodes not the path color
      start.make_start()
      return True 
    
    for neighbor in current.neighbors: # checks all the neighbors of the node we are on right now
      if neighbor.is_weight(): #If the neighbor is weighted
        temp_g_score = g_score[current] + 15 # add 15 to the g_score, instead of 1 
      else: #If the neighbor is not weighted
        temp_g_score = g_score[current] + int(not is_greedy) # add 1 to the g_score IF the algorithm is not greedy, which is the normal thing to add. If the algorithm is greedy, then do not add anything, since Greddy algorithms do not care about the g-score
        

      if temp_g_score < g_score[neighbor]: #If we found a  better way to reach this node
        came_from[neighbor] = current #Adds to the came_from table, making sure that now the neighbor is set to coming from this current node, and not any node that gives this a higher g-score
        g_score[neighbor] = g_score_multiplier * temp_g_score #set the g_score of the neghbor to the calculated g_score we got for it
        f_score[neighbor] = g_score_multiplier * temp_g_score + h_score_multiplier * heuristic(neighbor.get_pos(), end.get_pos()) #set the f_score with g_score plus the new calculated f_score
        if neighbor not in open_set_hash: # If the neighor is not in the open_set
          count += 1 #Adding the count of the things in the set
          open_set.put((f_score[neighbor], count, neighbor)) #Add the neighbor into the open_set
          open_set_hash.add(neighbor) #Add it into the open_set_hash as well
          if not (neighbor.is_weight() or neighbor.is_half_closed() or neighbor.is_half_open() or neighbor.is_closed()):
            neighbor.make_open() #Makes it look open
          elif neighbor.is_weight():
            neighbor.make_half_open()

    draw()

    if current != start: #If the current node is not at the start
      if not (current.is_weight() or neighbor.is_half_closed() or neighbor.is_half_open()):
        current.make_closed()
      elif current.is_weight() or neighbor.is_half_open():
        current.make_half_closed()

  return False

def depth_first_search_algorithm(draw, grid, start, end): #Depth first search is a really bad pathfinding algorithm, so I added it in to show how bad it is
  for row in grid:
    for node in row:
      if node.is_weight():
        node.reset()

  open_set = []
  open_set.append(start)
  came_from = {}
  closed_set = []

  while open_set:
    for event in pygame.event.get(): #Allow the user to close the visualizer during the algorithm
      if event.type == pygame.QUIT:
        pygame.quit()
    
    current = open_set.pop()
    for neighbor in current.neighbors: # checks all the neighbors of the node we are on right now
      if neighbor == end: #If we found a way to reach this node
        came_from[neighbor] = current
        print("Reconstructing path")
        reconstruct_path(came_from, end, draw) # Reconstructs the path
        end.make_end() #Makes the start and end nodes not the path color
        start.make_start()
        return True 
      elif neighbor not in open_set and neighbor not in closed_set: # If the neighor is not in the open_set
        came_from[neighbor] = current
        open_set.append(neighbor) #Add the neighbor into the open_set
        if neighbor != start:
          if neighbor.is_closed:
            neighbor.make_open() #Makes it look open
          else :
            neighbor.make_closed() 

    draw()

    if current != start: #If the current node is not at the start
      closed_set.append(current)
      current.make_closed()

  return False


def make_grid(rows, width):
  grid = []
  gap = width // rows
  for i in range(rows):
    grid.append([])
    for j in range(rows):
      node = Node(i, j, gap, rows)
      grid[i].append(node)

  return grid

def draw_grid(win, rows, width):
  gap = width // rows
  for i in range(rows):
    pygame.draw.line(win, BLACK, (0, i * gap), (width, i * gap))
    for j in range(rows):
      pygame.draw.line(win, BLACK, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
  win.fill(WHITE)
  
  for row in grid:
    for node in row:
      node.draw(win)

  draw_grid(win, rows, width)
  pygame.display.update()

def get_clicked_pos(pos, rows, width):
  gap = width // rows
  y, x = pos

  row = y // gap
  col = x // gap

  return row, col

def clear_board_path(grid):
  for row in grid:
    for node in row:
      if node.is_open() or node.is_closed() or node.is_path():
        return True
  return False

def make_recursive_division_maze(draw, grid, start, end, rows):
  pass
      
def make_randomized_maze(draw, grid, start, end, has_weights, has_walls):
  for row in grid:
    for node in row:
      random_number = random.randint(1, 2)
      random_number_two = random.randint(1, 2)
      random_number_three = random.randint(1, 2)
      if node != start and node != end:
        if random_number == 2 and has_weights and not node.is_barrier() and not has_walls:
          node.make_weight()
        elif random_number == 2 and has_walls and not has_weights and not node.is_weight():
          if random_number_two == 1 or random_number_three == 1:
            node.make_barrier()
        elif random_number == 2 and has_weights and has_walls:
          if random_number_two == 1 and not node.is_weight():
            node.make_barrier()
          elif not node.is_barrier():
            node.make_weight()
      draw()

def stair_maze(draw, grid, start, end):
  y_of_just_placed = ROWS - 2
  for row in grid:
    for node in row:
      if node.row >= 1 and node.row <= ROWS - 1:
        if node.col == y_of_just_placed:
          node.make_barrier()
          y_of_just_placed = y_of_just_placed - 1

def remove_wall(a,b,grid):	# removes the barrier between the current cell and chosen_one
	x = (a.row + b.row) // 2
	y = (a.col + b.col) // 2
	grid[x][y].reset()
	return grid

def iterative_backtracking_maze(draw, grid, start, end):
  for row in grid:
    for node in row:
      if node != start and node != end:
        node.make_barrier()
  print("Got to checkpoint 1")
  stack = []
  current = grid[1][1]
  current.make_visited()
  stack.append(current)
  
  print("Got to checkpoint 2")

  while stack:
    for event in pygame.event.get(): #Allow the user to close the visualizer during the algorithm
      if event.type == pygame.QUIT:
        pygame.quit()

    current = stack.pop()
    if current.maze_neighbors:
      stack.append(current)
      chosen_neighbor = random.choice(current.maze_neighbors)
      chosen_neighbor.make_visited()
      grid = remove_wall(current, chosen_neighbor, grid)
      current.make_visited()
      current = chosen_neighbor
      if chosen_neighbor not in stack:
        print("Got to checkpoint 5")
        stack.append(chosen_neighbor)
    else:
      print("Got to checkpoint 3-2")
      current.make_visited()
      current = stack.pop(-1)
      print("Got to checkpoint 4-2")

  return True

def node_update_neighbors(grid):
  for row in grid:
    for node in row:
      node.update_neighbors(grid)

def caption():
  pygame.display.set_caption("Path Finding Algorithm and Visualizer")

def main(win, width):
  
  grid = make_grid(ROWS, width)

  start = None
  end = None

  run = True

  while run:
    draw(win, grid, ROWS, width)
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        run = False


      if pygame.mouse.get_pressed()[0]:
        pos = pygame.mouse.get_pos()
        row, col = get_clicked_pos(pos, ROWS, width)
        node = grid[row][col]
        
        if not start and node != end:
          start = node
          start.make_start()
        elif not end and node != start:
          end = node
          end.make_end()
        elif node != end and node != start:
          node.make_barrier()
      elif pygame.mouse.get_pressed()[2]:
        pos = pygame.mouse.get_pos()
        row, col = get_clicked_pos(pos, ROWS, width)
        node = grid[row][col]
        node.reset()
        if node == start:
          start = None
        elif node == end:
          end = None
      elif pygame.mouse.get_pressed()[1]:
        pos = pygame.mouse.get_pos()
        row, col = get_clicked_pos(pos, ROWS, width)
        node = grid[row][col]

        if not start and node != end:
          start = node
          start.make_start()
        elif not end and node != start:
          end = node
          end.make_end()
        elif node != end and node != start and not node.is_barrier():
          node.make_weight()


      if event.type == pygame.KEYDOWN:
        no_algorithm_previously_run = clear_board_path(grid)
          

          
        if start and end and not no_algorithm_previously_run:
          if event.key == pygame.K_a:
            pygame.display.set_caption("A* Path Finding Algorithm and Visualizer")  
            node_update_neighbors(grid)
            algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end, False, 1.000_000_1, 1, True)
          elif event.key == pygame.K_j:
            pygame.display.set_caption("Djikstra's Path Finding Algorithm and Visualizer")
            node_update_neighbors(grid)
            algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end, False, 0, 1, True)
          elif event.key == pygame.K_g:
            pygame.display.set_caption("Greedy Best-first Search Path Finding Algorithm and Visualizer")
            node_update_neighbors(grid)
            algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end, True, 1, 1, True)
          elif event.key == pygame.K_s:
            pygame.display.set_caption("Swarm Path Finding Algorithm and Visualizer")
            node_update_neighbors(grid)
            algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end, False, 1, 1.08, True)
          elif event.key == pygame.K_b:            
            pygame.display.set_caption("Breadth-First Search Path Finding Algorithm and Visualizer")
            node_update_neighbors(grid)
            algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end, False, 0, 1, False)
          elif event.key == pygame.K_x:
            pygame.display.set_caption("Convergent Swarm Path Finding Algorithm and Visualizer")
            node_update_neighbors(grid)
            algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end, False, 1, 1.02, True)
          elif event.key == pygame.K_d:
            pygame.display.set_caption("Depth-first Search Path Finding Algorithm and Visualizer")
            node_update_neighbors(grid)
            depth_first_search_algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)
          if event.key == pygame.K_1:
            make_randomized_maze(lambda: draw(win, grid, ROWS, width), grid, start, end, True, False)
          elif event.key == pygame.K_2:
            make_randomized_maze(lambda: draw(win, grid, ROWS, width), grid, start, end, True, True)
          elif event.key == pygame.K_3:
            make_randomized_maze(lambda: draw(win, grid, ROWS, width), grid, start, end, False, True)
          elif event.key == pygame.K_4:
            stair_maze(lambda: draw(win, grid, ROWS, width), grid, start, end)
          elif event.key == pygame.K_5:
            for row in grid:
              for node in row:
                node.maze_update_neighbors(grid)
            iterative_backtracking_maze(lambda: draw(win, grid, ROWS, width), grid, start, end)
            

        if event.key == pygame.K_r:
          caption()
          start = None
          end = None
          grid = make_grid(ROWS, width)
        elif event.key == pygame.K_c:
          caption()
          for row in grid:
            for node in row:
              if node.is_weight() or node.is_barrier() or node.is_open() or node.is_closed() or node.is_path() or node.is_half_open() or node.is_half_closed():
                node.reset()
        elif event.key == pygame.K_LCTRL:
          caption()
          for row in grid:
            for node in row:
              if node.is_open() or node.is_closed() or node.is_path():
                node.reset()
              elif node.is_half_closed() or node.is_half_open():
                node.make_weight()
        


  pygame.quit()




main(WIN, WIDTH)



