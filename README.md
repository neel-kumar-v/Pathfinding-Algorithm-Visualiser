# Pathfinding Algorithm Visualiser
 
An implementation of Wikipedia's A* pathfinding pseudocode, along with a visualiser and an extension including Breadth-first search, Dijkstra's, etc.
## How To Use

### Making the 'Maze'
#### Custom
* Click on any 2 nodes to add the start and end nodes.
* If both the start and end nodes have been placed, then clicking will now add wall nodes.
* Use middle mouse to add weighted nodes. 
  * __Note:__ Some algorithms will ignore these weighed nodes
* Right click on a square to reset it.
  * __Note:__ If a start or an end node is reset, the program will force you to place that first before any more walls.
#### Preset/Random
* Press 1 for randomly placed weight nodes
* Press 2 for a mix of randomly placed weight and wall nodes
* Press 3 for randomly placed wall nodes
* Press 4 for a diagonal line of walls through the middle leaving only 1 opening
You can also edit these using the editing tools mentioned above

### Algorithms
* Press A for A*
* Press B for Breadth-first search
* Press J for Dijkstra's
* Press D for Depth-first search
* Press G for Greedy Best-first Search


### Clearing the board
* Press R to completely reset the board
* Press C to clear everything except the start and end nodes
* Press left CTRL to clear the algorithm's search

### Known Issues
* DFS sometimes crashes the program
