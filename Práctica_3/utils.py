import sys 
import random
import copy
import math

def readProblemInstance(filename):
	with open(filename) as f:
		maze = []
		rows, cols = [int(x) for x in next(f).split()]
		maze = [[int(x) for x in line.split()] for line in f]
	
	return rows, cols, maze

def writeProblemInstance(filename):
	with open(filename,'w') as f:
		f.write("%s %s\n", self.rows, self.cols)
		[[int(x) for x in line.split()] for line in f]
	return True	


def getProblemInstance(rows, cols, maxDivisions, garbageCount, seed):
    """
    This method generates a new problem instance. 
    Cells with value 0 means empty cells. Cells with value 1 are walls. 

    Returns a maze (problem instance)
     
    Parameters:
     rows  -- rows of the maze (Int)
     cols  -- cols of the maze (Int)
     seed  -- or the random generator (Int)
     
    """
    random.seed(seed)
    maze = generateWalls(rows, cols, maxDivisions)

    # Open holes in the walls
    openWalls(rows, cols, maze)
		
    # Randomly place garbage in the maze
    while (garbageCount > 0):
        randX = random.randint(0, cols - 1)
        randY = random.randint(0, rows - 1)
        while (maze[randY][randX] in [1,3]):
            randX = random.randint(0, cols - 1)
            randY = random.randint(0, rows - 1)

        maze[randY][randX] = 3
        garbageCount -= 1

    # Randomly place the agent in the maze
    randX = random.randint(0, cols - 1)
    randY = random.randint(0, rows - 1)
    while (maze[randY][randX] in [1,3]):
        randX = random.randint(0, cols - 1)
        randY = random.randint(0, rows - 1)
        
    maze[randY][randX] = 2
    
    return maze

def generateWalls(rows, cols, maxDivisions):
    """
    This method generates a series of walls in a maze, 
    by subdividing the maze space into random rooms.
    No wall edges shall be 1 cell away from each other,
    in order to avoid walls being next to each other.

    Returns: a maze with its walls created

    Parameters:
     rows -- Number of rows in the maze
     cols -- Number of columns in the maze
     maxDivisions -- Maximum number of wall divisions to try to make.
                     If a tiny room cannot be further subdivided into
                     more rooms, that also counts as a try.
    """

    # List of wall edges currently created. Wall edges can be
    # together (4-way wall), but never between 1 and 2 cells apart
    edgeList = [(-1,-1), (cols, -1), (-1, rows), (cols, rows)]

    # List of created rooms. Initially, only the empty maze exists.
    # A room is specified by its four corners, and by in which direction
    # the next cut should be performed (horizontal: 0 or vertical: 1)
    roomList = [(((0,0),(cols - 1, 0), (0, rows - 1), (cols - 1, rows - 1)),1)]

    # As long as there are divisions to perform, extract a room from the room list,
    # divide it, and update the roomList with the newly created rooms
    while (maxDivisions > 0):
        currentRoom = roomList.pop(0)
        newRooms = divideRoom(currentRoom, edgeList)
        for room in newRooms:
            roomList.append(room)
        maxDivisions -= 1

    # Generate maze, and populate with empty space as specified in the rooms
    maze = [ [ 1 for _ in range(cols)] for _ in range(rows)]
    for room in roomList:
        for x in range(room[0][0][0], room[0][1][0] + 1):
            for y in range(room[0][0][1], room[0][2][1] + 1):
                maze[y][x] = 0

    return maze

def divideRoom(roomToDivide, edgeList):
    """
    This method divides a room into two rooms, either with an horizontal
    wall or a vertical wall (as specified in the roomToDivide tuple).
    The newly added wall must not be at a distance between 1 and 2 of
    any pre-existing edge, in order to avoid impossible wall cuts.

    Returns: two rooms if the division was successful, or the original one
             if it was unsuccessful

    Parameters:
     roomToDivide -- The room to try dividing, as a 2-tuple with the following elements:
                        - [0]: 4-tuple of room corners. Each one is a 2-tuple of the (x,y)
                                coordinates of such corner.
                        - [1]: value indicating whether to perform a horizontal/vertical
                                cut. Gets flipped on each cut attempt

     edgeList -- list of current wall edges in the maze. Each one is a 2-tuple of the (x,y)
                  coordinates of such edge.
    """

    successfulCut = False

    # Horizontal cut
    if (roomToDivide[1] == 0):
        # Avoid dividing 1 or 2-row rooms
        if (roomToDivide[0][2][1] - roomToDivide[0][0][1] <= 3):
            return [(roomToDivide[0], 1 - roomToDivide[1])]

        while (not successfulCut):
            successfulCut = True

            # Choose a random position in the horizontal axis, inside the room
            randomPos = random.randint(roomToDivide[0][0][1], roomToDivide[0][2][1])

            # Try cutting the room in two
            potentialEdges = ((roomToDivide[0][0][0], randomPos),(roomToDivide[0][1][0],randomPos))

            # Avoid edges being diagonally close to each other
            for edge in potentialEdges:
                for existingEdge in edgeList:
                    if (euclidean_distance(edge, existingEdge) >= 1 and euclidean_distance(edge, existingEdge) < 2):
                        successfulCut = False

            if (not successfulCut):
                continue

            # Create new rooms
            newRoom1 = ((roomToDivide[0][0], roomToDivide[0][1], (roomToDivide[0][2][0], randomPos - 1), (roomToDivide[0][3][0], randomPos - 1)), 1 - roomToDivide[1])
            newRoom2 = (((roomToDivide[0][0][0], randomPos + 1), (roomToDivide[0][1][0], randomPos + 1), roomToDivide[0][2], roomToDivide[0][3]), 1 - roomToDivide[1])

            # Add new wall edges to edge list
            edgeList.append(potentialEdges[0])
            edgeList.append(potentialEdges[1])

            return [newRoom1, newRoom2]


    # Vertical cut
    elif (roomToDivide[1] == 1):
        # Avoid dividing 1 or 2-column rooms
        if (roomToDivide[0][1][0] - roomToDivide[0][0][0] <= 3):
            return [(roomToDivide[0], 1 - roomToDivide[1])]

        while (not successfulCut):
            successfulCut = True

            # Choose a random position in the vertical axis, inside the room
            randomPos = random.randint(roomToDivide[0][0][0], roomToDivide[0][1][0])

            # Try cutting the room in two
            potentialEdges = ((randomPos,roomToDivide[0][0][1]),(randomPos, roomToDivide[0][2][1]))

            # Avoid edges being diagonally close to each other
            for edge in potentialEdges:
                for existingEdge in edgeList:
                    if (euclidean_distance(edge, existingEdge) >= 1.0 and euclidean_distance(edge, existingEdge) < 2):
                        successfulCut = False

            if (not successfulCut):
                continue

            # Create new rooms
            newRoom1 = ((roomToDivide[0][0], (randomPos - 1, roomToDivide[0][1][1]), roomToDivide[0][2], (randomPos - 1, roomToDivide[0][3][1])), 1 - roomToDivide[1])
            newRoom2 = (((randomPos + 1, roomToDivide[0][0][1]), roomToDivide[0][1], (randomPos + 1, roomToDivide[0][2][1]), roomToDivide[0][3]), 1 - roomToDivide[1])

            # Add new wall edges to edge list
            edgeList.append(potentialEdges[0])
            edgeList.append(potentialEdges[1])

            return [newRoom1, newRoom2]

    else:
        raise ValueError("Wrong cut direction!")

def openWalls(rows, cols, maze):
    """
    Given a maze with some walls already built, this method
    randomly deletes some of these walls, in order to have each room
    fully connected with its neighboring rooms.

    This method works by detecting all wall intersections (including walls next to the maze bounds),
    and then traversing each cardinal direction until the next intersection. Excluding these two bounds,
    a random wall in-between is removed

    Returns: nothing. The maze is modified in-place

    Parameters:
     rows -- The number of rows in the maze
     cols -- The number of columns in the maze
     maze -- The maze itself, as a 2D Python matrix
    """
    
    # List of wall intersections
    intersectionList = []

    # Build intersection list
    for x in range(cols):
        for y in range(rows):
            if (testIntersection(maze, x, y, rows, cols)):
                intersectionList.append((x, y))

    # For each intersection, traverse each cardinal direction until the bounds of the maze or until
    # the next intersection. Then, remove a random wall in-between these two points (unless they are)
    # 2 units apart.
    # However, if an empty space is found while travelling from an intersection to the next one,
    # then a hole was already opened in this wall, so if that's the case, skip that iteration 
    cardinalDisplacements = [(-1,0),(1,0),(0,-1),(0,1)]
    for intersection in intersectionList:
        for disp in cardinalDisplacements:
            # Skip directions that lead points in the bound towards OOB positions
            if ((intersection[0] == 0 and disp[0] == -1) or
                (intersection[0] == cols - 1 and disp[0] == 1) or
                (intersection[1] == 0 and disp[1] == -1) or
                (intersection[1] == rows - 1 and disp[1] == 1)):
                continue

            # Traverse the wall in the specified direction. If a 0 is found,
            # ignore the wall and carry on with the next iteration
            nextPoint = (intersection[0] + disp[0], intersection[1] + disp[1])
            holeFound = False
            while (not testIntersection(maze, nextPoint[0], nextPoint[1], rows, cols)):
                if (maze[nextPoint[1]][nextPoint[0]] != 1):
                    holeFound = True
                    break
                
                nextPoint = (nextPoint[0] + disp[0], nextPoint[1] + disp[1])

            if (holeFound):
                continue

            # Open a hole randomly in-between both points, unless both intersections
            # are only a unit apart
            if (euclidean_distance(intersection, nextPoint) > 1.1):
                randX = random.randint(min(intersection[0] + disp[0],nextPoint[0] - disp[0]), max(intersection[0] + disp[0],nextPoint[0] - disp[0]))
                randY = random.randint(min(intersection[1] + disp[1],nextPoint[1] - disp[1]), max(intersection[1] + disp[1],nextPoint[1] - disp[1]))
                maze[randY][randX] = 0


def testIntersection(maze, x, y, rows, cols):
    """
    This method tests if a given position (x,y) in a maze is an intersection.
    An intersection is a wall, either in the bounds of the maze, or a wall
    surrounded by 3 or more walls (only in the cardinal directions)

    Returns: true if it is an intersection. False otherwise

    Parameters:
     maze -- The maze in which the intersection test is to be performed
     x -- The x-coordinate of the point to test
     y -- The y-coordinate of the point to test
     rows -- The number of rows in the maze
     cols -- The number of columns in the maze
    """
    if (maze[y][x] != 1):
        return False

    else:
        if (x == 0 or x == cols - 1 or y == 0 or y == rows - 1):
            return True
        else:
            numWallNeighbors = 0
            cardinalDisplacements = [(-1,0),(1,0),(0,-1),(0,1)]
            for disp in cardinalDisplacements:
                if (maze[y+disp[1]][x+disp[0]] == 1):
                    numWallNeighbors += 1
            
            return (numWallNeighbors >= 3)


def printMaze(maze):
	for i in range(len(maze)):
		print(maze[i])


def euclidean_distance(p1, p2): # simple function, I hope you are more comfortable
  return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def manhattan_distance(p1, p2): # simple function, I hope you are more comfortable
  return abs(p1[0]-p2[0])+abs(p1[1]-p2[1])
