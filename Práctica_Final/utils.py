# -*- coding: utf-8 -*-

# -- Sheet --

# <br><br><br>
# <h1><font color="#B30033" size=5>Intelligent Systems - Course 2022-2023</font></h1>
# 
# 
# 
# <h1><font color="#B30033" size=5>Lab 1: State Space Search</font></h1>
# 
# 
# <br>
# <div style="text-align: left">
# <font color="#4E70BE" size=3>Lecturers:</font><br>
# <ul>
#   <li><font color="#4E70BE" size=3>Juan Carlos Alfaro Jiménez (JuanCarlos.Alfaro@uclm.es)</font><br></li>
#   <li><font color="#4E70BE" size=3>Guillermo Tomás Fernández Martín (Guillermo.Fernandez@uclm.es)</font><br></li>
#   <li><font color="#4E70BE" size=3>Mª Julia Flores Gallego (Julia.Flores@uclm.es)</font><br></li>
#   <li><font color="#4E70BE" size=3> José Antonio Gámez Martín (Jose.Gamez@uclm.es)</font><br></li>
#   <li><font color="#4E70BE" size=3> Ismael García Varea (Ismael.Garcia@uclm.es)</font><br></li>
#   <li><font color="#4E70BE" size=3> Luis González Naharro (Luis.GNaharro@uclm.es)</font><br></li>
# </ul>
# </div>


# <br>
# <div style="text-align: left">
# <font color="#4E70BE" size=3>Integrantes Grupo 7:</font><br>
# <ul>
#   <li><font color="#4E70BE" size=3>Daniel Cabañero Pardo</font><br></li>
#   <li><font color="#4E70BE" size=3>Pedro Jesús Martínez Herrero</font><br></li>
# 
# </ul>


# --------------------
# ## 1. Introduction
# 
# In this assignment, we will put into practice the techniques for searching the state of spaces. To do that, some of the algorithms seen in units two and three will be implemented and used to solve a classical problem: searching paths on maps where the locations will be cities identified by their latitude and longitude values, as in most [geographical systems](https://en.wikipedia.org/wiki/Geographic_coordinate_system).
# 
# We will also analyze and compare the performance of the algorithms by running them over different instances of the problem, and providing distinct initial and goal states.


# ## 2. Problem Description
# 
# The concept of map we will use is simple: it can be represented by a graph with cities and undirected connections (that is, they work exactly the same in both ways), which indicate that there is a specific road between them two, which can be used for moving from one to the other in one action. Also, these edges will have associated a number of units, which tipycally represents the real/driving distance between the two cities.
# 
# We opted to use realistic maps so that the cities are real, and the driving distances are also extracted from a navigation API. But the connections are established so that only some of them are taken.
# 
# A map is a particular problem, but then we need to answer queries where there will be an initial state and a final state. In the most simple way, both will be the location/city. So to reach city B from A, we would aim at finding the finding the shortest path (smallest cost).


# ## 3. Assignment Development
# 
# During the development of the assignment, you will be given a set of maps, in which you should perform a list of searches. The dimensionality, both in the number of cities and in their connectivity, will be variable, and your algorithms should be efficient enough to work properly in all of them. Some other scenarios (maps and searches) will be kept for the evaluation/correction/interview, so make your code general enough to load them easily.


# ### 3.1 Input Problems
# 
# Every scenario will have associated a JSON file with the following structure: 
# 
# ```JSON
# {
#     "map": {
#         "cities": [
#             {
#                 "id": id_city_0,
#                 "name": name_city_0,
#                 "lat": latitude_city_0,
#                 "lon": longitude_city_0
#             }
#         ],
#         "roads": [
#             {
#                 "origin": origin_city_id,
#                 "destination": destination_city_id,
#                 "distance": road_distance
#             }
#         ]
#     },
#     "departure": departure_city_id,
#     "goal": goal_city_id
# }
# ```
# 
# There are three general keys in the JSON: 
# 
# - `map`: A dictionary that represents the map of the problem.
# - `departure`: The trip departure city id, this is, the initial state.
# - `goal`: The trip goal city id, this is, the end state.
# 
# In the map dictionary, there are two keys: 
# - `cities`: An array with the cities, this is, the nodes of the map.
# - `roads`: An array with the roads, this is, the connections between nodes.
# 
# Finally, a city is represented as: 
# - `id`: The id of the city, used for most operations.
# - `name`: The name of the city, used for representing the solution in a human readable way.
# - `lat`: The latitude of the city, used for plotting representations.
# - `lon`: The longitude of the city, used for plotting representations.
# 
# And a road is represented as: 
# - `origin`: The origin city id.
# -  `destination`: The destination city id.
# -  `distance`: The distance in kilometers using that road.
# 
# The roads will be directed but the JSON will have symmetric roads, meaning that there will be a road from A to B and a second road from B to A.


# ## 4. Work plan


# ### 4.1 Problem Formalization and Examples
# 
# First of all, path finding in maps must be formalized as a problem of search in the space of states, by defining its basic elements. All implementations must refer to search in graphs, so it is important to take into consideration that repeated states must be controlled. 


# ### 4.2 Implementation
# 
# Below, you will have the class structure regarding the problem at hand. You will have to complete the following classes by implementing the algorithms studied in theory. Add all your imports in this cell to have the notebook properly organized.


# =============================================================================
# Imports
# =============================================================================

# Standard
import json
import random
import copy
import itertools
import time
import geopy.distance
from abc import ABC, abstractmethod

# Third party
import geopandas as gpd
from shapely.geometry import Point
from queue import PriorityQueue
from numpy import sin, cos, arccos, pi, round

# #### Class `Action`
# This class provides the **representation of the actions** that will be performed by the traveler. An action is defined by the `origin` and `destination` of the trip, as well as the cost of the action.
# 
# Methods you must add: 
# 
# - `__init__(self, args)`: Constructor of the class, with the necessary arguments
# 
# Methods recommended: 
# 
# - `__repr__(self)`: String representation of the objects. Useful to debug list of `Action`
# - `__str__(self)`: Method used when `print(Action)` is called. Useful to debug single `Action`


class Action:

    def __init__(self, origin, dest, actioncost):
        self.origin = origin
        self.dest = dest
        self.actioncost = actioncost

    def __str__(self):
        return str(self.origin) + " ---> " + str(self.dest) + "   con distancia: " + str(self.actioncost) + " km"

# #### Class `State`
# 
# This class provides the **representation of a state** in the search space. In this problem, a state is defined by the city in which the traveler is in a particular moment. Note that the map itself does not need to be part of the state given that it does not change during the search.
# 
# Methods you must add: 
# 
# - `__init__(self, args)`: Constructor of the class, with the necessary arguments
# - `__eq__(self, other)`: Equals method. Used for hash table comparison
# - `__hash__(self)`: Hashing method. Used to generate unique hashes of the objects. Used for hash table.
# - `apply_action(self, args)`: given a valid `Action`, returns the new `State` generated from applying the `Action` to the current `State`. 
# 
# Methods recommended: 
# 
# - `__repr__(self)`: String representation of the objects. Useful to debug list of `State`
# - `__str__(self)`: Method used when `print(State)` is called. Useful to debug single `State`


class State:
    def __init__(self, idcity):
        self.id = idcity

    def __eq__(self, othercity):
        if not isinstance(othercity, State):
            return NotImplemented
        return self.id == othercity.id
    
    def __hash__(self):
        return hash(self.id)
        
    #Si el estado actual coincide con el origen de la acción, se aplicará la acción
    def apply_action(self, nextaction):
        if nextaction.origin == self.id:
            self.id = nextaction.dest
            return State(self.id)
        else:
            print('Ciudad actual no coincide con accion')
    #----------200-------------
    def __str__(self):
        return f"{self.id}"


# #### Class `Node`. 
# This class provides a **representation of a node** in the search graph. A `Node` is defined by the `State` it represents, its parent `Node` and the `Action` taken to reach the current `Node` from the parent `Node`. 
# 
# **It can also have any other attributes necessary for the search algorithms**.
# 
# Methods you must add: 
# 
# - `__init__(self, args)`: Constructor of the class, with the necessary arguments
# - `__eq__(self, other)`: Equals method. Used for hash table comparison
# 
# Methods recommended: 
# 
# - `__repr__(self)`: String representation of the objects. Useful to debug list of `Node`
# - `__str__(self)`: Method used when `print(Node)` is called. Useful to debug single `Node`


class Node:
    
    def __init__(self, state, parent, action):
        self.state = state 
        self.parent = parent 
        self.action = action 
        #Al definir aquí gCost, hCost y fCost, nos será más facil y evidente la implementación del código, ya que cada nodo tendrá
        # sus costes asociados
        self.gCost = 0.0
        self.hCost = 0.0
        self.fCost = 0.0

    #Si el nodo pasado por parámetro es instancia de la clase Node, devolvemos True si está en el mismo estado y tiene el mismo
    #padre, ya que pueden estar en el mismo estado, pero llegar a través de distintos padres (caminos)
    def __eq__(self, othernode):
        if not isinstance(othernode, Node):
            return NotImplemented
        return self.state == othernode.state and self.parent == othernode.parent
    
    def __str__(self):
        return f"(Nodo ||Estado: {self.state}, Padre: {self.parent}, Action: {self.action})"

# #### Class `Problem`
# This class provides the **representation of the search problem**. This class reads from a file an instance of the problem to be solved. It is defined by the `map`, the `initial_state` and the `final_city` as well as several auxiliary structures to plot the problem. This class must also generate all the `Actions` that can be taken in the problem. It is recommended to store them in a dictionary of the form `{'origin_city_id': [action_1, action_2, action_3, ...]}`. 
# 
# Methods you must add: 
# 
# - `__init__(self, args)`: Constructor of the class, with the necessary arguments
# - Method to generate the `Action` dictionary from the map
# - Method to check if a `State` is the goal state
# 
# Methods recommended: 
# 
# - Method to get the `Actions` associated to a particular `Node`
# 
# The class `Problem` also has an auxiliary method to plot the whole map and to plot the map with a solution of the problem formed by a list of actions. This can be called as `problem.plot_map()` for the whole map and all its connections, or as `problem.plot_map([action_1, action_2, ...])` to plot just the solution.


class Problem:

    def __init__(self, problem, initial, goal):
        # Method to read the problem JSON file
        #with open(filename, 'r', encoding='utf8') as file:
        #    problem = json.load(file)
        
        # Auxiliary structures for the plot_map function
        self.cities = {city['id']: city for city in problem['map']['cities']}
        self.gdf = gpd.GeoDataFrame(problem['map']['cities'])
        self.gdf['Coordinates'] = list(zip(self.gdf.lon, self.gdf.lat))
        self.gdf['Coordinates'] = self.gdf['Coordinates'].apply(Point)
        self.gdf.set_geometry('Coordinates', inplace=True)
        # TODO: Add your code here to complete the constructor
        self.problem = problem
        self.warehouse = problem['warehouse']
        self.initial_state = State(initial)
        self.final_city = State(goal)
        self.actions = {}
        self.dicc = {}
    #---------300----------
        
    # TODO: Add your code here to implement the class methods
    #Cargar estados, acciones y métodos
    def getActions(self):
        #Almacenamos las acciones en un diccionario de una entrada 'Acciones'
        self.actions = {'Acciones' : self.problem['map']['roads']}

        #Recorriendo el diccionario auxiliar self.cities y el diccionario self.actions podemos asociar las id's de las ciudades
        #a los orígenes de las acciones para tenerlas clasificadas por idCity, de la forma que se muestra en el enunciado
        for i in self.cities:
            acc=[]
            for j in range(len(self.problem['map']['roads'])):
                if self.cities[i]['id'] == self.actions['Acciones'][j]['origin']:
                    acc.append(self.actions['Acciones'][j])
            self.dicc[i] = acc
        return self.dicc
    
    def getActionsNode(self, node):
        #Llamamos a la función anterior y le realizamos una consulta al diccionario que nos devuelve por la id requerida
        self.dicc = self.getActions()
        return self.dicc[node.state.id]
    
    def isGoal(self, state):
        return state.id == self.final_city.id
    
    #Esta función resulta de utilidad para obtener el nodo Inicial en la clase Search, antes de empezar el algoritmo
    def getInitialState(self):
        return self.initial_state
    
    def plot_map(self, action_list, world_name='Spain'):
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        city_ids = {self.cities[city]['name']: city for city in self.cities}
        # We restrict to Spain.
        ax = world[world.name == world_name].plot(
            color='white', edgecolor='black',linewidth=3,figsize=(100,70))

        self.gdf.plot(ax=ax, color='red',markersize=500)
        for x, y, label in zip(self.gdf.Coordinates.x, self.gdf.Coordinates.y, self.gdf.name):
            ax.annotate(f'{city_ids[label]} -- {label}', xy=(x, y), xytext=(8, 3), textcoords="offset points",fontsize=60)
        roads = itertools.chain.from_iterable(self.actions.values())

        for road in roads:
            #Hemos tenido que cambiar la forma de llamar a las carreteras, ya que con el diccionario que estabamos usando no 
            #reconocía road.origin
            slat = self.cities[road['origin']]['lat']
            slon = self.cities[road['origin']]['lon']
            dlat = self.cities[road['destination']]['lat']
            dlon = self.cities[road['destination']]['lon']

            for i in range(len(action_list)):
                if action_list[i].origin == road['origin'] and action_list[i].dest == road['destination']:
                    color = 'red'
                    linewidth = 15
                else:
                    color = 'lime'
                    linewidth = 5
                ax.plot([slon , dlon], [slat, dlat], linewidth=linewidth, color=color, alpha=0.5)

# #### Class `Search`
# 
# The `Search` class is in abstract class that contains some attributes:
# - The `Problem` to solve.
# - The list of `open` nodes, i.e. nodes in the frontier, which data structure varies from the different algorithms.
# - The list of `closed` nodes to implement the graph search, which must be implemented using a `set` data structure.
# - The statistics from the algorithm, this is:
#     - The execution time (in ms) to obtain the solution.
#     - The cost of the solution.
#     - The number of generated nodes.
#     - The number of expanded nodes.
#     - The maximum number of nodes simultaneously stored in memory.
#     - The solution obtained (sequence of actions).
# 
# This class also provides some abstract methods:
# - `insert_node(self, node, node_list)`: Method to insert a node in the proper place in the list. May vary from one algorithm to another.
# - `extract_node(self, node_list)`: Method to extract a node from the open node list. Can vary from one data structure to another.
# - `is_empty(self, node_list)`: Method to check if the open node list is empty. Can vary from one data structure to another.
# 
# Methods you must add: 
# - `__init__(self, args)`: Constructor of the class, with the necessary arguments
# - `get_successors(self, node)`: this method implements the successors function and should return a list with all the valid `Node` successors of a given `Node`. You must implement this method.
# - `do_search(self)`: this method implements the graph search you have studied in class. It also provides some statistics of the search process.
# - A method that returns a human readable list of cities from a list of actions. It should be used to return a readable solution instead of the raw list of actions. 
# 
# Note that these methods have to be compatible with both informed and uninformed search. This is why you have `insert_node`, `extract_node` and `is_empty`: as you will need to use different data structures for informed and uninformed algorithms, just by implementing those methods you can make the general `Search` class agnostic to the data structure underlying. 


class Search(ABC):
    @abstractmethod
    def insert_node(self, node, node_list):
        pass
    
    @abstractmethod
    def extract_node(self, node_list):
        pass

    @abstractmethod
    def is_empty(self, node_list):
        pass
    
    def __init__(self, prob, heuristica):
        self.problem = prob
        self.open = []
        self.closed = set()

        self.cost = 0
        self.expandedNodes = 0
        self.generatedNodes = 0
        self.nodesInMemory = 0
        self.solution = []
        self.diccAct = {}
        self.calc_heurist = False

        #Aplicamos una heurística u otra en función del String que nos viene por parámetro
        if heuristica == 'euclidea':
            self.heuristica = EuclideanHeuristic(self.problem)
            self.calc_heurist = True
        elif heuristica == 'optimista':
            self.heuristica = OptimisticHeuristic(self.problem)
            self.calc_heurist = True
        elif heuristica == '':
            self.heuristica = None
        else:
            print('Heuristica no Implementada')
        
            

    #Esta función nos devuelve los sucesores de un Nodo dado, para ello necesitamos las acciones que se pueden tomar desde
    #ese nodo, poner ese nodo como nodo padre y aplicar cada una de las acciones.
    def get_successors(self, node):
        succ = []
        nodeFa = []
        actions = self.problem.getActionsNode(node)

        for i in range(len(actions)):
            nextAction = Action(actions[i]['origin'],actions[i]['destination'],actions[i]['distance'])
            #Se emplea copy.deepcopy porque de no ser así el nodo almacenado en nodeFa cambiará al aplicar las acciones abajo
            #Lo mismo pasa con nodeState, tenemos que usar copy.deepcopy para que mantenga el valor inicial y no cambie
            nodeFa.append(copy.deepcopy(node))
            nodeState = copy.deepcopy(node.state)
            succ.append(Node(nodeState.apply_action(nextAction), nodeFa[i], nextAction))
        return succ

    
    def doSearch(self):
        #Sacamos el nodo inicial y lo insertamos en la lista de abiertos y sumamos 1 a los nodos generados
        initNode = Node(self.problem.getInitialState(), None, None)
        self.insert_node(initNode, self.open)
        self.generatedNodes += 1
        
    
        tiempoInicio = time.perf_counter()
        
        #Inicializamos una variable booleana que será la condición de nuestro bucle, cuando se alcance la meta se pondrá 
        #a True y saldremos del bucle
        finish = False
        while not finish:
            #Si no tenemos nodo inicial la ejecución se detiene
            if self.is_empty(self.open):
                return NotImplemented
                
            
            #Dependiendo del tipo de lista que estemos empleando (array o Priority Queue), tenemos dos formas de tratar
            #con ellas. Aquí sacamos en número de máximo de nodos almacenados en la lista de abiertos
            if isinstance(self.open,list):
                if len(self.open) >= self.nodesInMemory:
                    self.nodesInMemory = len(self.open)
            else:
                if self.open.qsize() >= self.nodesInMemory:
                    self.nodesInMemory = self.open.qsize()
            
            
            #--------------CurrentNode---------------
            #Sacamos el primer nodo de la lista de abiertos con el que trabajar en cada iteración
            currentNode = self.extract_node(self.open)


            #Si estamos en un modo solución, salimos del bucle poniendo finish = True
            if finish != self.problem.isGoal(currentNode.state):
                finish = True
                break

            #Si el nodo no ha sido expandido, lo expandimos (generando sus sucesores e introduciendolo en la lista de cerrados)
            if currentNode.state not in self.closed:
                    self.closed.add(currentNode.state)
                    suc = self.get_successors(currentNode)
                    self.expandedNodes += 1

                    #Para cada uno de los sucesores calculamos sus costes (g, h, f) y los insertamos en la lista de abiertos
                    for i in suc:
                        i.gCost = i.parent.gCost + i.action.actioncost
                        if self.calc_heurist:
                            i.hCost = self.heuristica.get_hcost(i)
                            i.fCost = i.gCost + i.hCost
                        self.insert_node(i, self.open)
                        self.generatedNodes += 1
                    
        #Fuera ya del bucle reconstruirmos la solución a partir del nodo final
        
        while currentNode.parent is not None:
            self.solution.append(currentNode.action)
            self.cost += currentNode.action.actioncost
            currentNode = currentNode.parent
        self.solution.reverse()

        #Calculamos el tiempo de ejecución
        #tiempoFinal = time.perf_counter()
        #tiempoTotal = round((tiempoFinal - tiempoInicio)*1000,4)
        
        #Dibujamos el mapa
        #self.problem.plot_map(self.solution)
       
        # print(
        # '''
        # ==============
        #  ESTADÍSTICAS
        # ==============
        # ''')
        # print("Tiempo de ejecución: " + str(tiempoTotal) + " milisegundos \n")
        # print("Nodos Expandidos: ", self.expandedNodes)
        # print("Nodos Generados: ", self.generatedNodes)
        # print("Número máximo de nodos almacenados simultáneamente en la memoria: ", self.nodesInMemory + len(self.closed))
        # print("Secuencia de acciones para la solución obtenida:")
        # for i in self.solution:
        #     print(i)
        #print("Con un coste de: " + str(self.cost) + " km \n")
        return self.cost
        
    
                    

# #### Uninformed Search: `DepthFirst` and `BreadthFirst`
# 
# These two classes also inherit from `Search` and will implement the depth first and breadth first. As explained before, if you have implemented `get_successors(self, node)` and `do_search(self)` properly, you just have to implement the `insert_node(self, node, node_list)`, `extract_node` and `is_empty` functions. 


class DepthFirst(Search):
    #Al ser profundidad, insertamos por la izquierda en la lista de abiertos
    def insert_node(self, node, node_list):
        node_list.insert(0, node)

    def extract_node(self, node_list):
        nodeextracted = node_list[0]
        node_list.pop(0)
        return nodeextracted

    def is_empty(self, node_list):
        return node_list == []

class BreadthFirst(Search):
    #Al ser anchura, insertamos por la derecha en la lista de abiertos
    def insert_node(self, node, node_list):
        node_list.append(node)

    def extract_node(self, node_list):
        nodeextracted = node_list[0]
        node_list.pop(0)
        return nodeextracted

    def is_empty(self, node_list):
        return node_list == []

# #### Informed Search: `BestFirst` and `AStar`
# 
# These two classes also inherit from `Search` and will implement the best first and $A^*$ search strategies, respectively. 
# 
# The main difference between these three algorithms is the way in which the cost function for a specific node ($f(n) = g(n) + h(n)$) is computed. Assuming that $g(n)$ is the real accumulated cost from the **initial state** to `n.state` and that $h(n)$ is the heuristic cost from `n.state` state to the **goal state**, $f(n)$ is computed as:
# 
# - Best First: $f(n) = h(n)$
# - A$^*$: $f(n) = g(n) + h(n)$
# 
# As before, once the `get_successors(self,node)` and `do_search(self)` methods have been implemented in the parent class, we have to implement the `insert_node(self, node)` method, which will insert the `node` into the `self.open` list of nodes according to the corresponding values of the cost function, as well as the `extract_node` and `is_empty` methods.
# 
# You also have to implement a new `__init__(self, args)` constructor so that you can expand the behavior of the informed search algorithms with a `Heuristic` and any other methods you need.
# 
# It is greatly encouraged that you use the [Priority Queue](https://docs.python.org/3/library/queue.html#queue.PriorityQueue) structure for the informed search, as it will be an efficient structure to have your nodes ordered, rather than having to sort the list every single time. 


class BestFirst(Search):
    def __init__(self, parent_args, children_args):
        # Calling the constructor of the parent class
        # with its corresponding arguments
        super().__init__(parent_args, children_args)
        #Superponemos la declaración de la lista de abiertos como array en la clase Search para trabajar con PriorityQueue
        self.open = PriorityQueue()

        #TODO: Add your new code here
    #Al ser BestFirst insertamos en función del hCost
    def insert_node(self, node, node_list):
        node_list.put((node.hCost, node))
        
    def extract_node(self, node_list):
        return node_list.get()[1]

    def is_empty(self, node_list):
        return not node_list.not_empty

class AStar(Search):
    def __init__(self, parent_args, children_args):
        # Calling the constructor of the parent class
        # with its corresponding arguments
        super().__init__(parent_args, children_args)
        self.open = PriorityQueue()

        #TODO: Add your new code here
    #Al ser AStar insertamos en función del fCost (gCost + hCost)
    def insert_node(self, node, node_list):
        node_list.put((node.fCost, node))
        
    def extract_node(self, node_list):
        return node_list.get()[1]

    def is_empty(self, node_list):
        return not node_list.not_empty

# #### Heuristics
# 
# An informed search must have an heuristic, and the way to implement is by creating a class for each heuristic. The different classes must inherit from the abstract class `Heuristic` provided here. They must implement the `get_hcost(self, node)` method to return the heuristic of a node. They can also implement a constructor where some information about the problem is given to compute that heuristic.


class Heuristic(ABC):   
    @abstractmethod
    def get_hcost(self, node):
        pass

# As an example, the optimistic heuristic is given below. Take into account that you can add information to your heuristic by adding elements in the constructor of the class.


class OptimisticHeuristic(Heuristic):
    def __init__(self, info):
        self.info = info

    def get_hcost(self, node):
        return 0

class EuclideanHeuristic(Heuristic):
    def __init__(self, prob):
        self.prob = prob

    #Obtenemos hCost a partir de la función vista arriba del todo, insertando latitudes y longitudes de la ciudad respecto a la 
    #ciudad final
    def get_hcost(self, node):
        latitude1 = self.prob.cities[node.state.id]['lat']
        latitude2 = self.prob.cities[self.prob.final_city.id]['lat']
        longitude1 = self.prob.cities[node.state.id]['lon']
        longitude2 = self.prob.cities[self.prob.final_city.id]['lon']
        point1,point2 = (latitude1,longitude1), (latitude2,longitude2)
        return geopy.distance.distance(point1,point2).km + random.uniform(0, 0.0000001)
        #return getDistanceBetweenPointskm(latitude1, longitude1, latitude2, longitude2) + random.uniform(0,0.000001)

# ### 4.3 Study and improvement of the algorithms
# Once the algorithms have been implemented, you must study their performance. In order to do that,
# you must compare the quality of the solutions obtained, as well as the number of expanded nodes for
# instances of different sizes. Factors as the maximum size of problem that can be solved (without memory
# overflow), or the effect of using more complex scenarios, are also important. Moreover, you can propose
# alternative implementations that increase the efficiency of the algorithms.


###### CELDA DE EJECUCIÓN ######

#Precargamos problem con el que trabaja la clase Problem cambiando el .json que queremos emplear
#with open("/data/notebook_files/problem.json", 'r', encoding='utf8') as file:
#    problem = json.load(file)

#Creamos una instancia de Problem con el .json a utilizar, el Estado (ciudad) Inicial y el Estado (Ciudad) Final
#prob = Problem("/data/notebook_files/problem.json")

#Ponemos en marcha la búsqueda insertando el algoritmo que queremos emplear (DepthFirst, BreadthFirst, BestFirst o AStar)
#e insertando la heurística a utilizar ('', 'euclidea' u 'optimista'), los dos primeros algoritmos no emplean heurística 
#search = BestFirst(prob, 'euclidea')
#search.doSearch()

# ### 4.4 Report
# Besides the notebook containing the implementation, the assignment consists in the elaboration of a report, which will have a later deadline, but you should be developing when your code starts solving problems
# correctly. 
# 
# In particular, among other issues that the student deems of interest to discuss, it should include:
# 
# - A brief description of the problem, a description of the implementation, the performance evaluation, and the description of improvements if they exist. 
# - The formalization of the problem.
# - For informed search algorithms one (or several) heuristic functions must be provided. Apart from their description and motivation, an analysis should be included indicating whether the proposed heuristic is considered admissible and/or consistent.
# - The study of performance of implemented algorithms should be based on testing the algorithms over several instances, presenting tables or graphics that summarize results (do not include screenshots).
# 
# The memory must not include figures with source code, unless this is necessary to explain some key concept (data structures, improvements in efficiency, etc). In such cases, you are allowed to include
# properly formatted pseudocode.


# ## 5. Submission and evaluation
# The work must be made in pairs, although in some exceptional cases you can present it individually. The deadline for submission is 6th November, 2022. Interviews and evaluations will be in the following week. 
# 
# You must work on your notebook on the Datalore project, as the day of the deadline it will run some automated tests and collect the notebooks in the state they are. No changes will be allowed after the deadline. 
# 
# Some considerations related to the evaluation:
# - This is 30% of the lab grade. Lab2 (70%) needs the resolution of this part. Late submissions
# (together with lab2) or failed assignments will be evaluated globally but with a penalization of factor
# 0.9, as the students can only get 90% of the lab grade.
# - Attendance with performance to the lab not only will provide half of the participation grade, but
# it will also be the best foundation for successful resolution of the labs.
# - The assignment will be evaluated during an individual interview with the professors. Dates for the
# interviews will be published with some advance in Campus Virtual.
# - We will provide a set of preliminary test cases (several maps and queries) that must be correctly
# solved. Otherwise, the assignment will be considered as failed.
# - In order to get a mark in the assignment you will have to answer, individually, a set of basic
# questions about the code organization.
# - In the non-continuous evaluation we will require the implementation of the same strategies plus
# these extra two: Depth-limited search and Iterative deepening search.




