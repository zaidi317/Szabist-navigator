"""
A* pathfinding algorithm for campus navigation
"""

import heapq
from typing import Dict, List, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coordinates derived from campus map pixel positions (scale ≈ 0.3 m/px).
# X = east (metres), Y = south (metres), floor 0 = ground, floor 1 = first.
# ---------------------------------------------------------------------------
NODE_COORDS: Dict[str, Dict] = {
    "FountainArea":     {"x":  56, "y":  50, "floor": 0},
    "LostAndFound":     {"x": 101, "y":  29, "floor": 0},
    "helmetarea":       {"x":  24, "y":  75, "floor": 0},
    "Cafe":             {"x":  27, "y": 150, "floor": 0},
    "SSC":              {"x":  21, "y": 228, "floor": 1},
    "100Entrance":      {"x":  77, "y": 156, "floor": 0},
    "ReceptionBackside":{"x": 110, "y": 102, "floor": 0},
    "Courtyard":        {"x": 122, "y": 168, "floor": 0},
    "stairs":           {"x":  93, "y": 228, "floor": 0},
    "PattiArea":        {"x":  72, "y": 252, "floor": 0},
    "Mosque":           {"x": 137, "y": 258, "floor": 0},
    "FirstAidRoom":     {"x":  18, "y": 235, "floor": 1},
}

NODE_LABELS: Dict[str, str] = {
    "FountainArea":     "Fountain Area",
    "LostAndFound":     "Lost & Found",
    "helmetarea":       "Helmet Area",
    "Cafe":             "Canteen",
    "SSC":              "SSC Room",
    "100Entrance":      "Classrooms",
    "ReceptionBackside":"Reception (Backyard)",
    "Courtyard":        "Courtyard",
    "stairs":           "Stairs",
    "PattiArea":        "Stationary Shop",
    "Mosque":           "Mosque",
    "FirstAidRoom":     "First Aid Room",
}

# Penalty added to heuristic per floor change (metres equivalent)
FLOOR_CHANGE_PENALTY = 5.0

class CampusGraph:
    """
    Represents the campus as a graph for pathfinding
    """
    
    def __init__(self):
        """
        Initialize campus graph with predefined locations and connections
        
        Graph structure: {location: [(neighbor, distance), ...]}
        Distances are approximate meters between locations
        """
        # Distances computed from map pixel coordinates (scale ≈ 0.3 m/px), rounded to 5 m.
        self.graph = {
            "FountainArea":     [("helmetarea", 40),  ("LostAndFound", 50)],
            "LostAndFound":     [("FountainArea", 50), ("ReceptionBackside", 75)],
            "helmetarea":       [("FountainArea", 40), ("Cafe", 75)],
            "Cafe":             [("helmetarea", 75),  ("SSC", 80),  ("100Entrance", 50)],
            "SSC":              [("Cafe", 80),         ("PattiArea", 55), ("FirstAidRoom", 20)],
            "100Entrance":      [("Cafe", 50),         ("ReceptionBackside", 65), ("Courtyard", 45), ("stairs", 75)],
            "ReceptionBackside":[("LostAndFound", 75), ("100Entrance", 65), ("Courtyard", 65)],
            "Courtyard":        [("ReceptionBackside", 65), ("100Entrance", 45), ("Mosque", 90)],
            "stairs":           [("100Entrance", 75), ("PattiArea", 30)],
            "PattiArea":        [("SSC", 55),          ("stairs", 30),  ("Mosque", 65)],
            "Mosque":           [("Courtyard", 90),    ("PattiArea", 65)],
            "FirstAidRoom":     [("SSC", 20)],
        }

        # Build (x, y) coordinate map from NODE_COORDS for the heuristic
        self.coordinates = {
            node_id: (c["x"], c["y"]) for node_id, c in NODE_COORDS.items()
        }

        logger.info(f"Initialized campus graph with {len(self.graph)} locations")
    
    def add_edge(self, from_loc: str, to_loc: str, distance: float):
        """
        Add bidirectional edge between two locations
        
        Args:
            from_loc: Starting location
            to_loc: Destination location
            distance: Distance between locations in meters
        """
        if from_loc not in self.graph:
            self.graph[from_loc] = []
        if to_loc not in self.graph:
            self.graph[to_loc] = []
        
        self.graph[from_loc].append((to_loc, distance))
        self.graph[to_loc].append((from_loc, distance))
        logger.info(f"Added edge: {from_loc} <-> {to_loc} ({distance}m)")
    
    def set_coordinates(self, location: str, x: float, y: float):
        """
        Set coordinates for a location (for heuristic calculation)
        
        Args:
            location: Location name
            x: X coordinate
            y: Y coordinate
        """
        self.coordinates[location] = (x, y)
    
    def _euclidean_distance(self, loc1: str, loc2: str) -> float:
        """Euclidean XY distance plus a flat penalty per floor change."""
        if loc1 not in self.coordinates or loc2 not in self.coordinates:
            return 0
        x1, y1 = self.coordinates[loc1]
        x2, y2 = self.coordinates[loc2]
        xy_dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        floor1 = NODE_COORDS.get(loc1, {}).get("floor", 0)
        floor2 = NODE_COORDS.get(loc2, {}).get("floor", 0)
        return xy_dist + FLOOR_CHANGE_PENALTY * abs(floor2 - floor1)

    def get_node_info(self, node_id: str) -> Dict:
        """Return full info dict for a node: id, label, x, y, floor, neighbors."""
        coords = NODE_COORDS.get(node_id, {"x": 0.0, "y": 0.0, "floor": 0})
        return {
            "node_id":   node_id,
            "label":     NODE_LABELS.get(node_id, node_id),
            "x":         coords["x"],
            "y":         coords["y"],
            "floor":     coords["floor"],
            "neighbors": [n for n, _ in self.graph.get(node_id, [])],
        }
    
    def get_neighbors(self, location: str) -> List[Tuple[str, float]]:
        """Get neighbors of a location"""
        return self.graph.get(location, [])
    
    def is_valid_location(self, location: str) -> bool:
        """Check if location exists in graph"""
        return location in self.graph


class AStarPathfinder:
    """
    A* pathfinding algorithm implementation
    """
    
    def __init__(self, graph: CampusGraph):
        """
        Initialize pathfinder
        
        Args:
            graph: CampusGraph object
        """
        self.graph = graph
    
    def find_path(self, start: str, end: str, heuristic_weight: float = 1.0) -> Optional[List[str]]:
        """
        Find optimal path from start to end location using A* algorithm
        
        Args:
            start: Starting location
            end: Destination location
            heuristic_weight: Weight for heuristic (higher = more aggressive)
            
        Returns:
            List of locations representing the path, or None if no path exists
        """
        # Validate locations
        if not self.graph.is_valid_location(start):
            logger.error(f"Invalid start location: {start}")
            return None
        
        if not self.graph.is_valid_location(end):
            logger.error(f"Invalid end location: {end}")
            return None
        
        if start == end:
            logger.info(f"Start and end are the same: {start}")
            return [start]
        
        # Priority queue: (f_score, counter, current_location)
        counter = 0
        open_set = [(0, counter, start)]
        
        # Track visited nodes
        came_from = {}
        
        # g_score: actual cost from start
        g_score = {location: float('inf') for location in self.graph.graph}
        g_score[start] = 0
        
        # f_score: g + h (estimated total cost)
        f_score = {location: float('inf') for location in self.graph.graph}
        f_score[start] = heuristic_weight * self._heuristic(start, end)
        
        # Track open set items
        open_set_hash = {start}
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                logger.info(f"Found path from {start} to {end}: {' -> '.join(path)}")
                return path
            
            open_set_hash.discard(current)
            
            # Check all neighbors
            for neighbor, edge_cost in self.graph.get_neighbors(current):
                tentative_g = g_score[current] + edge_cost
                
                # Found a better path
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic_weight * self._heuristic(neighbor, end)
                    
                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                        open_set_hash.add(neighbor)
        
        logger.warning(f"No path found from {start} to {end}")
        return None
    
    def _heuristic(self, current: str, goal: str) -> float:
        """Heuristic function for A*"""
        return self.graph._euclidean_distance(current, goal)
    
    def find_alternative_paths(self, start: str, end: str, num_paths: int = 3) -> List[List[str]]:
        """
        Find multiple alternative paths (simple implementation)
        
        Args:
            start: Starting location
            end: Destination location
            num_paths: Number of alternative paths to find
            
        Returns:
            List of paths
        """
        paths = []
        path = self.find_path(start, end)
        
        if path:
            paths.append(path)
        
        logger.info(f"Found {len(paths)} alternative path(s)")
        return paths
    
    def estimate_travel_time(self, path: List[str], speed_mps: float = 1.5) -> float:
        """
        Estimate travel time for a path
        
        Args:
            path: List of locations
            speed_mps: Walking speed in meters per second (default: 1.5 m/s)
            
        Returns:
            Estimated time in seconds
        """
        if not path or len(path) < 2:
            return 0
        
        total_distance = 0
        for i in range(len(path) - 1):
            current = path[i]
            next_loc = path[i + 1]
            
            # Find edge distance
            neighbors = self.graph.get_neighbors(current)
            for neighbor, distance in neighbors:
                if neighbor == next_loc:
                    total_distance += distance
                    break
        
        travel_time = total_distance / speed_mps
        return travel_time
