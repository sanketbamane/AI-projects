from typing import Dict, Any, List
import math, logging
logger = logging.getLogger("drom.optimizer")
from planner.vehicle_routing import travel_time_seconds

def haversine(a, b):
    R = 6371e3
    lat1 = math.radians(a[0]); lat2 = math.radians(b[0])
    dlat = lat2 - lat1; dlon = math.radians(b[1]-a[1])
    x = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(x), math.sqrt(1-x))
    return R * c

class RouteOptimizer:
    def __init__(self):
        try:
            from ortools.constraint_solver import routing_enums_pb2
            from ortools.constraint_solver import pywrapcp
            self.has_or_tools = True
            self.routing_enums_pb2 = routing_enums_pb2
            self.pywrapcp = pywrapcp
        except Exception:
            logger.warning("OR-Tools not available, using greedy fallback.")
            self.has_or_tools = False

    def _build_matrix(self, locations: List[Dict[str, Any]]):
        n = len(locations)
        mat = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i==j: mat[i][j]=0
                else:
                    a=(locations[i]['lat'], locations[i]['lon'])
                    b=(locations[j]['lat'], locations[j]['lon'])
                    mat[i][j]=int(haversine(a,b))
        return mat

    def optimize(self, request: Dict[str, Any]) -> Dict[str, Any]:
        locations = request['locations']
        vehicles = request['vehicles']
        depot = request.get('depot_index', 0)
        matrix = self._build_matrix(locations)
        if self.has_or_tools:
            return self._optimize_ortools(matrix, locations, vehicles, depot)
        else:
            return self._optimize_greedy(matrix, locations, vehicles, depot)

    def _optimize_ortools(self, matrix, locations, vehicles, depot):
        manager = self.pywrapcp.RoutingIndexManager(len(matrix), len(vehicles), depot)
        routing = self.pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            fn = manager.IndexToNode(from_index)
            tn = manager.IndexToNode(to_index)
            return matrix[fn][tn]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        demands = [loc.get('demand', 0) for loc in locations]
        if any(demands):
            def demand_cb(from_index):
                return demands[manager.IndexToNode(from_index)]
            demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
            routing.AddDimensionWithVehicleCapacity(
                demand_idx, 0, [v.get('capacity',100) for v in vehicles], True, "Capacity")

        # time windows (if present)
        # Here we create a simplistic time dimension based on distance->time conversion
        time = "Time"
        def time_callback(from_index, to_index):
            fn = manager.IndexToNode(from_index)
            tn = manager.IndexToNode(to_index)
            return int(travel_time_seconds(matrix[fn][tn]))
        time_idx = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(time_idx, 300, 24*3600, False, time)

        search_params = self.pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = self.routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.time_limit.seconds = 15

        solution = routing.SolveWithParameters(search_params)
        if solution:
            routes=[]
            for v_idx in range(len(vehicles)):
                index = routing.Start(v_idx)
                route=[]
                route_distance=0
                while not routing.IsEnd(index):
                    node = manager.IndexToNode(index)
                    route.append(int(node))
                    prev=index
                    index = solution.Value(routing.NextVar(index))
                    route_distance += routing.GetArcCostForVehicle(prev, index, v_idx)
                route.append(manager.IndexToNode(index))
                routes.append({"vehicle_id": vehicles[v_idx]['id'], "route": route, "distance_m": int(route_distance)})
            return {"routes": routes}
        else:
            logger.warning("OR-Tools failed, fallback.")
            return self._optimize_greedy(matrix, locations, vehicles, depot)

    def _optimize_greedy(self, matrix, locations, vehicles, depot):
        n=len(matrix)
        unassigned=set(range(n))
        routes=[]
        for v in vehicles:
            cap = v.get('capacity',100)
            cur = depot
            route=[cur]
            load=0
            if cur in unassigned:
                unassigned.discard(cur)
            while True:
                candidates = [(matrix[cur][j], j) for j in unassigned if j!=cur]
                if not candidates: break
                candidates.sort()
                dist, nxt = candidates[0]
                demand = locations[nxt].get('demand',0)
                if load + demand > cap:
                    break
                route.append(nxt)
                load += demand
                unassigned.discard(nxt)
                cur = nxt
            route.append(depot)
            routes.append({"vehicle_id": v['id'], "route": route, "distance_m": sum(matrix[route[i]][route[i+1]] for i in range(len(route)-1))})
        if unassigned:
            for idx in list(unassigned):
                routes[0]['route'].insert(-1, idx)
                routes[0]['distance_m'] = sum(matrix[routes[0]['route'][i]][routes[0]['route'][i+1]] for i in range(len(routes[0]['route'])-1))
        return {"routes": routes}