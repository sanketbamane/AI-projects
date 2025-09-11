# utility functions for route cost/time estimation
def travel_time_seconds(distance_m, avg_speed_kmph=40):
    # convert meters to hours: distance_m / 1000 / speed_kmph
    hours = (distance_m/1000.0) / avg_speed_kmph
    return int(hours*3600)