import numpy as np;
import matplotlib.pyplot as plt;

class Rocket():
    # constructor
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # Method that allows us to move up our rocket
    def move_rocket(self, x_increment=0, y_increment=0):
        self.x += x_increment
        self.y += y_increment 

    # Calculate euclidian distance between two rockets
    def get_distance_to_another_rocket(self, other_rocket):
        x_diff = self.x - other_rocket.x
        y_diff = self.y - other_rocket.y
        distance = (x_diff ** 2 + y_diff ** 2) ** 0.5
        return round(distance) 

rocket1 = Rocket(0, 0)
rocket2 = Rocket(5, 5)
print(rocket1.get_distance_to_another_rocket(rocket2))
# for i in range(0, 100): rocket1.move_rocket(0, 1)
# print(f"Rocket 1 is at the {rocket1.y} height and {rocket1.x} length")