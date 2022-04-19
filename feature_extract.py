import math
class ExtractFeature:
    x = []
    y = []
    def __init__(self, xlist, ylist):
        if(len(xlist) == 0 or len(ylist) == 0): return
        self.x = xlist
        self.y = ylist

    def get_angle(self, p1, p2, p3):
        p1 -= 1
        p2 -= 1
        p3 -= 1
        x = self.x
        y = self.y
        if(len(x) == 0 or len(y) == 0): return
        beta = math.atan( (y[p3] - y[p2]) / (x[p3] - x[p2]) )
        alpha = math.atan( (y[p1] - y[p2]) / (x[p1] - x[p2]) )
        return beta - alpha

    def get_distance(self, p1, p2):
        p1 -= 1
        p2 -= 1
        distance =  math.sqrt(math.pow(self.x[p2] - self.x[p1], 2) + math.pow(self.y[p2] - self.y[p1], 2))
        return abs(distance)


    


