class Node:

   def __init__(self, isZone = False, reverseStar=list(), forwardStar=list()):
      self.forwardStar = forwardStar
      self.reverseStar = reverseStar
      self.isZone = isZone