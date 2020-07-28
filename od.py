class OD:

   def __init__(self, origin, destination, demand = 0):
      self.origin = origin
      self.destination = destination
      self.demand = demand
      self.cost = None
      self.pi_rs = []
