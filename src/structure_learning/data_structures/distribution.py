
class Distribution:
    
    def __init__(self, particles, logp, theta=None):
        self.particles = particles
        self.logp = logp
        self.theta = theta

    def visualise(self):
        pass