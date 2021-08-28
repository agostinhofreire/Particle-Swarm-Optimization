from utils.utils import *
import cv2
import imutils
import random

class PSO:

    def __init__(self, n_particles=20, iterations=200, no_improvement=40, space_h=300, space_w=300, w=0.5, c1=0.8, c2=0.9, custom_space=""):

        self.n_particles = n_particles
        self.iterations = iterations
        self.no_improvement = no_improvement
        self.no_improvement_counter = 0

        self.space_h, self.space_w = space_h, space_w

        self.particles = []

        self.space, self.min, self.locations = create_space(space_h, space_w, custom_space)
        self.reset_view = np.ones((space_h, space_w))
        self.running_space = np.ones((space_h, space_w))

        self.g_best = {"id": -1, "value": np.inf, "pos": -1}

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.init_particles()

    def init_particles(self):

        for i in range(self.n_particles):
            pos = np.array([random.randint(0, self.space.shape[0]-1), random.randint(0, self.space.shape[1]-1)])

            particle = {
                "id": i,
                "pos": pos,
                "vel": np.array([0, 0]),
                "p_best_value": np.inf,
                "p_best_pos": pos

            }

            self.running_space[tuple(pos)] = particle["p_best_value"]
            self.particles.append(particle)


    def run(self):

        for i in range(self.iterations):

            self.evaluate_particles()

            self.particles_dynamics()

            g_best_value = self.g_best["value"]
            g_best_pos = self.g_best["pos"]

            if self.no_improvement_counter > self.no_improvement:
                break

            print(f"[INFO] Iteration {i+1} of {self.iterations} - G_BEST: {g_best_value} at {g_best_pos}\n")

            cv2.imshow("space", imutils.resize(self.space, width=600))
            cv2.imshow("particles", imutils.resize(self.running_space, width=600))
            cv2.waitKey(1)
            self.running_space = self.reset_view.copy()

            self.no_improvement_counter += 1

        g_best_value = self.g_best["value"]
        g_best_pos = self.g_best["pos"]
        print("[INFO] Stopping fitting...")
        print(f"[INFO] G_BEST: {g_best_value} at {g_best_pos}\n")


    def particles_dynamics(self):

        for particle in self.particles:

            personal = (self.c1 * random.random()) * (particle["p_best_pos"] - particle["pos"])
            social = (self.c2 * random.random()) * (self.g_best["pos"] - particle["pos"])
            vel = (self.w * particle["vel"]) + personal + social
            vel = vel.astype("int64")

            particle["vel"] = vel

            particle["pos"] = self.add_velocity(particle["pos"], vel)

    def evaluate_particles(self):
        for particle in self.particles:

            pos = particle["pos"]
            fitness = self.space[tuple(pos)]
            self.running_space[tuple(pos)] = self.space[tuple(particle["pos"])]

            if fitness < particle["p_best_value"]:
                particle["p_best"] = fitness

            if fitness < self.g_best["value"]:
                self.no_improvement_counter = 0
                self.g_best["id"] = particle["id"]
                self.g_best["value"] = fitness
                self.g_best["pos"] = pos

    def add_velocity(self, pos, vel):
        new_pos = pos + vel

        new_pos[0] = min(new_pos[0], self.space_h - 1)
        new_pos[0] = max(new_pos[0], 0)

        new_pos[1] = min(new_pos[1], self.space_w - 1)
        new_pos[1] = max(new_pos[1], 0)

        return new_pos

if __name__ == '__main__':
    pso = PSO(n_particles=200, iterations=1000, no_improvement=100, space_h=20, space_w=20, w=0.5, c1=0.8, c2=0.9, custom_space="custom_space.png")
    pso.run()
