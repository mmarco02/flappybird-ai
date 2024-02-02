import pygame
import sys
import random

class FlappyBird:
    def __init__(self):
        pygame.init()

        self.screen_width = 600
        self.screen_height = 400
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Flappy Bird Clone")


        self.clock = pygame.time.Clock()

        self.bird_size = 30
        self.bird_color = (255, 255, 0)
        self.bird_x = 100
        self.bird_y = self.screen_height // 2 - self.bird_size // 2
        self.bird_velocity = 0
        self.jump_strength = -10

        self.pipe_width = 50
        self.pipe_color = (0, 255, 0)
        self.pipes = []

        self.pipe_gap = 150
        self.pipe_speed = 5
        self.pipe_interval = 65
        self.pipe_timer = 0

        self.score = 0
        self.font = pygame.font.Font(None, 36)
        self.distance = 0
        self.game_over = None

    def get_distance_to_next_pipe(self):
        if not self.pipes:
            return 0
        next_pipe = self.pipes[0]
        return next_pipe['x'] - self.bird_x

    def get_vertical_distance_to_next_bottom_pipe(self):
        if not self.pipes:
            return 0
        next_pipe = self.pipes[0]
        return next_pipe['height'] + self.pipe_gap - self.bird_y

    def get_vertical_distance_to_next_top_pipe(self):
        if not self.pipes:
            return 0
        next_pipe = self.pipes[0]
        return next_pipe['height'] - self.bird_y


    def get_bird_y(self):
        return self.bird_y

    def get_score(self):
        return self.score

    def get_distance(self):
        return self.distance

    def get_bird_velocity(self):
        return self.bird_velocity

    def get_collided(self):
        return self.game_over

    def reset(self):
        self.bird_y = self.screen_height // 2 - self.bird_size // 2
        self.bird_velocity = 0
        self.pipes = []
        self.pipe_timer = 0
        self.score = 0
        self.distance = 0
        self.game_over = False

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.jump()

            self.distance += 1
            self.update()
            self.draw()

            pygame.display.flip()
            self.clock.tick() #default: 30

    def render(self):
        self.distance += 1
        self.update()
        self.draw()

        pygame.display.flip()
        self.clock.tick() #default: 30

    def jump(self):
        self.bird_velocity = self.jump_strength

    def update(self):
        if not self.game_over:
            self.bird_velocity += 1
            self.bird_y += self.bird_velocity

            self.generate_pipes()
            self.move_pipes()

            if self.pipes:
                if self.pipes[0]['x'] + self.pipe_width < self.bird_x:
                    self.pipes.remove(self.pipes[0])

            self.game_over = self.check_collision()

    def generate_pipes(self):
        # Check if there are no pipes yet, then add the first pipe before the game loop starts
        if not self.pipes:
            pipe_height = random.randint(100, 300)
            self.pipes.append({
                'x': self.screen_width,
                'height': pipe_height,
            })

        self.pipe_timer += 1
        if self.pipe_timer == self.pipe_interval:
            pipe_height = random.randint(100, 300)
            self.pipes.append({
                'x': self.screen_width,
                'height': pipe_height,
            })
            self.pipe_timer = 0

        self.pipes = [pipe for pipe in self.pipes if pipe['x'] + self.pipe_width > 0]

    def move_pipes(self):
        for pipe in self.pipes:
            pipe['x'] -= self.pipe_speed

            if pipe['x'] + self.pipe_width < self.bird_x and pipe[
                'x'] + self.pipe_width + self.pipe_speed >= self.bird_x:
                self.score += 1

    def draw(self):
        self.screen.fill((0, 0, 0))

        for pipe in self.pipes:
            pygame.draw.rect(self.screen, self.pipe_color, (pipe['x'], 0, self.pipe_width, pipe['height']))
            pygame.draw.rect(self.screen, self.pipe_color, (pipe['x'], pipe['height'] + self.pipe_gap, self.pipe_width,
                                                            self.screen_height - pipe['height'] - self.pipe_gap))

        pygame.draw.rect(self.screen, self.bird_color, (self.bird_x, self.bird_y, self.bird_size, self.bird_size))

        self.draw_score()

    def draw_score(self):
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        distance_text = self.font.render(f"Distance: {self.distance}", True, (255, 255, 255))
        velocity_text = self.font.render(f"Velocity: {self.bird_velocity}", True, (255, 255, 255))
        next_pipe_text = self.font.render(f"Next Pipe: {self.get_distance_to_next_pipe()}", True, (255, 255, 255))
        next_bottom_pipe_text = self.font.render(
            f"Next Bottom Pipe: {self.get_vertical_distance_to_next_bottom_pipe()}",
            True, (255, 255, 255))
        next_top_pipe_text = self.font.render(f"Next Top Pipe: {self.get_vertical_distance_to_next_top_pipe()}",
                                              True, (255, 255, 255))
        bird_y_text = self.font.render(f"Bird Y: {self.bird_y}", True, (255, 255, 255))

        self.screen.blit(score_text, (10, 10))
        self.screen.blit(distance_text, (10, 30))
        self.screen.blit(bird_y_text, (10, 50))
        self.screen.blit(velocity_text, (10, 70))
        self.screen.blit(next_pipe_text, (10, 90))
        self.screen.blit(next_top_pipe_text, (10, 110))
        self.screen.blit(next_bottom_pipe_text, (10, 130))

    def check_collision(self):
        for pipe in self.pipes:
            if (
                    self.bird_x < pipe['x'] + self.pipe_width
                    and self.bird_x + self.bird_size > pipe['x']
                    and (self.bird_y < pipe['height'] or self.bird_y + self.bird_size > pipe['height'] + self.pipe_gap)
            ):
                return True

        if self.bird_y + self.bird_size > self.screen_height or self.bird_y < 0:
            return True

        return False


if __name__ == '__main__':
    game = FlappyBird()
    game.run()
