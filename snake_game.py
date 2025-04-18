import pygame
import random
import sys
from enum import Enum
from typing import List, Tuple

# Initialisierung
pygame.init()

# Konstanten
WINDOW_SIZE = 800
GRID_SIZE = 20
GRID_COUNT = WINDOW_SIZE // GRID_SIZE

# Farben
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)

# Richtungen
class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.GRID_SIZE = GRID_SIZE
        self.GRID_COUNT = GRID_COUNT
        self.reset_game()
        self.game_state = "MENU"  # MENU, PLAYING, PAUSED, GAME_OVER
        self.font = pygame.font.Font(None, 36)

    def reset_game(self):
        self.snake = [(GRID_COUNT // 2, GRID_COUNT // 2)]
        self.direction = Direction.RIGHT
        self.food = self.spawn_food()
        self.score = 0
        self.game_over = False

    def spawn_food(self) -> Tuple[int, int]:
        while True:
            food = (random.randint(0, GRID_COUNT-1), random.randint(0, GRID_COUNT-1))
            if food not in self.snake:
                return food

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if self.game_state == "MENU":
                    if event.key == pygame.K_SPACE:
                        self.game_state = "PLAYING"
                elif self.game_state == "PLAYING":
                    if event.key == pygame.K_UP and self.direction != Direction.DOWN:
                        self.direction = Direction.UP
                    elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                        self.direction = Direction.DOWN
                    elif event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                        self.direction = Direction.LEFT
                    elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                        self.direction = Direction.RIGHT
                    elif event.key == pygame.K_p:
                        self.game_state = "PAUSED"
                elif self.game_state == "PAUSED":
                    if event.key == pygame.K_p:
                        self.game_state = "PLAYING"
                elif self.game_state == "GAME_OVER":
                    if event.key == pygame.K_r:
                        self.reset_game()
                        self.game_state = "PLAYING"
                    elif event.key == pygame.K_m:
                        self.reset_game()
                        self.game_state = "MENU"

    def update(self):
        if self.game_state != "PLAYING":
            return

        # Bewegung der Schlange
        x, y = self.snake[0]
        if self.direction == Direction.UP:
            y -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.RIGHT:
            x += 1

        # Kollisionsprüfung
        if (x < 0 or x >= GRID_COUNT or y < 0 or y >= GRID_COUNT or
            (x, y) in self.snake):
            self.game_state = "GAME_OVER"
            return

        # Neue Kopfposition
        self.snake.insert(0, (x, y))

        # Prüfen, ob Essen gefunden wurde
        if (x, y) == self.food:
            self.score += 1
            self.food = self.spawn_food()
        else:
            self.snake.pop()

    def draw(self):
        self.screen.fill(BLACK)

        if self.game_state == "MENU":
            self.draw_menu()
        elif self.game_state in ["PLAYING", "PAUSED"]:
            # Zeichne Schlange
            for segment in self.snake:
                pygame.draw.rect(self.screen, GREEN,
                               (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE,
                                GRID_SIZE - 2, GRID_SIZE - 2))

            # Zeichne Essen
            pygame.draw.rect(self.screen, RED,
                           (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE,
                            GRID_SIZE - 2, GRID_SIZE - 2))

            # Zeichne Punktestand
            score_text = self.font.render(f'Score: {self.score}', True, WHITE)
            self.screen.blit(score_text, (10, 10))

            if self.game_state == "PAUSED":
                self.draw_pause_screen()

        elif self.game_state == "GAME_OVER":
            self.draw_game_over()

        pygame.display.flip()

    def draw_menu(self):
        title = self.font.render('SNAKE', True, GREEN)
        start_text = self.font.render('Drücke LEERTASTE zum Starten', True, WHITE)
        
        title_rect = title.get_rect(center=(WINDOW_SIZE/2, WINDOW_SIZE/3))
        start_rect = start_text.get_rect(center=(WINDOW_SIZE/2, WINDOW_SIZE/2))
        
        self.screen.blit(title, title_rect)
        self.screen.blit(start_text, start_rect)

    def draw_pause_screen(self):
        s = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        s.set_alpha(128)
        s.fill(BLACK)
        self.screen.blit(s, (0, 0))
        
        pause_text = self.font.render('PAUSE', True, WHITE)
        continue_text = self.font.render('Drücke P zum Fortfahren', True, WHITE)
        
        pause_rect = pause_text.get_rect(center=(WINDOW_SIZE/2, WINDOW_SIZE/2 - 50))
        continue_rect = continue_text.get_rect(center=(WINDOW_SIZE/2, WINDOW_SIZE/2 + 50))
        
        self.screen.blit(pause_text, pause_rect)
        self.screen.blit(continue_text, continue_rect)

    def draw_game_over(self):
        game_over_text = self.font.render('GAME OVER', True, RED)
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        restart_text = self.font.render('Drücke R zum Neustarten', True, WHITE)
        menu_text = self.font.render('Drücke M für Menü', True, WHITE)
        
        game_over_rect = game_over_text.get_rect(center=(WINDOW_SIZE/2, WINDOW_SIZE/3))
        score_rect = score_text.get_rect(center=(WINDOW_SIZE/2, WINDOW_SIZE/2))
        restart_rect = restart_text.get_rect(center=(WINDOW_SIZE/2, WINDOW_SIZE/2 + 50))
        menu_rect = menu_text.get_rect(center=(WINDOW_SIZE/2, WINDOW_SIZE/2 + 100))
        
        self.screen.blit(game_over_text, game_over_rect)
        self.screen.blit(score_text, score_rect)
        self.screen.blit(restart_text, restart_rect)
        self.screen.blit(menu_text, menu_rect)

    def run(self):
        while True:
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(10)

if __name__ == "__main__":
    game = SnakeGame()
    game.run() 