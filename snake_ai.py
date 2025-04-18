import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import pickle
import os
from collections import deque
import random
import time
from snake_game import SnakeGame, Direction
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class SnakeAI:
    def __init__(self, input_size=12, hidden_size=256, output_size=3, 
                 learning_rate=0.001, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995, memory_size=100000,
                 batch_size=64, target_update=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural Network
        self.policy_net = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_net = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_size)
        
        # Hyperparameter
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Statistiken
        self.scores = []
        self.avg_scores = []
        self.epsilon_history = []
        self.loss_history = []
        self.best_score = 0
        self.best_episode = 0
        
        self.load_model()

    def get_state(self, game):
        head = game.snake[0]
        food = game.food
        
        # Danger straight ahead
        danger_straight = False
        danger_right = False
        danger_left = False
        
        # Check direction
        if game.direction == Direction.UP:
            point_l = (head[0] - 1, head[1])
            point_r = (head[0] + 1, head[1])
            point_u = (head[0], head[1] - 1)
            danger_straight = point_u in game.snake or point_u[1] < 0
            danger_right = point_r in game.snake or point_r[0] >= game.GRID_COUNT
            danger_left = point_l in game.snake or point_l[0] < 0
        elif game.direction == Direction.DOWN:
            point_l = (head[0] + 1, head[1])
            point_r = (head[0] - 1, head[1])
            point_u = (head[0], head[1] + 1)
            danger_straight = point_u in game.snake or point_u[1] >= game.GRID_COUNT
            danger_right = point_r in game.snake or point_r[0] < 0
            danger_left = point_l in game.snake or point_l[0] >= game.GRID_COUNT
        elif game.direction == Direction.LEFT:
            point_l = (head[0], head[1] + 1)
            point_r = (head[0], head[1] - 1)
            point_u = (head[0] - 1, head[1])
            danger_straight = point_u in game.snake or point_u[0] < 0
            danger_right = point_r in game.snake or point_r[1] < 0
            danger_left = point_l in game.snake or point_l[1] >= game.GRID_COUNT
        elif game.direction == Direction.RIGHT:
            point_l = (head[0], head[1] - 1)
            point_r = (head[0], head[1] + 1)
            point_u = (head[0] + 1, head[1])
            danger_straight = point_u in game.snake or point_u[0] >= game.GRID_COUNT
            danger_right = point_r in game.snake or point_r[1] >= game.GRID_COUNT
            danger_left = point_l in game.snake or point_l[1] < 0
        
        # State vector
        state = [
            # Danger
            danger_straight,
            danger_right,
            danger_left,
            
            # Direction
            game.direction == Direction.LEFT,
            game.direction == Direction.RIGHT,
            game.direction == Direction.UP,
            game.direction == Direction.DOWN,
            
            # Food location
            food[0] < head[0],  # food left
            food[0] > head[0],  # food right
            food[1] < head[1],  # food up
            food[1] > head[1],  # food down
            
            # Snake length
            len(game.snake) / (game.GRID_COUNT * game.GRID_COUNT)
        ]
        
        return torch.FloatTensor(state).to(self.device)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        
        with torch.no_grad():
            return self.policy_net(state).argmax().item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.stack(batch[0])
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.stack(batch[3])
        done_batch = torch.FloatTensor(batch[4]).to(self.device)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma * (1 - done_batch)) + reward_batch
        
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()

    def train(self, episodes=10000, save_interval=100, render_interval=10):
        game = SnakeGame()
        recent_scores = deque(maxlen=100)
        start_time = time.time()
        episode_steps = 0
        update_count = 0
        
        print("Training startet...")
        print(f"Gerät: {self.device}")
        print("Drücken Sie Strg+C zum Beenden")
        print("-" * 50)
        
        try:
            for episode in range(episodes):
                game.reset_game()
                game.game_state = "PLAYING"
                state = self.get_state(game)
                total_reward = 0
                episode_loss = 0
                steps_without_food = 0
                
                while True:
                    action = self.get_action(state)
                    
                    # Konvertiere Aktion
                    if action == 1:  # Rechts
                        if game.direction == Direction.UP:
                            game.direction = Direction.RIGHT
                        elif game.direction == Direction.RIGHT:
                            game.direction = Direction.DOWN
                        elif game.direction == Direction.DOWN:
                            game.direction = Direction.LEFT
                        else:
                            game.direction = Direction.UP
                    elif action == 2:  # Links
                        if game.direction == Direction.UP:
                            game.direction = Direction.LEFT
                        elif game.direction == Direction.LEFT:
                            game.direction = Direction.DOWN
                        elif game.direction == Direction.DOWN:
                            game.direction = Direction.RIGHT
                        else:
                            game.direction = Direction.UP
                    
                    # Spiel aktualisieren
                    old_length = len(game.snake)
                    game.update()
                    new_length = len(game.snake)
                    done = game.game_state == "GAME_OVER"
                    
                    # Belohnung berechnen
                    reward = 0
                    if done:
                        reward = -10
                    elif new_length > old_length:
                        reward = 10
                        steps_without_food = 0
                    else:
                        steps_without_food += 1
                        if steps_without_food > 100:
                            done = True
                            reward = -10
                    
                    # Nächsten Zustand holen
                    next_state = self.get_state(game)
                    
                    # Speichern und Trainieren
                    self.memory.push(state, action, reward, next_state, done)
                    loss = self.train_step()
                    if loss:
                        episode_loss += loss
                    
                    state = next_state
                    total_reward += reward
                    episode_steps += 1
                    
                    # Visualisierung
                    if (episode + 1) % render_interval == 0:
                        game.draw()
                        pygame.display.flip()
                        pygame.time.delay(20)
                    
                    if done:
                        break
                
                # Episode Update
                score = game.score
                self.scores.append(score)
                recent_scores.append(score)
                self.avg_scores.append(np.mean(recent_scores))
                self.epsilon_history.append(self.epsilon)
                self.loss_history.append(episode_loss)
                
                # Epsilon Update
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
                
                # Target Network Update
                update_count += 1
                if update_count % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                # Bester Score Update
                if score > self.best_score:
                    self.best_score = score
                    self.best_episode = episode + 1
                    print(f"\nNeuer Highscore! Score: {self.best_score} in Episode {self.best_episode}")
                
                # Speichern und Statistiken
                if (episode + 1) % save_interval == 0:
                    self.save_model()
                    self.plot_progress()
                    
                    elapsed_time = time.time() - start_time
                    avg_time_per_episode = elapsed_time / (episode + 1)
                    remaining_episodes = episodes - (episode + 1)
                    estimated_remaining_time = remaining_episodes * avg_time_per_episode
                    
                    print(f"\nEpisode: {episode + 1}/{episodes}")
                    print(f"Score: {score}, Best Score: {self.best_score}")
                    print(f"Avg Score (last 100): {np.mean(recent_scores):.2f}")
                    print(f"Epsilon: {self.epsilon:.4f}")
                    print(f"Durchschnittlicher Loss: {episode_loss:.4f}")
                    print(f"Geschätzte verbleibende Zeit: {estimated_remaining_time/60:.1f} Minuten")
                    print("-" * 50)
                
                # Event Handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
        
        except KeyboardInterrupt:
            print("\nTraining wurde manuell beendet")
            self.save_model()
            self.plot_progress()
        
        print("\nTraining abgeschlossen!")
        print(f"Bester Score: {self.best_score} (Episode {self.best_episode})")
        print(f"Durchschnittlicher Score (letzte 100): {np.mean(recent_scores):.2f}")
        
        self.save_model()
        self.plot_progress()

    def save_model(self):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scores': self.scores,
            'avg_scores': self.avg_scores,
            'epsilon_history': self.epsilon_history,
            'loss_history': self.loss_history,
            'best_score': self.best_score,
            'best_episode': self.best_episode,
            'epsilon': self.epsilon
        }, 'snake_model.pth')

    def load_model(self):
        if os.path.exists('snake_model.pth'):
            checkpoint = torch.load('snake_model.pth')
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scores = checkpoint['scores']
            self.avg_scores = checkpoint['avg_scores']
            self.epsilon_history = checkpoint['epsilon_history']
            self.loss_history = checkpoint['loss_history']
            self.best_score = checkpoint['best_score']
            self.best_episode = checkpoint['best_episode']
            self.epsilon = checkpoint['epsilon']

    def plot_progress(self):
        plt.figure(figsize=(15, 5))
        
        # Score Plot
        plt.subplot(131)
        plt.plot(self.scores)
        plt.title('Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        # Average Score Plot
        plt.subplot(132)
        plt.plot(self.avg_scores)
        plt.title('Average Scores (last 100)')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
        
        # Epsilon Plot
        plt.subplot(133)
        plt.plot(self.epsilon_history)
        plt.title('Epsilon')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

    def play(self, episodes=5, render=True):
        game = SnakeGame()
        self.epsilon = 0  # Keine Exploration im Spiel-Modus
        total_score = 0
        
        for episode in range(episodes):
            game.reset_game()
            game.game_state = "PLAYING"
            state = self.get_state(game)
            done = False
            
            while not done:
                action = self.get_action(state)
                
                if action == 1:  # Rechts
                    if game.direction == Direction.UP:
                        game.direction = Direction.RIGHT
                    elif game.direction == Direction.RIGHT:
                        game.direction = Direction.DOWN
                    elif game.direction == Direction.DOWN:
                        game.direction = Direction.LEFT
                    else:
                        game.direction = Direction.UP
                elif action == 2:  # Links
                    if game.direction == Direction.UP:
                        game.direction = Direction.LEFT
                    elif game.direction == Direction.LEFT:
                        game.direction = Direction.DOWN
                    elif game.direction == Direction.DOWN:
                        game.direction = Direction.RIGHT
                    else:
                        game.direction = Direction.UP
                
                game.update()
                done = game.game_state == "GAME_OVER"
                state = self.get_state(game)
                
                if render:
                    game.draw()
                    pygame.display.flip()
                    pygame.time.delay(50)
            
            total_score += game.score
            print(f"Episode {episode + 1}: Score = {game.score}")
        
        if episodes > 1:
            print(f"Durchschnittlicher Score: {total_score/episodes:.2f}")

if __name__ == "__main__":
    ai = SnakeAI()
    print("Snake KI Training")
    print("================")
    print("Die KI wird jetzt trainiert. Sie können das Training jederzeit mit Strg+C beenden.")
    print("Das Spiel wird alle 10 Episoden visualisiert.")
    print("Statistiken werden alle 100 Episoden angezeigt.")
    print("\nTraining startet in 3 Sekunden...")
    time.sleep(3)
    ai.train(episodes=10000, save_interval=100, render_interval=10)
    print("\nTraining abgeschlossen. Starte Demo-Spiele...")
    time.sleep(2)
    ai.play(episodes=5) 