import pygame
import sys
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Letter Collection Game with AI")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
COLORS = [RED, GREEN, BLUE, YELLOW, PURPLE]

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DeepQLearningAI:
    def __init__(self, input_size, output_size, learning_rate=0.001, discount_factor=0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.output_size = output_size
        self.policy_net = DQN(input_size, output_size).to(self.device)
        self.target_net = DQN(input_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        self.discount_factor = discount_factor
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.update_target_every = 100
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.steps = 0
        self.model_file = "deep_ai_model.pth"
        
        self.load_model()
    
    def get_state(self, dot_x, dot_y, letters, target_index, total_letters):
        state = []
        
        state.append(dot_x / WIDTH)
        state.append(dot_y / HEIGHT)
        
        if target_index < len(letters):
            target_letter = letters[target_index]
            state.append(target_letter.x / WIDTH)
            state.append(target_letter.y / HEIGHT)
            state.append(1.0 if not target_letter.collected else 0.0)
            
            dist = math.sqrt((dot_x - target_letter.x)**2 + (dot_y - target_letter.y)**2)
            state.append(dist / math.sqrt(WIDTH**2 + HEIGHT**2))
        else:
            state.extend([0.0, 0.0, 0.0, 0.0])
        
        nearby_letters = 0
        for i, letter in enumerate(letters):
            if i != target_index and not letter.collected:
                dist = math.sqrt((dot_x - letter.x)**2 + (dot_y - letter.y)**2)
                if dist < 200:
                    state.append(letter.x / WIDTH)
                    state.append(letter.y / HEIGHT)
                    state.append(dist / math.sqrt(WIDTH**2 + HEIGHT**2))
                    nearby_letters += 1
                    if nearby_letters >= 2:
                        break
        
        while nearby_letters < 2:
            state.extend([0.0, 0.0, 0.0])
            nearby_letters += 1
        
        state.append(target_index / max(1, total_letters))
        
        expected_size = 14
        while len(state) < expected_size:
            state.append(0.0)
        
        return np.array(state[:expected_size], dtype=np.float32)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(4)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_array = np.array(states)
        next_states_array = np.array(next_states)
        
        states_tensor = torch.FloatTensor(states_array).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states_array).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        next_q_values = self.target_net(next_states_tensor).max(1)[0].detach()
        target_q_values = rewards_tensor + (self.discount_factor * next_q_values * ~dones_tensor)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self):
        try:
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps,
                'input_size': self.input_size,
                'output_size': self.output_size
            }, self.model_file)
            print("Model saved successfully!")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        try:
            if os.path.exists(self.model_file):
                checkpoint = torch.load(self.model_file)
                
                if (checkpoint.get('input_size') == self.input_size and 
                    checkpoint.get('output_size') == self.output_size):
                    
                    self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                    self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.epsilon = checkpoint['epsilon']
                    self.steps = checkpoint['steps']
                    print("Model loaded successfully!")
                    print(f"Current epsilon: {self.epsilon:.3f}")
                    print(f"Steps: {self.steps}")
                else:
                    print("Model sizes don't match, starting from scratch")
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            else:
                print("No pre-trained model found, starting from scratch")
                self.target_net.load_state_dict(self.policy_net.state_dict())
        except Exception as e:
            print(f"Error loading model: {e}")

class SmartDot:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.radius = 10
        self.speed = 5
        self.score = 0
        self.total_reward = 0
        
        self.input_size = 14
        self.output_size = 4
        self.ai = DeepQLearningAI(self.input_size, self.output_size)
        
        self.last_state = None
        self.last_action = None
        self.steps_without_progress = 0
        self.episode_reward = 0
        self.rewards_history = []
    
    def get_action_from_ai(self, letters, target_index, total_letters):
        state = self.ai.get_state(self.x, self.y, letters, target_index, total_letters)
        action = self.ai.act(state)
        return action, state
    
    def apply_action(self, action):
        if action == 0:
            self.y = max(self.radius, self.y - self.speed)
        elif action == 1:
            self.y = min(HEIGHT - self.radius, self.y + self.speed)
        elif action == 2:
            self.x = max(self.radius, self.x - self.speed)
        elif action == 3:
            self.x = min(WIDTH - self.radius, self.x + self.speed)
    
    def give_reward(self, reward, next_letters, next_target_index, total_letters, done):
        if self.last_state is not None and self.last_action is not None:
            next_state = self.ai.get_state(self.x, self.y, next_letters, next_target_index, total_letters)
            self.ai.remember(self.last_state, self.last_action, reward, next_state, done)
            self.ai.replay()
            
            self.episode_reward += reward
            self.total_reward += reward
        
        self.last_state = None
        self.last_action = None
    
    def draw(self):
        pygame.draw.circle(screen, GREEN, (int(self.x), int(self.y)), self.radius)

class Letter:
    def __init__(self, char, x, y):
        self.char = char
        self.x = x
        self.y = y
        self.radius = 20
        self.color = random.choice(COLORS)
        self.collected = False
    
    def draw(self):
        if not self.collected:
            pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)
            font = pygame.font.SysFont('arial', 30)
            text = font.render(self.char, True, WHITE)
            text_rect = text.get_rect(center=(self.x, self.y))
            screen.blit(text, text_rect)

def create_letters(word):
    letters = []
    for char in word:
        while True:
            x = random.randint(50, WIDTH - 50)
            y = random.randint(50, HEIGHT - 50)
            
            overlap = False
            for letter in letters:
                if math.sqrt((x - letter.x)**2 + (y - letter.y)**2) < 60:
                    overlap = True
                    break
            
            if not overlap:
                break
        
        letters.append(Letter(char, x, y))
    return letters

def find_target_letter(letters, target_index):
    if target_index >= len(letters):
        return None
    
    target_letter = letters[target_index]
    if not target_letter.collected:
        return target_letter
    return None

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_progress_reward(dot, target_letter, wrong_letters):
    if target_letter is None:
        return 0
    
    dist_to_target = distance(dot.x, dot.y, target_letter.x, target_letter.y)
    max_dist = math.sqrt(WIDTH**2 + HEIGHT**2)
    progress_reward = (1.0 - (dist_to_target / max_dist)) * 0.5
    
    penalty = 0
    for letter in wrong_letters:
        if not letter.collected:
            dist_to_wrong = distance(dot.x, dot.y, letter.x, letter.y)
            if dist_to_wrong < 150:
                penalty += (1.0 - (dist_to_wrong / 150)) * 0.3
    
    return progress_reward - penalty

def main():
    clock = pygame.time.Clock()
    dot = SmartDot()
    
    input_active = True
    user_text = ''
    font = pygame.font.SysFont('arial', 36)
    
    while input_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                dot.ai.save_model()
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and user_text:
                    input_active = False
                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                else:
                    if event.unicode.isalpha():
                        user_text += event.unicode
        
        screen.fill(BLACK)
        
        pygame.draw.rect(screen, WHITE, (WIDTH//2 - 150, HEIGHT//2 - 25, 300, 50), 2)
        
        text_surface = font.render(user_text, True, WHITE)
        screen.blit(text_surface, (WIDTH//2 - 140, HEIGHT//2 - 15))
        
        instruction = font.render("Enter a word and press Enter", True, WHITE)
        screen.blit(instruction, (WIDTH//2 - 150, HEIGHT//2 - 70))
        
        pygame.display.flip()
        clock.tick(30)
    
    letters = create_letters(user_text)
    target_index = 0
    game_over = False
    episode_count = 0
    max_episodes = 200
    total_letters = len(letters)
    
    while not game_over and episode_count < max_episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                dot.ai.save_model()
                pygame.quit()
                sys.exit()
        
        target_letter = find_target_letter(letters, target_index)
        action, state = dot.get_action_from_ai(letters, target_index, total_letters)
        
        dot.last_state = state
        dot.last_action = action
        
        dot.apply_action(action)
        
        wrong_letters = [letter for i, letter in enumerate(letters) if i != target_index and not letter.collected]
        
        progress_reward = calculate_progress_reward(dot, target_letter, wrong_letters)
        dot.give_reward(progress_reward, letters, target_index, total_letters, False)
        
        collision_occurred = False
        for i, letter in enumerate(letters):
            if not letter.collected:
                dist = distance(dot.x, dot.y, letter.x, letter.y)
                if dist < dot.radius + letter.radius:
                    collision_occurred = True
                    
                    if i == target_index:
                        letter.collected = True
                        dot.give_reward(10.0, letters, target_index + 1, total_letters, False)
                        target_index += 1
                        dot.score += 10
                        dot.steps_without_progress = 0
                        
                        if target_index >= len(letters):
                            dot.give_reward(50.0, letters, target_index, total_letters, True)
                            game_over = True
                    
                    else:
                        dot.give_reward(-5.0, letters, target_index, total_letters, False)
                        dot.steps_without_progress += 10
        
        if not collision_occurred:
            dot.steps_without_progress += 1
            if dot.steps_without_progress > 100:
                dot.give_reward(-2.0, letters, target_index, total_letters, False)
                dot.steps_without_progress = 0
        
        screen.fill(BLACK)
        
        for letter in letters:
            letter.draw()
        
        dot.draw()
        
        if target_letter and not target_letter.collected:
            pygame.draw.line(screen, YELLOW, (dot.x, dot.y), 
                            (target_letter.x, target_letter.y), 1)
        
        score_text = font.render(f"Score: {dot.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        reward_text = font.render(f"Total Reward: {dot.total_reward:.2f}", True, WHITE)
        screen.blit(reward_text, (10, 50))
        
        episode_text = font.render(f"Episode: {episode_count + 1}/{max_episodes}", True, WHITE)
        screen.blit(episode_text, (10, 90))
        
        epsilon_text = font.render(f"Exploration: {dot.ai.epsilon:.3f}", True, WHITE)
        screen.blit(epsilon_text, (10, 130))
        
        if game_over:
            win_text = font.render("You Win! Collected all letters!", True, GREEN)
            screen.blit(win_text, (WIDTH//2 - 200, HEIGHT//2))
            
            dot.rewards_history.append(dot.episode_reward)
            
            episode_count += 1
            if episode_count < max_episodes:
                pygame.time.wait(500)
                letters = create_letters(user_text)
                target_index = 0
                game_over = False
                dot.x = WIDTH // 2
                dot.y = HEIGHT // 2
                dot.episode_reward = 0
                dot.steps_without_progress = 0
        
        pygame.display.flip()
        clock.tick(60)
    
    dot.ai.save_model()
    
    final_font = pygame.font.SysFont('arial', 48)
    final_text = final_font.render(f"Training Complete! Total Reward: {dot.total_reward:.2f}", True, GREEN)
    screen.blit(final_text, (WIDTH//2 - 300, HEIGHT//2))
    pygame.display.flip()
    
    pygame.time.wait(5000)
    dot.ai.save_model()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
