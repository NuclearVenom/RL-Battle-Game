import random
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional
from enum import Enum

class Move(Enum):
    """Enumeration for different moves available in battle"""
    LIGHT_ATTACK = 0    # Low damage, high accuracy
    HEAVY_ATTACK = 1    # High damage, low accuracy  
    HEAL = 2           # Restore health, limited uses

class BattleResult(Enum):
    """Enumeration for battle outcomes"""
    PLAYER_WIN = 1
    AI_WIN = -1
    ONGOING = 0

class Fighter:
    """Represents a fighter in the battle with health, moves, and stats"""
    
    def __init__(self, name: str, max_health: int = 100):
        """
        Initialize a fighter
        
        Args:
            name: Fighter's name
            max_health: Maximum health points
        """
        self.name = name
        self.max_health = max_health
        self.health = max_health
        self.heal_uses = 5  # Maximum number of heal uses per battle
        
    def reset(self):
        """Reset fighter to initial state for new battle"""
        self.health = self.max_health
        self.heal_uses = 5
        
    def is_alive(self) -> bool:
        """Check if fighter is still alive"""
        return self.health > 0
    
    def take_damage(self, damage: int):
        """
        Apply damage to the fighter
        
        Args:
            damage: Amount of damage to take
        """
        self.health = max(0, self.health - damage)
        
    def heal_self(self, amount: int) -> bool:
        """
        Heal the fighter if heal uses are available
        
        Args:
            amount: Amount of health to restore
            
        Returns:
            bool: True if healing was successful, False if no heals left
        """
        if self.heal_uses <= 0:
            return False
            
        self.heal_uses -= 1
        self.health = min(self.max_health, self.health + amount)
        return True
    
    def get_health_percentage(self) -> float:
        """Get health as percentage of max health"""
        return (self.health / self.max_health) * 100

class BattleEngine:
    """Core battle system that handles move execution and damage calculation"""
    
    # Move configurations: (base_damage, accuracy, heal_amount)
    MOVE_CONFIGS = {
        Move.LIGHT_ATTACK: {"damage": 15, "accuracy": 0.85, "description": "Light Attack (15 dmg, 85% accuracy)"},
        Move.HEAVY_ATTACK: {"damage": 35, "accuracy": 0.55, "description": "Heavy Attack (35 dmg, 55% accuracy)"},
        Move.HEAL: {"heal": 25, "accuracy": 1.0, "description": "Heal (25 HP, limited uses)"}
    }
    
    @staticmethod
    def execute_move(attacker: Fighter, defender: Fighter, move: Move) -> Tuple[bool, str]:
        """
        Execute a move and return the result
        
        Args:
            attacker: Fighter making the move
            defender: Fighter receiving the move (if attack)
            move: Move being executed
            
        Returns:
            Tuple of (success, description) where success indicates if move hit/worked
        """
        config = BattleEngine.MOVE_CONFIGS[move]
        
        # Check if move hits based on accuracy
        hits = random.random() < config["accuracy"]
        
        if move == Move.HEAL:
            if attacker.heal_uses <= 0:
                return False, f"{attacker.name} has no heals left!"
            
            if attacker.heal_self(config["heal"]):
                return True, f"{attacker.name} heals for {config['heal']} HP! ({attacker.heal_uses} heals left)"
            else:
                return False, f"{attacker.name} couldn't heal!"
                
        else:  # Attack moves
            if hits:
                damage = config["damage"]
                defender.take_damage(damage)
                return True, f"{attacker.name} hits {defender.name} for {damage} damage!"
            else:
                return False, f"{attacker.name}'s attack misses!"

class QLearningAgent:
    """Q-Learning agent for battle decision making"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, epsilon: float = 0.2):
        """
        Initialize Q-Learning agent
        
        Args:
            learning_rate: How quickly to learn from new experiences
            discount_factor: How much to value future rewards
            epsilon: Exploration rate (chance to make random moves)
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table: Dict[str, Dict[int, float]] = {}
        
        # Track last state and action for learning
        self.last_state = None
        self.last_action = None
        
    def get_state_key(self, ai_health: int, player_health: int, ai_heals: int, player_heals: int) -> str:
        """
        Convert battle state to string key for Q-table
        
        Args:
            ai_health: AI's current health
            player_health: Player's current health  
            ai_heals: AI's remaining heals
            player_heals: Player's remaining heals
            
        Returns:
            String representation of state
        """
        # Discretize health into ranges for manageable state space
        ai_health_range = min(ai_health // 10, 10)  # 0-10 range
        player_health_range = min(player_health // 10, 10)  # 0-10 range
        
        return f"{ai_health_range}-{player_health_range}-{ai_heals}-{player_heals}"
    
    def get_q_value(self, state: str, action: int) -> float:
        """
        Get Q-value for state-action pair
        
        Args:
            state: Current state key
            action: Action index (0, 1, or 2)
            
        Returns:
            Q-value for the state-action pair
        """
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        return self.q_table[state][action]
    
    def update_q_value(self, state: str, action: int, reward: float, next_state: str):
        """
        Update Q-value using Q-learning formula
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state after action
        """
        current_q = self.get_q_value(state, action)
        
        # Find maximum Q-value for next state
        max_next_q = 0.0
        if next_state in self.q_table:
            if self.q_table[next_state]:
                max_next_q = max(self.q_table[next_state].values())
        
        # Q-learning update formula
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q
    
    def choose_action(self, state: str, available_actions: List[int]) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state key
            available_actions: List of valid action indices
            
        Returns:
            Chosen action index
        """
        # Exploration: random action
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Exploitation: best known action
        q_values = [self.get_q_value(state, action) for action in available_actions]
        max_q = max(q_values)
        
        # Handle ties by choosing randomly among best actions
        best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def learn(self, reward: float, next_state: str):
        """
        Learn from the last action taken
        
        Args:
            reward: Reward received for the last action
            next_state: State after the last action
        """
        if self.last_state is not None and self.last_action is not None:
            self.update_q_value(self.last_state, self.last_action, reward, next_state)
    
    def save_q_table(self, filename: str):
        """Save Q-table to file for persistence"""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load_q_table(self, filename: str):
        """Load Q-table from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Loaded AI brain with {len(self.q_table)} learned states")
        else:
            print("No existing AI brain found. Starting fresh.")

class OpponentBot:
    """Different types of bots for training the AI"""
    
    def __init__(self, bot_type: str = "random"):
        """
        Initialize opponent bot
        
        Args:
            bot_type: Type of bot strategy to use
        """
        self.bot_type = bot_type
    
    def choose_action(self, own_health: int, enemy_health: int, own_heals: int, enemy_heals: int) -> Move:
        """
        Choose action based on bot strategy
        
        Args:
            own_health: Bot's current health
            enemy_health: Enemy's current health
            own_heals: Bot's remaining heals
            enemy_heals: Enemy's remaining heals
            
        Returns:
            Chosen move
        """
        if self.bot_type == "random":
            return self._random_strategy(own_heals)
        elif self.bot_type == "aggressive":
            return self._aggressive_strategy(own_health, own_heals)
        elif self.bot_type == "defensive":
            return self._defensive_strategy(own_health, enemy_health, own_heals)
        elif self.bot_type == "smart":
            return self._smart_strategy(own_health, enemy_health, own_heals)
        else:
            return self._random_strategy(own_heals)
    
    def _random_strategy(self, own_heals: int) -> Move:
        """Random move selection"""
        moves = [Move.LIGHT_ATTACK, Move.HEAVY_ATTACK]
        if own_heals > 0:
            moves.append(Move.HEAL)
        return random.choice(moves)
    
    def _aggressive_strategy(self, own_health: int, own_heals: int) -> Move:
        """Aggressive strategy: mostly attacks, heal only when very low"""
        if own_health < 20 and own_heals > 0:
            return Move.HEAL
        
        # Prefer heavy attacks when aggressive
        return random.choice([Move.HEAVY_ATTACK, Move.HEAVY_ATTACK, Move.LIGHT_ATTACK])
    
    def _defensive_strategy(self, own_health: int, enemy_health: int, own_heals: int) -> Move:
        """Defensive strategy: heal more often, prefer safe attacks"""
        if own_health < 40 and own_heals > 0:
            return Move.HEAL
        
        # Prefer light attacks for consistency
        return random.choice([Move.LIGHT_ATTACK, Move.LIGHT_ATTACK, Move.HEAVY_ATTACK])
    
    def _smart_strategy(self, own_health: int, enemy_health: int, own_heals: int) -> Move:
        """Smart strategy: contextual decision making"""
        # Emergency healing
        if own_health < 25 and own_heals > 0:
            return Move.HEAL
        
        # If enemy is low, go for the kill
        if enemy_health < 35:
            return Move.HEAVY_ATTACK
        
        # If we're healthy and they're healthy, balanced approach
        if own_health > 60:
            return random.choice([Move.LIGHT_ATTACK, Move.HEAVY_ATTACK])
        
        # If we're hurt but not critical, prefer safe damage
        return Move.LIGHT_ATTACK

class BattleGame:
    """Main game class that orchestrates battles and training"""
    
    def __init__(self):
        """Initialize the battle game"""
        self.player = Fighter("Player")
        self.ai = Fighter("AI")
        self.agent = QLearningAgent()
        self.q_table_file = "ai_brain.pkl"
        
        # Statistics tracking
        self.game_stats = {"player_wins": 0, "ai_wins": 0}
        self.training_stats = {"ai_wins": 0, "bot_wins": 0}
        
    def display_status(self):
        """Display current battle status"""
        print("\n" + "="*50)
        print("BATTLE STATUS")
        print("="*50)
        print(f"🤺 {self.player.name}: {self.player.health}/{self.player.max_health} HP "
              f"(💊 {self.player.heal_uses} heals)")
        print(f"🤖 {self.ai.name}: {self.ai.health}/{self.ai.max_health} HP "
              f"(💊 {self.ai.heal_uses} heals)")
        print("="*50)
    
    def display_moves(self):
        """Display available moves to player"""
        print("\nYour moves:")
        print("1. Light Attack (15 damage, 85% accuracy)")
        print("2. Heavy Attack (35 damage, 55% accuracy)")
        if self.player.heal_uses > 0:
            print(f"3. Heal (25 HP, {self.player.heal_uses} uses left)")
        else:
            print("3. Heal (UNAVAILABLE - no uses left)")
    
    def get_player_move(self) -> Move:
        """
        Get move choice from player
        
        Returns:
            Player's chosen move
        """
        while True:
            try:
                choice = int(input("\nChoose your move (1-3): "))
                if choice == 1:
                    return Move.LIGHT_ATTACK
                elif choice == 2:
                    return Move.HEAVY_ATTACK
                elif choice == 3:
                    if self.player.heal_uses > 0:
                        return Move.HEAL
                    else:
                        print("❌ No heals remaining!")
                        continue
                else:
                    print("❌ Invalid choice! Choose 1, 2, or 3.")
            except ValueError:
                print("❌ Please enter a number!")
    
    def get_ai_move(self) -> Move:
        """
        Get AI's move using Q-learning
        
        Returns:
            AI's chosen move
        """
        # Get current state
        state = self.agent.get_state_key(
            self.ai.health, self.player.health, 
            self.ai.heal_uses, self.player.heal_uses
        )
        
        # Determine available actions for AI
        available_actions = [0, 1]  # Light and heavy attack always available
        if self.ai.heal_uses > 0:
            available_actions.append(2)  # Heal available
        
        # Choose action using Q-learning
        action = self.agent.choose_action(state, available_actions)
        
        # Learn from previous action if exists
        if self.agent.last_state is not None:
            # Intermediate reward of 0 (battle ongoing)
            self.agent.learn(0, state)
        
        # Store current state and action for next learning step
        self.agent.last_state = state
        self.agent.last_action = action
        
        return Move(action)
    
    def execute_turn(self, attacker: Fighter, defender: Fighter, move: Move) -> str:
        """
        Execute a turn and return description
        
        Args:
            attacker: Fighter making the move
            defender: Fighter receiving the move
            move: Move being executed
            
        Returns:
            Description of what happened
        """
        success, description = BattleEngine.execute_move(attacker, defender, move)
        return description
    
    def check_battle_end(self) -> BattleResult:
        """
        Check if battle has ended
        
        Returns:
            Battle result enum
        """
        if not self.player.is_alive():
            return BattleResult.AI_WIN
        elif not self.ai.is_alive():
            return BattleResult.PLAYER_WIN
        else:
            return BattleResult.ONGOING
    
    def play_battle(self) -> BattleResult:
        """
        Play a single battle between player and AI
        
        Returns:
            Battle result
        """
        # Reset fighters for new battle
        self.player.reset()
        self.ai.reset()
        self.agent.last_state = None
        self.agent.last_action = None
        
        print("\n🗡️  BATTLE BEGINS! 🗡️")
        
        turn_count = 0
        while True:
            turn_count += 1
            self.display_status()
            
            # Player turn
            print(f"\n--- Turn {turn_count}: Your Turn ---")
            self.display_moves()
            player_move = self.get_player_move()
            player_result = self.execute_turn(self.player, self.ai, player_move)
            print(f"⚔️  {player_result}")
            
            # Check if AI is defeated
            battle_result = self.check_battle_end()
            if battle_result != BattleResult.ONGOING:
                break
            
            # AI turn
            print(f"\n--- Turn {turn_count}: AI's Turn ---")
            ai_move = self.get_ai_move()
            ai_result = self.execute_turn(self.ai, self.player, ai_move)
            print(f"🤖 {ai_result}")
            
            # Check if player is defeated
            battle_result = self.check_battle_end()
            if battle_result != BattleResult.ONGOING:
                break
        
        # Final learning update for AI
        if battle_result == BattleResult.AI_WIN:
            final_reward = 10  # AI wins
            self.game_stats["ai_wins"] += 1
        else:
            final_reward = -10  # AI loses
            self.game_stats["player_wins"] += 1
        
        # Final state after battle
        final_state = self.agent.get_state_key(
            self.ai.health, self.player.health,
            self.ai.heal_uses, self.player.heal_uses
        )
        
        self.agent.learn(final_reward, final_state)
        
        return battle_result
    
    def train_against_bot(self, num_battles: int, bot_type: str, show_progress: bool = True):
        """
        Train AI against a bot for specified number of battles
        
        Args:
            num_battles: Number of training battles
            bot_type: Type of bot to train against
            show_progress: Whether to show training progress
        """
        bot = OpponentBot(bot_type)
        training_results = {"ai_wins": 0, "bot_wins": 0}
        
        print(f"\n🏋️ Training AI against {bot_type} bot for {num_battles} battles...")
        
        for battle_num in range(num_battles):
            # Reset fighters
            ai_fighter = Fighter("AI")
            bot_fighter = Fighter("Bot")
            
            self.agent.last_state = None
            self.agent.last_action = None
            
            # Random starting turn
            ai_turn = random.choice([True, False])
            
            # Battle loop
            while ai_fighter.is_alive() and bot_fighter.is_alive():
                
                if ai_turn:
                    # AI's turn
                    state = self.agent.get_state_key(
                        ai_fighter.health, bot_fighter.health,
                        ai_fighter.heal_uses, bot_fighter.heal_uses
                    )
                    
                    available_actions = [0, 1]
                    if ai_fighter.heal_uses > 0:
                        available_actions.append(2)
                    
                    action = self.agent.choose_action(state, available_actions)
                    
                    # Learn from previous action
                    if self.agent.last_state is not None:
                        self.agent.learn(0, state)
                    
                    self.agent.last_state = state
                    self.agent.last_action = action
                    
                    # Execute AI move
                    ai_move = Move(action)
                    BattleEngine.execute_move(ai_fighter, bot_fighter, ai_move)
                    
                else:
                    # Bot's turn
                    bot_move = bot.choose_action(
                        bot_fighter.health, ai_fighter.health,
                        bot_fighter.heal_uses, ai_fighter.heal_uses
                    )
                    BattleEngine.execute_move(bot_fighter, ai_fighter, bot_move)
                
                ai_turn = not ai_turn
            
            # Determine winner and give final reward
            if ai_fighter.is_alive():
                final_reward = 10
                training_results["ai_wins"] += 1
            else:
                final_reward = -10
                training_results["bot_wins"] += 1
            
            # Final learning update
            final_state = self.agent.get_state_key(
                ai_fighter.health, bot_fighter.health,
                ai_fighter.heal_uses, bot_fighter.heal_uses
            )
            self.agent.learn(final_reward, final_state)
            
            # Show progress
            if show_progress and (battle_num + 1) % max(1, num_battles // 10) == 0:
                progress = (battle_num + 1) / num_battles * 100
                ai_win_rate = training_results["ai_wins"] / (battle_num + 1) * 100
                print(f"Progress: {progress:.0f}% - AI Win Rate: {ai_win_rate:.1f}%")
        
        # Update overall training stats
        self.training_stats["ai_wins"] += training_results["ai_wins"]
        self.training_stats["bot_wins"] += training_results["bot_wins"]
        
        # Reduce exploration over time
        self.agent.epsilon = max(0.05, self.agent.epsilon * (0.99 ** num_battles))
        
        print(f"\n✅ Training completed!")
        print(f"Final AI win rate: {training_results['ai_wins']/num_battles*100:.1f}%")
        print(f"New exploration rate: {self.agent.epsilon:.3f}")
        
        # Save progress
        self.agent.save_q_table(self.q_table_file)
        print("🧠 AI brain saved!")
    
    def mixed_training(self, total_battles: int = 400):
        """
        Train AI against multiple bot types for diverse experience
        
        Args:
            total_battles: Total number of training battles
        """
        bot_types = ["random", "aggressive", "defensive", "smart"]
        battles_per_bot = total_battles // len(bot_types)
        
        print(f"\n🎯 Starting comprehensive training with {total_battles} battles...")
        print("This will train against different strategies for well-rounded AI.")
        
        for bot_type in bot_types:
            print(f"\n--- Training Phase: {bot_type.upper()} Bot ---")
            self.train_against_bot(battles_per_bot, bot_type, show_progress=True)
        
        print(f"\n🏆 Mixed training complete!")
        total_trained = sum(self.training_stats.values())
        if total_trained > 0:
            ai_win_rate = self.training_stats["ai_wins"] / total_trained * 100
            print(f"Overall training performance: {ai_win_rate:.1f}% win rate")
    
    def show_statistics(self):
        """Display game and training statistics"""
        print(f"\n📊 BATTLE STATISTICS")
        print("="*40)
        
        # Player vs AI battles
        total_player_battles = sum(self.game_stats.values())
        if total_player_battles > 0:
            print(f"\n🤺 Player vs AI Battles ({total_player_battles} total):")
            player_win_rate = self.game_stats["player_wins"] / total_player_battles * 100
            ai_win_rate = self.game_stats["ai_wins"] / total_player_battles * 100
            print(f"   Player wins: {self.game_stats['player_wins']} ({player_win_rate:.1f}%)")
            print(f"   AI wins: {self.game_stats['ai_wins']} ({ai_win_rate:.1f}%)")
        else:
            print("\n🤺 No player battles yet.")
        
        # Training battles
        total_training_battles = sum(self.training_stats.values())
        if total_training_battles > 0:
            print(f"\n🏋️ Training Battles ({total_training_battles} total):")
            training_win_rate = self.training_stats["ai_wins"] / total_training_battles * 100
            print(f"   AI wins: {self.training_stats['ai_wins']} ({training_win_rate:.1f}%)")
            print(f"   Bot wins: {self.training_stats['bot_wins']} ({100-training_win_rate:.1f}%)")
        else:
            print("\n🏋️ No training battles yet.")
        
        # AI learning status
        print(f"\n🧠 AI Learning Status:")
        print(f"   Exploration rate: {self.agent.epsilon:.3f}")
        print(f"   States learned: {len(self.agent.q_table)}")
    
    def run(self):
        """Main game loop"""
        print("⚔️  WELCOME TO BATTLE ARENA! ⚔️")
        print("Face off against an AI that learns from every battle!")
        print("\nGame Rules:")
        print("• Light Attack: 15 damage, 85% accuracy")
        print("• Heavy Attack: 35 damage, 55% accuracy") 
        print("• Heal: Restore 25 HP, limited to 5 uses per battle")
        print("• First to reach 0 HP loses!")
        
        # Load existing AI brain
        self.agent.load_q_table(self.q_table_file)
        
        while True:
            print("\n" + "="*50)
            print("MAIN MENU")
            print("="*50)
            print("1. 🗡️  Battle the AI")
            print("2. 🏋️  Train AI against bots")
            print("3. 🎯 Mixed training (recommended)")
            print("4. 📊 View statistics")
            print("5. 🔄 Reset AI brain")
            print("6. 🚪 Quit game")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                result = self.play_battle()
                self.display_status()
                if result == BattleResult.PLAYER_WIN:
                    print("\n🎉 Congratulations! You won the battle!")
                else:
                    print("\n💀 You were defeated! The AI grows stronger...")
                
                # Reduce AI exploration after human battles
                self.agent.epsilon = max(0.05, self.agent.epsilon * 0.98)
                self.agent.save_q_table(self.q_table_file)
                
            elif choice == "2":
                print("\n🤖 Bot types available:")
                print("1. Random (unpredictable moves)")
                print("2. Aggressive (prefers heavy attacks)")
                print("3. Defensive (heals often, safe attacks)")
                print("4. Smart (contextual decisions)")
                
                bot_choice = input("Choose bot type (1-4): ").strip()
                bot_types = {"1": "random", "2": "aggressive", "3": "defensive", "4": "smart"}
                
                if bot_choice in bot_types:
                    try:
                        num_battles = int(input("Number of training battles (default 100): ") or "100")
                        self.train_against_bot(num_battles, bot_types[bot_choice])
                    except ValueError:
                        print("❌ Invalid number. Using default 100 battles.")
                        self.train_against_bot(100, bot_types[bot_choice])
                else:
                    print("❌ Invalid choice.")
                    
            elif choice == "3":
                try:
                    total_battles = int(input("Total training battles (default 400): ") or "400")
                    self.mixed_training(total_battles)
                except ValueError:
                    print("❌ Invalid number. Using default 400 battles.")
                    self.mixed_training(400)
                    
            elif choice == "4":
                self.show_statistics()
                
            elif choice == "5":
                confirm = input("⚠️  Reset AI brain? This will erase all learning! (y/n): ")
                if confirm.lower() == 'y':
                    self.agent.q_table = {}
                    self.agent.epsilon = 0.2
                    self.game_stats = {"player_wins": 0, "ai_wins": 0}
                    self.training_stats = {"ai_wins": 0, "bot_wins": 0}
                    print("🧠 AI brain reset! Starting fresh.")
                else:
                    print("Reset cancelled.")
                    
            elif choice == "6":
                self.agent.save_q_table(self.q_table_file)
                print("🧠 AI brain saved!👋 Thanks for playing! ")
                print("⭐ If you enjoyed the game, make sure to give a star on GitHub: https://github.com/NuclearVenom/RL-Battle-Game")
                break
                
            else:
                print("❌ Invalid choice. Please choose 1-6.")

if __name__ == "__main__":
    game = BattleGame()
    game.run()
