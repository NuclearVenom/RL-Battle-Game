# ⚔️ RL Battle Game

<span style="color: #00ffffc9; font-family: monospace; font-weight: bold;">A turn-based battle arena where you fight an AI opponent that **learns and adapts** using Reinforcement Learning and the Q-Learning algorithm. The more it plays, the smarter it gets.
</span>

*Developed by [Ranasurya Ghosh](https://github.com/NuclearVenom)* <br><br>
[![LICENSE](https://img.shields.io/badge/LICENSE-MIT-blue)](./LICENSE)

---



## Table of Contents



- [⚔️ RL Battle Game](#️-rl-battle-game)
  - [Table of Contents](#table-of-contents)
  - [What Is This?](#what-is-this)
  - [How to Play](#how-to-play)
  - [Game Mechanics](#game-mechanics)
  - [How the AI Works](#how-the-ai-works)
    - [What Is Reinforcement Learning?](#what-is-reinforcement-learning)
    - [What Is Q-Learning?](#what-is-q-learning)
    - [How Q-Learning Works in This Game](#how-q-learning-works-in-this-game)
    - [The Q-Table: The AI's Brain](#the-q-table-the-ais-brain)
    - [Exploration vs. Exploitation](#exploration-vs-exploitation)
    - [Training Against Bots](#training-against-bots)
  - [Project Structure](#project-structure)
  - [| `BattleGame` | Orchestrates everything — battles, training, menus, stats |](#-battlegame--orchestrates-everything--battles-training-menus-stats-)
  - [Installation \& Running](#installation--running)
  - [Key Parameters \& Tuning](#key-parameters--tuning)



---



## What Is This?



**RL Battle Game** is a command-line fighting game written in Python. You play as a warrior facing off against an AI opponent in a turn-based battle. What makes this game special is that the AI is **not scripted** — it doesn't follow a fixed set of rules that a human programmer wrote. Instead, it *teaches itself* how to fight through trial and error, using a technique from the field of **Artificial Intelligence** called **Reinforcement Learning (RL)**.



Every battle — whether against you or against practice bots — makes the AI a little bit wiser. Its knowledge is saved to a file (`ai_brain.pkl`) so it remembers everything it has ever learned, even after you close the program.



---



## How to Play



Run the game:



```bash

python game.py

```



You'll be greeted with a main menu:



```

⚔️  WELCOME TO BATTLE ARENA! ⚔️



1. 🗡️  Battle the AI

2. 🏋️  Train AI against bots

3. 🎯 Mixed training (recommended)

4. 📊 View statistics

5. 🔄 Reset AI brain

6. 🚪 Quit game

```



Choose **option 1** to fight the AI directly, or use options 2–3 to train the AI first and make it a stronger opponent.



---



## Game Mechanics



Each fighter starts with **100 HP**. Players alternate turns. The first fighter to reach **0 HP** loses.



On your turn, you choose one of three moves:



| Move | Damage | Accuracy | Notes |
| --- |---| --- |---|
| ⚡ Light Attack | 15 HP | 85% | Reliable, consistent damage |
| 💥 Heavy Attack | 35 HP | 55% | High risk, high reward |
| 💊 Heal | +25 HP | 100% | Restores HP, limited to **5 uses** per battle |
Strategy matters. Should you play it safe with light attacks? Gamble on a heavy attack when the opponent is low on health? Save your heals for a desperate moment? The AI is asking itself the exact same questions — and learning the answers over time.



---



## How the AI Works



### What Is Reinforcement Learning?



Imagine you are learning to ride a bicycle for the first time. Nobody gives you a textbook that says "when you tilt left by 12 degrees, apply 3 Newtons of force to the right pedal." Instead, you just *try*, fall over, adjust, and try again. Over time, your brain figures out what works and what doesn't — without anyone explicitly programming the rules into you.



**Reinforcement Learning (RL)** is the same idea, but for computers. An AI *agent* (the AI fighter in this game) interacts with an *environment* (the battle), takes *actions* (Light Attack, Heavy Attack, Heal), and receives *rewards* or *penalties* based on the outcome.



The goal of the agent is simple: **maximize its total reward over time.**



In this game:

- **Winning a battle** → the AI receives a reward of `+10`

- **Losing a battle** → the AI receives a penalty of `-10`

- **During the battle** (every turn) → the AI receives `0` (neutral — the fight isn't over yet)



The AI doesn't know upfront which moves are good or bad. It has to *discover* this through experience.



---



### What Is Q-Learning?



**Q-Learning** is one of the most famous algorithms in Reinforcement Learning. Think of it as the AI building a giant **cheat sheet** (called a Q-Table) that maps every possible situation it might face to a score for each possible action.



The "Q" stands for **Quality** — as in, how *good* is it to take a certain action in a certain situation?



The core idea:



> "If I'm in *this situation* and I take *this action*, how much total reward can I expect to get in the long run?"



The AI uses this cheat sheet to pick the action with the highest expected reward. And after every turn, it updates the cheat sheet based on what actually happened.



---



### How Q-Learning Works in This Game



Let's walk through a real example step by step.



**Step 1: Observe the Situation (State)**



Before the AI makes a move, it looks at the current battle situation. In this game, the "state" is defined by four things:



- The AI's current HP (bucketed into ranges of 10)

- The player's current HP (bucketed into ranges of 10)

- How many heals the AI has left

- How many heals the player has left



For example, a state might look like: `"7-3-2-5"` meaning the AI is at 70–79 HP, the player is at 30–39 HP, the AI has 2 heals left, and the player has 5 heals left.



Bucketing health into ranges (instead of tracking every exact HP value) keeps the number of possible states manageable. Without this, there would be millions of combinations to track.



**Step 2: Choose an Action**



The AI looks up this state in its Q-Table and checks the score (Q-value) for each possible move. It then picks the move with the highest score. This is called **exploitation** — using what it already knows.



**Step 3: Take the Action & Observe the Reward**



The AI executes the move. The battle plays out. After the battle ends, the AI gets a reward (`+10` or `-10`).



**Step 4: Update the Q-Table (The Learning Step)**



This is where the magic happens. The AI updates its Q-Table using the **Q-Learning formula**:



```

New Q-Value = Old Q-Value + Learning Rate × (Reward + Discount Factor × Best Future Q-Value − Old Q-Value)

```



Let's decode this in plain English:



- **Old Q-Value**: How good the AI *thought* the action was before this experience.

- **Reward**: What it actually got (good or bad outcome).

- **Best Future Q-Value**: The best score the AI can expect to get from the *next* state it lands in. This is what makes Q-Learning *forward-thinking* — it doesn't just consider the immediate reward, but also how good the situation it ends up in is.

- **Learning Rate** (`0.1`): Controls how drastically the AI updates its beliefs. A low value means it learns slowly and steadily.

- **Discount Factor** (`0.9`): How much the AI cares about *future* rewards vs *immediate* rewards. A value of `0.9` means future rewards are nearly as important as present ones.



After thousands of battles, the Q-Table gradually fills up with accurate scores for every situation. The AI learns things like:

- "When the player is at low HP, Heavy Attack is worth risking."

- "When I'm at low HP and I still have heals, Heal is the best move."

- "Light Attack is a safe default when things are uncertain."



---



### The Q-Table: The AI's Brain



The Q-Table is simply a dictionary (a lookup table) stored in memory — and saved to disk as `ai_brain.pkl` after every session.



It looks something like this conceptually:



| State | Light Attack (Q) | Heavy Attack (Q) | Heal (Q) |
| --- |---| --- |---|
| `"7-3-2-5"` | 4.2 | 6.8 | 1.1 |
| `"2-8-0-3"` | 2.1 | -1.3 | — |
| `"5-5-3-3"` | 3.0 | 2.7 | 3.5 |
The AI always picks the action with the highest Q-value for the current state. In the first row, it would choose Heavy Attack (6.8 is the highest).



The table grows with experience. A freshly reset AI has an empty table and knows nothing. After hundreds of training battles, it might have learned thousands of distinct states.



---



### Exploration vs. Exploitation



There's a classic dilemma in Reinforcement Learning:



- **Exploitation**: Use what you already know — pick the move with the best known Q-value.

- **Exploration**: Try something random — maybe you'll discover a better strategy you haven't tried yet.



If the AI *only* exploits, it gets stuck with whatever strategy it learned early on, even if a better one exists. If it *only* explores, it never settles on a good strategy. The balance is crucial.



This game solves it with the **epsilon-greedy strategy**, controlled by a parameter called `epsilon` (ε):



- With probability ε → pick a **random move** (explore)

- With probability 1 − ε → pick the **best known move** (exploit)



At the start of training, `epsilon = 0.2`, meaning the AI explores 20% of the time. As training progresses, epsilon slowly decreases (down to a minimum of `0.05`). The AI starts curious and experimental, then gradually becomes more confident and decisive — just like a student who guesses at first but becomes more certain as they study.



---



### Training Against Bots



You can train the AI against automated bots (no human needed) for hundreds of battles at once. Four bot personalities are available:



| Bot Type | Behavior |
| --- |---|
| **Random** | Makes completely unpredictable moves |
| **Aggressive** | Spams heavy attacks, heals only near death |
| **Defensive** | Heals often, prefers safe light attacks |
| **Smart** | Makes contextual decisions (heals when low, goes for the kill when enemy is weak) |
**Mixed Training** (option 3) trains the AI against all four bot types sequentially. This gives it a well-rounded education — it doesn't just learn to beat one playstyle. Think of it like a martial artist sparring with opponents of different styles before a real match.



---



## Project Structure



```

rl-battle-game/
│
├── game.py            # Main source file (all game logic)
└── ai_brain.pkl       # Saved Q-Table (created automatically after first run)

```



**Key classes inside `game.py`:**



| Class | Role |
| --- |---|
| `Move` | Enum defining the three possible moves |
| `BattleResult` | Enum for win/loss/ongoing states |
| `Fighter` | Represents a combatant (health, heals, actions) |
| `BattleEngine` | Executes moves and calculates damage/healing |
| `QLearningAgent` | The brain — holds the Q-Table, chooses actions, learns |
| `OpponentBot` | Practice bots with four different strategies |
| `BattleGame` | Orchestrates everything — battles, training, menus, stats |
---



## Installation & Running



**Requirements:** Python 3.7+ with NumPy



```bash

# Install dependencies

pip install numpy



# Run the game

python game.py

```



No other libraries or setup required. The game creates `ai_brain.pkl` automatically in the same directory.



---



## Key Parameters & Tuning



You can tweak the AI's learning behavior by modifying these values in the `QLearningAgent` constructor:



| Parameter | Default | What It Does |
| --- |---| --- |
| `learning_rate` | `0.1` | How fast the AI updates its beliefs. Higher = faster but less stable learning. |
| `discount_factor` | `0.9` | How much the AI values future rewards. Closer to `1.0` = more long-term thinking. |
| `epsilon` | `0.2` | Starting exploration rate. Higher = more random early behavior. |
And in `BattleEngine.MOVE_CONFIGS` you can adjust damage values, accuracy, and healing amounts to change the game balance entirely.



---
<br>

<span style="font-family: 'Courier New', cursive; font-size: 20px; font-style: italic;">The AI starts as a blank slate. Every battle is a lesson. Train it well.</span>
