import discord
from discord.ext import commands
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

# Download NLTK data
nltk.download('punkt')

# File to store bomb locations
data = "data.txt"
bomb_locations = []

# Discord bot token
TOKEN = ''

# Initialize the bot and slash commands
bot = commands.Bot(intents=discord.Intents.default())

def load_bomb_locations():
    if os.path.exists(data):
        with open(data, "r") as file:
            return [line.strip() for line in file.readlines()]
    else:
        return []

def save_bomb_locations(locations):
    with open(data, "w") as file:
        for location in locations:
            file.write(location + "\n")

def add_bomb_location(location):
    bomb_locations.append(location)
    save_bomb_locations(bomb_locations)

def preprocessing(user_input):
    tokens = word_tokenize(user_input)
    return ' '.join(tokens)

def predict_bomb_location(data, user_input):
    if not data:
        return "No bomb locations added yet."

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data)
    user_input = preprocessing(user_input)
    user_input_vectorized = vectorizer.transform([user_input])
    similarity_scores = np.dot(X, user_input_vectorized.T)
    most_similar_index = np.argmax(similarity_scores)
    return max(1, min(most_similar_index + 1, 25))


def display_grid(predicted_location):
    grid_size = 5
    if isinstance(predicted_location, str):
        return predicted_location
    else:
        grid = [["💎" for _ in range(grid_size)] for _ in range(grid_size)]
        row = (predicted_location - 1) // grid_size
        col = (predicted_location - 1) % grid_size
        if 0 <= row < grid_size and 0 <= col < grid_size:
            grid[row][col] = "💣"
            result = ""
            for r in range(grid_size):
                for c in range(grid_size):
                    result += grid[r][c] + " "
                result += "\n"
            return result
        else:
            return "Predicted location is outside the grid."

@bot.slash_command(name='addbomb', description='Add a bomb location')
async def add_bomb(ctx, location: str):
    add_bomb_location(location)
    await ctx.respond(f'Bomb location "{location}" added!')

@bot.slash_command(name='predict', description='Predict the bomb location')
async def predict(ctx, user_input: str):
    predicted_location = predict_bomb_location(bomb_locations, user_input)
    grid_message = display_grid(predicted_location)
    await ctx.respond(f'Predicted Bomb location:\n{grid_message}')

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

bot.run(TOKEN)
