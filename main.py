


import discord
from discord.ext import commands
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import asyncio

nltk.download('punkt')

data = "data.txt"
activity = discord.Activity(type = discord.ActivityType.playing,name = "Stake Mines")

bot = commands.Bot(intents=discord.Intents.default(), help_command=None,activity = activity)


def load_bomb_locations():
    if os.path.exists(data):
        with open(data, "r") as file:
            return [line.strip() for line in file.readlines()]
    else:
        return []


def save_bomb_locations(locations):
    with open(data, "a") as file:
        for location in locations:
            file.write(location + "\n")


def preprocessing(user_input):
    tokens = word_tokenize(user_input)
    return ' '.join(tokens)


def predict_bomb_location(data_txt, user_input):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data_txt)
    user_input = preprocessing(user_input)
    user_input_vectorized = vectorizer.transform([user_input])
    similarity_scores = np.dot(X, user_input_vectorized.T)
    most_similar_index = np.argmax(similarity_scores)
    return min(most_similar_index, 25)


def display_grid(predicted_location):
    grid_size = 5
    grid = [["💎" for _ in range(grid_size)] for _ in range(grid_size)]
    row = (predicted_location) // grid_size
    col = (predicted_location) % grid_size
    if 0 <= row < grid_size and 0 <= col < grid_size:
        grid[row][col] = "💣"
        # Convert grid to a string
        grid_str = "\n".join(["".join(row) for row in grid])
        return grid_str
    else:
        return "Predicted location is outside the grid."


@bot.slash_command(name='predict', description='Predict the bomb location')
async def predict(ctx, your_last_bomb_location1: int, your_last_bomb_location2: int):
    save_bomb_locations([str(your_last_bomb_location1), str(your_last_bomb_location2)])  # Save bomb locations
    bomb_locations = load_bomb_locations()  # Load bomb locations
    predicted_location = predict_bomb_location(bomb_locations, f"{your_last_bomb_location1} {your_last_bomb_location2}")  # Predict bomb location
    grid_image = display_grid(predicted_location)  # Display grid
    thinking_msg = await ctx.respond("Thinking...")
    await asyncio.sleep(3)
    await thinking_msg.edit(content=f"Predicted Bomb location: \n{grid_image}")


@bot.event
async def on_message(message):
    if bot.user in message.mentions and 'help' in message.content.lower():
        embed = discord.Embed(
            title="Stake Predictor Bot",
            description=("Hello! I'm a stake predictor bot by @luffy_nub.\n\n"
                         "To use me, you need two of your previous games' bomb locations. It's very important to get the two last bets mine location.\n"
                         "I only work now for only 1 mine game.\n\n"
                         "To use me, type `/predict` and add the two last games' locations.\n\n"
                         "And yeah, remember it is done by ML, so predictions aren't accurate. It just can give you an idea where the bomb might be based on the previous location added by us to make sure to add only locations which are immediately played for better results...hehe"),
            color=discord.Color.blue()
        )
        embed.set_footer(text="Stake Mine Predictor Bot | Developed by @luffy_nub")

        await message.channel.send(embed=embed)

    await bot.process_commands(message)


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

bot.run('MTIyMTQ4OTY1MjY5NTk2MTcwMg.GE5ObN.UvUXHFhdLoLJndRknv3_59DDns-5WXWTHIK9ZI')
