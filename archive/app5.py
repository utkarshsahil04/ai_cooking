# Improved AI Cooking System (Beginner Friendly)

import random

def get_user_ingredients():
    ingredients = input("Enter the ingredients you have, separated by commas: ")
    return [item.strip().lower() for item in ingredients.split(",")]


def score_recipe(user_ingredients, recipe_ingredients):
    score = 0
    for ingredient in recipe_ingredients:
        if ingredient in user_ingredients:
            score += 2  # Good match
        else:
            score -= 1  # Missing important ingredient
    return score


def find_best_recipe(user_ingredients, recipes_db):
    best_match = None
    best_score = float('-inf')

    for recipe in recipes_db:
        score = score_recipe(user_ingredients, recipe['ingredients'])
        if score > best_score:
            best_score = score
            best_match = recipe

    return best_match, best_score


def suggest_random_recipe():
    random_recipes = [
        {
            "name": "Mystery Stir-Fry",
            "ingredients": ["any vegetables", "oil", "salt", "pepper"],
            "steps": [
                "Chop all vegetables.",
                "Heat oil in a pan.",
                "Stir-fry vegetables until cooked.",
                "Add salt and pepper to taste."
            ]
        },
        {
            "name": "Quick Veggie Salad",
            "ingredients": ["lettuce", "tomato", "cucumber", "olive oil", "lemon"],
            "steps": [
                "Chop all vegetables.",
                "Mix with olive oil and lemon.",
                "Serve fresh."
            ]
        }
    ]
    return random.choice(random_recipes)


def display_recipe(recipe):
    print(f"\nHere's a recipe you can try: {recipe['name']}")
    print("Ingredients:")
    for ingredient in recipe['ingredients']:
        print(f"- {ingredient}")
    print("\nSteps:")
    for idx, step in enumerate(recipe['steps'], 1):
        print(f"{idx}. {step}")


def main():
    # Expanded recipe database
    recipes = [
        {
            "name": "Chicken Fried Rice",
            "ingredients": ["chicken", "rice", "onion", "soy sauce", "egg"],
            "steps": [
                "Cook the rice and let it cool.",
                "Stir-fry chopped chicken and onion.",
                "Add rice and soy sauce.",
                "Push everything to the side, scramble an egg, then mix together."
            ]
        },
        {
            "name": "Tomato Pasta",
            "ingredients": ["pasta", "tomato", "garlic", "olive oil", "basil"],
            "steps": [
                "Boil pasta until al dente.",
                "Cook garlic in olive oil.",
                "Add chopped tomato and basil to the pan.",
                "Mix sauce with pasta and serve."
            ]
        },
        {
            "name": "Veggie Omelette",
            "ingredients": ["egg", "onion", "tomato", "spinach", "cheese"],
            "steps": [
                "Beat eggs in a bowl.",
                "Add chopped onion, tomato, and spinach.",
                "Pour into a heated pan and cook until set.",
                "Sprinkle cheese before folding the omelette."
            ]
        },
        {
            "name": "Grilled Cheese Sandwich",
            "ingredients": ["bread", "cheese", "butter"],
            "steps": [
                "Butter the bread slices.",
                "Place cheese between two slices.",
                "Grill until golden brown."
            ]
        },
        {
            "name": "Fruit Smoothie",
            "ingredients": ["banana", "milk", "honey", "berries"],
            "steps": [
                "Add all ingredients to a blender.",
                "Blend until smooth.",
                "Serve chilled."
            ]
        },
        {
            "name": "Simple Soup",
            "ingredients": ["water", "salt", "vegetables", "pepper"],
            "steps": [
                "Boil water.",
                "Add chopped vegetables.",
                "Season with salt and pepper.",
                "Simmer until vegetables are soft."
            ]
        }
    ]

    print("Welcome to the AI Cooking Assistant!")
    user_ingredients = get_user_ingredients()
    best_recipe, best_score = find_best_recipe(user_ingredients, recipes)

    if best_score > 0:
        display_recipe(best_recipe)
    else:
        print("\nNo perfect recipe found with your ingredients.")
        print("Here's a creative recipe you can try:")
        display_recipe(suggest_random_recipe())


if __name__ == "__main__":
    main()