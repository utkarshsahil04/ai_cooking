import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
import cv2
import json
from PIL import Image
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class APIBasedVegetableDishRecommender:
    def __init__(self):
        """Initialize the vegetable dish recommender with API integration."""
        self.vegetable_classifier = self._build_vegetable_classifier()
        self.vegetable_labels = self._load_vegetable_labels()
        
        # Load API keys from environment variables
        self.spoonacular_api_key = os.getenv("SPOONACULAR_API_KEY")
        self.edamam_app_id = os.getenv("EDAMAM_APP_ID")
        self.edamam_app_key = os.getenv("EDAMAM_APP_KEY")
        self.nutritionix_app_id = os.getenv("NUTRITIONIX_APP_ID")
        self.nutritionix_app_key = os.getenv("NUTRITIONIX_APP_KEY")
        
    def _build_vegetable_classifier(self):
        """Build the vegetable detection model based on EfficientNet."""
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Freeze the base model
        base_model.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(50, activation='sigmoid')(x)  # Multi-label classification for vegetables
        
        return Model(inputs=base_model.input, outputs=predictions)
    
    def _load_vegetable_labels(self):
        """Load vegetable labels."""
        return [
            "tomato", "onion", "garlic", "carrot", "broccoli", "spinach", "kale", 
            "bell pepper", "cucumber", "zucchini", "eggplant", "potato", "sweet potato",
            "cabbage", "lettuce", "cauliflower", "green beans", "peas", "corn", 
            "asparagus", "mushroom", "radish", "beetroot", "celery", "leek"
        ]
    
    def preprocess_image(self, image_path):
        """Preprocess the image for model input."""
        # Load and resize image
        if isinstance(image_path, str):
            img = Image.open(image_path)
            img = img.resize((224, 224))
            img_array = np.array(img)
        else:
            # Assume it's already a numpy array or similar
            img_array = cv2.resize(image_path, (224, 224))
        
        # Convert to RGB if grayscale
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Normalize
        img_array = img_array.astype(np.float32) / 255.0
        
        # Expand dimensions for batch processing
        return np.expand_dims(img_array, axis=0)
    
    def predict_vegetables(self, preprocessed_image):
        """Predict vegetables from the image."""
        # In a real model, this would use the trained weights
        # For this example, we'll simulate predictions
        
        # Simulate detection of 2-5 random vegetables
        num_vegetables = np.random.randint(2, 6)
        vegetable_indices = np.random.choice(
            len(self.vegetable_labels), 
            size=num_vegetables, 
            replace=False
        )
        
        detected_vegetables = [self.vegetable_labels[idx] for idx in vegetable_indices]
        confidence_scores = np.random.uniform(0.6, 0.95, size=num_vegetables)
        
        return list(zip(detected_vegetables, confidence_scores))
    
    def get_nutrition_data_from_api(self, vegetable):
        """
        Get nutrition data from Nutritionix API.
        """
        if not self.nutritionix_app_id or not self.nutritionix_app_key:
            return self._get_fallback_nutrition_data(vegetable)
            
        try:
            headers = {
                "x-app-id": self.nutritionix_app_id,
                "x-app-key": self.nutritionix_app_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "query": vegetable,
                "detailed": True
            }
            
            response = requests.post(
                "https://trackapi.nutritionix.com/v2/natural/nutrients",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                if "foods" in result and len(result["foods"]) > 0:
                    food = result["foods"][0]
                    return {
                        "name": vegetable,
                        "calories": food.get("nf_calories", 0),
                        "protein": food.get("nf_protein", 0),
                        "fiber": food.get("nf_dietary_fiber", 0),
                        "carbs": food.get("nf_total_carbohydrate", 0),
                        "fat": food.get("nf_total_fat", 0),
                        "vitamin_a": food.get("full_nutrients", {}).get("318", 0),  # Vitamin A
                        "vitamin_c": food.get("full_nutrients", {}).get("401", 0),  # Vitamin C
                        "potassium": food.get("full_nutrients", {}).get("306", 0),  # Potassium
                        "serving_weight_grams": food.get("serving_weight_grams", 100)
                    }
            
            return self._get_fallback_nutrition_data(vegetable)
            
        except Exception as e:
            print(f"Error fetching nutrition data: {e}")
            return self._get_fallback_nutrition_data(vegetable)
    
    def _get_fallback_nutrition_data(self, vegetable):
        """Fallback nutrition data when API fails."""
        # Basic nutrition data for common vegetables
        nutrition_data = {
            "tomato": {"calories": 22, "protein": 1.1, "fiber": 1.5, "carbs": 4.8, "fat": 0.2},
            "onion": {"calories": 40, "protein": 1.1, "fiber": 1.7, "carbs": 9.3, "fat": 0.1},
            "garlic": {"calories": 149, "protein": 6.4, "fiber": 2.1, "carbs": 33.1, "fat": 0.5},
            "carrot": {"calories": 41, "protein": 0.9, "fiber": 2.8, "carbs": 9.6, "fat": 0.2},
            "broccoli": {"calories": 34, "protein": 2.8, "fiber": 2.6, "carbs": 6.6, "fat": 0.4},
            "spinach": {"calories": 23, "protein": 2.9, "fiber": 2.2, "carbs": 3.6, "fat": 0.4},
            "bell pepper": {"calories": 31, "protein": 1.0, "fiber": 2.1, "carbs": 6.0, "fat": 0.3}
        }
        
        if vegetable in nutrition_data:
            return {"name": vegetable, **nutrition_data[vegetable]}
        
        # Generic data for unknown vegetables
        return {
            "name": vegetable,
            "calories": 30,
            "protein": 1.5,
            "fiber": 2.0,
            "carbs": 5.0,
            "fat": 0.2,
            "note": "Estimated values"
        }
    
    def fetch_recipes_from_spoonacular(self, vegetables):
        """
        Fetch recipes from Spoonacular API based on detected vegetables.
        """
        if not self.spoonacular_api_key:
            return self._get_fallback_recipes(vegetables)
            
        try:
            vegetable_list = ",".join([veg[0] for veg in vegetables])
            
            params = {
                "apiKey": self.spoonacular_api_key,
                "ingredients": vegetable_list,
                "number": 5,
                "ranking": 2,  # Maximize used ingredients
                "ignorePantry": True,
                "tags": "vegetarian,healthy"
            }
            
            response = requests.get(
                "https://api.spoonacular.com/recipes/findByIngredients",
                params=params
            )
            
            if response.status_code == 200:
                recipes = response.json()
                detailed_recipes = []
                
                for recipe in recipes[:3]:  # Limit to top 3 recipes to avoid API rate limits
                    recipe_id = recipe["id"]
                    
                    # Get detailed recipe information
                    recipe_info_response = requests.get(
                        f"https://api.spoonacular.com/recipes/{recipe_id}/information",
                        params={"apiKey": self.spoonacular_api_key}
                    )
                    
                    if recipe_info_response.status_code == 200:
                        recipe_info = recipe_info_response.json()
                        
                        detailed_recipes.append({
                            "title": recipe_info["title"],
                            "ingredients": [ingredient["original"] for ingredient in recipe_info.get("extendedIngredients", [])],
                            "instructions": self._parse_instructions(recipe_info.get("instructions", "")),
                            "image": recipe_info.get("image", ""),
                            "source_url": recipe_info.get("sourceUrl", ""),
                            "ready_in_minutes": recipe_info.get("readyInMinutes", 0),
                            "servings": recipe_info.get("servings", 0),
                            "health_score": recipe_info.get("healthScore", 0),
                            "diets": recipe_info.get("diets", []),
                            "nutrition": self._parse_nutrition(recipe_info.get("nutrition", {}))
                        })
                
                if detailed_recipes:
                    return detailed_recipes
            
            return self._get_fallback_recipes(vegetables)
            
        except Exception as e:
            print(f"Error fetching recipes: {e}")
            return self._get_fallback_recipes(vegetables)
    
    def _parse_instructions(self, instructions):
        """Parse recipe instructions into a list."""
        if not instructions:
            return []
            
        # Handle HTML instructions
        if "<ol>" in instructions or "<li>" in instructions:
            # Very simple HTML parsing, would need a proper HTML parser in production
            steps = []
            for step in instructions.split("<li>"):
                if "</li>" in step:
                    step_text = step.split("</li>")[0].strip()
                    if step_text:
                        steps.append(step_text)
            return steps
        
        # Handle numbered instructions
        if "1." in instructions and "2." in instructions:
            steps = []
            current_step = ""
            for line in instructions.split("\n"):
                line = line.strip()
                if not line:
                    continue
                    
                if line[0].isdigit() and "." in line[:3]:
                    if current_step:
                        steps.append(current_step.strip())
                    current_step = line
                else:
                    current_step += " " + line
                    
            if current_step:
                steps.append(current_step.strip())
                
            return steps
        
        # Handle paragraph format
        return [instructions]
    
    def _parse_nutrition(self, nutrition_data):
        """Parse nutrition data from Spoonacular."""
        if not nutrition_data:
            return {}
            
        nutrients = nutrition_data.get("nutrients", [])
        result = {}
        
        for nutrient in nutrients:
            name = nutrient.get("name", "").lower()
            if name in ["calories", "fat", "carbohydrates", "protein", "fiber"]:
                result[name] = {
                    "amount": nutrient.get("amount", 0),
                    "unit": nutrient.get("unit", "")
                }
                
        return result
    
    def fetch_recipes_from_edamam(self, vegetables):
        """
        Fetch recipes from Edamam API based on detected vegetables.
        """
        if not self.edamam_app_id or not self.edamam_app_key:
            return self._get_fallback_recipes(vegetables)
            
        try:
            vegetable_list = " ".join([veg[0] for veg in vegetables])
            
            params = {
                "app_id": self.edamam_app_id,
                "app_key": self.edamam_app_key,
                "q": vegetable_list,
                "health": "vegetarian",
                "diet": "balanced",
                "dishType": "Main course",
                "from": 0,
                "to": 5
            }
            
            response = requests.get(
                "https://api.edamam.com/search",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                hits = data.get("hits", [])
                
                recipes = []
                for hit in hits:
                    recipe = hit.get("recipe", {})
                    
                    recipes.append({
                        "title": recipe.get("label", ""),
                        "ingredients": recipe.get("ingredientLines", []),
                        "instructions": ["Visit the recipe URL for detailed instructions"],
                        "image": recipe.get("image", ""),
                        "source_url": recipe.get("url", ""),
                        "source": recipe.get("source", ""),
                        "servings": recipe.get("yield", 0),
                        "calories": recipe.get("calories", 0),
                        "total_time": recipe.get("totalTime", 0),
                        "diet_labels": recipe.get("dietLabels", []),
                        "health_labels": recipe.get("healthLabels", []),
                        "cautions": recipe.get("cautions", []),
                        "nutrients": self._parse_edamam_nutrients(recipe.get("totalNutrients", {}))
                    })
                
                if recipes:
                    return recipes
            
            return self._get_fallback_recipes(vegetables)
            
        except Exception as e:
            print(f"Error fetching recipes from Edamam: {e}")
            return self._get_fallback_recipes(vegetables)
    
    def _parse_edamam_nutrients(self, nutrients):
        """Parse nutrition data from Edamam API."""
        result = {}
        
        for key, nutrient in nutrients.items():
            if key in ["ENERC_KCAL", "FAT", "CHOCDF", "PROCNT", "FIBTG"]:
                name_map = {
                    "ENERC_KCAL": "calories",
                    "FAT": "fat",
                    "CHOCDF": "carbohydrates",
                    "PROCNT": "protein",
                    "FIBTG": "fiber"
                }
                
                result[name_map[key]] = {
                    "amount": nutrient.get("quantity", 0),
                    "unit": nutrient.get("unit", "")
                }
                
        return result
    
    def _get_fallback_recipes(self, vegetables):
        """Provide fallback recipes when API calls fail."""
        vegetable_list = [veg[0] for veg in vegetables]
        
        fallback_recipes = [
            {
                "title": f"Healthy {vegetable_list[0].capitalize()} Salad",
                "ingredients": vegetable_list + ["olive oil", "lemon juice", "salt", "pepper"],
                "instructions": [
                    "1. Wash and chop all vegetables.",
                    "2. Mix with olive oil, lemon juice, salt and pepper.",
                    "3. Serve immediately."
                ],
                "ready_in_minutes": 15,
                "servings": 2,
                "health_score": 95,
                "diets": ["vegetarian", "vegan", "gluten-free"],
                "note": "Fallback recipe - API data unavailable"
            },
            {
                "title": f"Roasted {vegetable_list[0].capitalize()} and {vegetable_list[1].capitalize() if len(vegetable_list) > 1 else 'Herbs'}",
                "ingredients": vegetable_list + ["olive oil", "rosemary", "thyme", "salt", "pepper"],
                "instructions": [
                    "1. Preheat oven to 425°F (220°C).",
                    "2. Chop vegetables into bite-sized pieces.",
                    "3. Toss with olive oil, herbs, salt and pepper.",
                    "4. Spread on baking sheet and roast for 25 minutes, turning halfway through."
                ],
                "ready_in_minutes": 35,
                "servings": 4,
                "health_score": 90,
                "diets": ["vegetarian", "vegan", "gluten-free"],
                "note": "Fallback recipe - API data unavailable"
            },
            {
                "title": f"{vegetable_list[0].capitalize()} Stir Fry",
                "ingredients": vegetable_list + ["soy sauce", "garlic", "ginger", "brown rice"],
                "instructions": [
                    "1. Cook brown rice according to package instructions.",
                    "2. Heat oil in a wok or large pan over high heat.",
                    "3. Add garlic and ginger, stir for 30 seconds.",
                    "4. Add vegetables and stir fry for 5-7 minutes until tender-crisp.",
                    "5. Add soy sauce to taste.",
                    "6. Serve over brown rice."
                ],
                "ready_in_minutes": 25,
                "servings": 2,
                "health_score": 85,
                "diets": ["vegetarian", "vegan"],
                "note": "Fallback recipe - API data unavailable"
            }
        ]
        
        return fallback_recipes
    
    def get_health_tips_for_vegetables(self, vegetables):
        """Get health tips for detected vegetables."""
        health_tips = {
            "tomato": "Rich in lycopene, which may reduce risk of heart disease and some cancers. Cook tomatoes to increase lycopene absorption.",
            "onion": "Contains quercetin, an antioxidant with anti-inflammatory properties. Regular consumption may help reduce blood pressure.",
            "garlic": "Contains allicin, which has antimicrobial properties. Let crushed garlic sit for 10 minutes before cooking to maximize benefits.",
            "carrot": "Excellent source of beta-carotene for eye health. Cooking carrots can actually increase nutrient availability.",
            "broccoli": "High in sulforaphane, which may have anti-cancer properties. Light steaming preserves more nutrients than boiling.",
            "spinach": "Rich in iron and antioxidants. Pair with vitamin C foods like lemon juice to increase iron absorption.",
            "kale": "Nutrient powerhouse with vitamin K, vitamin A, and calcium. Massage raw kale with oil to make it more tender and digestible.",
            "bell pepper": "More vitamin C than oranges. Red bell peppers contain more antioxidants than green ones.",
            "cucumber": "High water content makes them hydrating. Keep the skin on for added fiber and nutrients.",
            "zucchini": "Low in calories but high in water and fiber. Try using spiral-cut zucchini as a pasta alternative.",
            "eggplant": "Contains nasunin, an antioxidant that protects brain cell membranes. Salting before cooking can reduce bitterness.",
            "potato": "Good source of potassium and vitamin C. Keep the skin on for more fiber and nutrients.",
            "sweet potato": "Rich in beta-carotene and fiber. Orange varieties have more beta-carotene than lighter-colored ones.",
            "cabbage": "High in vitamin C and vitamin K. Fermented cabbage (sauerkraut) provides beneficial probiotics.",
            "cauliflower": "Versatile cruciferous vegetable that can substitute for rice, pizza crust, or mashed potatoes.",
            "asparagus": "Good source of folate and vitamin K. Quick cooking methods like steaming or grilling preserve nutrients best.",
            "mushroom": "One of the few vegetable sources of vitamin D, especially when exposed to sunlight before cooking.",
            "beetroot": "Contains nitrates that may help lower blood pressure and improve athletic performance."
        }
        
        tips = {}
        for veg, _ in vegetables:
            if veg in health_tips:
                tips[veg] = health_tips[veg]
            else:
                tips[veg] = "Rich in vitamins and minerals. Incorporate into a varied diet for maximum health benefits."
                
        return tips
    
    def analyze_and_recommend(self, image_path):
        """Main function to analyze a vegetable image and return healthy dish recommendations with API data."""
        # Preprocess the image
        preprocessed_image = self.preprocess_image(image_path)
        
        # Detect vegetables
        vegetables = self.predict_vegetables(preprocessed_image)
        
        # Get nutrition data from API
        nutrition_data = {}
        for veg, _ in vegetables:
            nutrition_data[veg] = self.get_nutrition_data_from_api(veg)
        
        # Get health tips
        health_tips = self.get_health_tips_for_vegetables(vegetables)
        
        # Choose which recipe API to use based on available keys
        if self.spoonacular_api_key:
            recipes = self.fetch_recipes_from_spoonacular(vegetables)
            recipe_source = "Spoonacular API"
        elif self.edamam_app_id and self.edamam_app_key:
            recipes = self.fetch_recipes_from_edamam(vegetables)
            recipe_source = "Edamam API"
        else:
            recipes = self._get_fallback_recipes(vegetables)
            recipe_source = "Fallback recipes (no API key available)"
        
        # Prepare result
        result = {
            "detected_vegetables": [
                {"name": veg[0], "confidence": float(veg[1])} 
                for veg in vegetables
            ],
            "nutrition_data": nutrition_data,
            "health_tips": health_tips,
            "recipes": recipes,
            "recipe_source": recipe_source
        }
        
        return result

# Function to display the results in a more user-friendly format
def display_results(result):
    print("\n===== HEALTHY VEGETABLE DISH RECOMMENDER =====")
    
    print("\n🍅 DETECTED VEGETABLES:")
    for veg in result["detected_vegetables"]:
        print(f"- {veg['name'].capitalize()} (Confidence: {veg['confidence']:.2f})")
    
    print("\n💪 NUTRITION DATA:")
    for veg, data in result["nutrition_data"].items():
        print(f"\n{veg.capitalize()}:")
        for key, value in data.items():
            if key != "name":
                print(f"  • {key.replace('_', ' ').capitalize()}: {value}")
    
    print("\n🌱 HEALTH TIPS:")
    for veg, tip in result["health_tips"].items():
        print(f"\n{veg.capitalize()}: {tip}")
    
    print(f"\n🍲 RECIPES (Source: {result['recipe_source']}):")
    for i, recipe in enumerate(result["recipes"], 1):
        print(f"\n{i}. {recipe['title']}")
        
        if "health_score" in recipe:
            print(f"  • Health Score: {recipe['health_score']}/100")
            
        if "diets" in recipe and recipe["diets"]:
            print(f"  • Suitable for: {', '.join(diet.capitalize() for diet in recipe['diets'])}")
            
        if "ready_in_minutes" in recipe:
            print(f"  • Ready in: {recipe['ready_in_minutes']} minutes")
            
        print("\n  Ingredients:")
        for ingredient in recipe["ingredients"][:10]:  # Limit to first 10 ingredients
            print(f"  - {ingredient}")
        
        if len(recipe["ingredients"]) > 10:
            print(f"  - ... and {len(recipe['ingredients']) - 10} more ingredients")
            
        print("\n  Instructions:")
        for i, step in enumerate(recipe["instructions"][:5], 1):  # Limit to first 5 steps
            print(f"  {step}")
            
        if len(recipe["instructions"]) > 5:
            print(f"  ... and {len(recipe['instructions']) - 5} more steps")
            
        if "source_url" in recipe and recipe["source_url"]:
            print(f"\n  Full recipe: {recipe['source_url']}")

# Example usage
def main():
    # Initialize the model
    recommender = APIBasedVegetableDishRecommender()
    
    # Example image path
    image_path = "vegetables.jpg"
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Demo mode: Generating sample output")
        result = recommender.analyze_and_recommend(np.random.rand(224, 224, 3))
    else:
        # Analyze the image
        result = recommender.analyze_and_recommend(image_path)
    
    # Display results
    display_results(result)

if __name__ == "__main__":
    main()