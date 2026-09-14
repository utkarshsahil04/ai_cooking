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

class HealthyVegetableDishRecommender:
    def __init__(self):
        """Initialize the vegetable dish recommender."""
        self.vegetable_classifier = self._build_vegetable_classifier()
        self.vegetable_labels = self._load_vegetable_labels()
        self.healthy_recipe_database = self._load_healthy_recipe_database()
        self.nutrition_info = self._load_nutrition_info()
        
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
    
    def _load_healthy_recipe_database(self):
        """Load database of healthy vegetable recipes."""
        return {
            "roasted vegetables": {
                "base_recipe": "1. Preheat oven to 425°F. 2. Chop vegetables into similar size pieces. 3. Toss with olive oil, salt, and pepper. 4. Roast for 25-30 minutes, stirring halfway through.",
                "health_benefits": "Low in calories, high in fiber and nutrients. Roasting enhances flavors while maintaining nutrients.",
                "variations": {
                    "mediterranean": "Add garlic, rosemary, and a squeeze of lemon juice after roasting.",
                    "spicy": "Add paprika, cumin, and chili flakes before roasting.",
                    "herb garden": "Add fresh thyme, oregano, and basil after roasting."
                }
            },
            "vegetable stir fry": {
                "base_recipe": "1. Heat a wok or large pan over high heat. 2. Add a small amount of oil. 3. Add vegetables starting with the firmest ones. 4. Cook for 5-7 minutes, stirring constantly. 5. Add preferred sauce and cook for additional 1-2 minutes.",
                "health_benefits": "Quick cooking preserves nutrients and texture. Can be made with minimal oil.",
                "variations": {
                    "asian": "Use soy sauce, ginger, and garlic as seasonings.",
                    "curry": "Add curry powder or paste and a splash of coconut milk.",
                    "simple": "Season with just salt, pepper, and a squeeze of lemon juice."
                }
            },
            "vegetable soup": {
                "base_recipe": "1. Sauté onions and garlic in a large pot. 2. Add chopped vegetables and sauté for a few minutes. 3. Add vegetable broth. 4. Simmer for 20-30 minutes until vegetables are tender. 5. Season with herbs and spices of choice.",
                "health_benefits": "Hydrating, filling, and nutrient-dense. Great way to use multiple vegetables.",
                "variations": {
                    "pureed": "Blend soup after cooking for a creamy texture without cream.",
                    "minestrone": "Add beans and a small amount of pasta near the end of cooking.",
                    "clear broth": "Keep vegetables in larger pieces and use a clear, light broth."
                }
            },
            "vegetable salad": {
                "base_recipe": "1. Wash and chop vegetables. 2. Make a simple dressing with olive oil, acid (lemon juice or vinegar), salt, and pepper. 3. Toss vegetables with dressing. 4. Optional: add nuts, seeds, or a small amount of cheese.",
                "health_benefits": "Raw vegetables retain all nutrients and enzymes. High in fiber, low in calories.",
                "variations": {
                    "greek": "Add feta cheese, olives, and oregano.",
                    "asian slaw": "Use rice vinegar, sesame oil, and add sesame seeds.",
                    "southwest": "Add black beans, corn, and lime juice."
                }
            },
            "vegetable curry": {
                "base_recipe": "1. Sauté onions, garlic, and spices. 2. Add vegetables and cook for a few minutes. 3. Add liquid (coconut milk, tomatoes, or broth). 4. Simmer until vegetables are tender, about 20 minutes. 5. Serve with rice or flatbread.",
                "health_benefits": "Spices like turmeric and cumin have anti-inflammatory properties. Can incorporate many vegetables.",
                "variations": {
                    "indian": "Use traditional curry spices like turmeric, coriander, and cumin.",
                    "thai": "Use Thai curry paste and coconut milk.",
                    "jamaican": "Use allspice, thyme, and scotch bonnet pepper."
                }
            },
            "stuffed vegetables": {
                "base_recipe": "1. Hollow out vegetables like bell peppers, zucchini, or eggplant. 2. Prepare a filling with grains, beans, and/or smaller diced vegetables. 3. Fill the vegetables. 4. Bake at 375°F until the vegetables are tender, about 25-35 minutes.",
                "health_benefits": "Portion-controlled and visually appealing. Great way to use multiple vegetables.",
                "variations": {
                    "mediterranean": "Use quinoa, feta, and olives as filling.",
                    "mexican": "Use rice, black beans, corn, and taco seasoning.",
                    "italian": "Use brown rice, diced tomatoes, and Italian herbs."
                }
            }
        }
    
    def _load_nutrition_info(self):
        """Load nutrition information for vegetables."""
        return {
            "tomato": {
                "calories": 22,
                "protein": 1.1,
                "fiber": 1.5,
                "vitamin_c": "27% DV",
                "vitamin_a": "17% DV",
                "potassium": "9% DV",
                "benefits": "High in lycopene, an antioxidant linked to reduced risk of heart disease and cancer."
            },
            "onion": {
                "calories": 40,
                "protein": 1.1,
                "fiber": 1.7,
                "vitamin_c": "12% DV",
                "folate": "5% DV",
                "potassium": "4% DV",
                "benefits": "Contains quercetin, an antioxidant with anti-inflammatory properties."
            },
            "garlic": {
                "calories": 4,
                "protein": 0.2,
                "fiber": 0.1,
                "vitamin_c": "2% DV",
                "vitamin_b6": "2% DV",
                "manganese": "2% DV",
                "benefits": "Contains allicin, which may help lower blood pressure and cholesterol."
            },
            "carrot": {
                "calories": 41,
                "protein": 0.9,
                "fiber": 2.8,
                "vitamin_a": "428% DV",
                "vitamin_k": "13% DV",
                "potassium": "12% DV",
                "benefits": "Excellent source of beta-carotene, which supports eye health."
            },
            "broccoli": {
                "calories": 55,
                "protein": 3.7,
                "fiber": 5.1,
                "vitamin_c": "135% DV",
                "vitamin_k": "116% DV",
                "folate": "14% DV",
                "benefits": "High in sulforaphane, a compound that may have anti-cancer properties."
            },
            "spinach": {
                "calories": 23,
                "protein": 2.9,
                "fiber": 2.2,
                "vitamin_a": "56% DV",
                "vitamin_c": "14% DV",
                "iron": "15% DV",
                "benefits": "Rich in iron and antioxidants, supports healthy blood and immune function."
            }
            # Add more vegetables as needed
        }
    
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
    
    def fetch_online_recipes(self, vegetables):
        """
        Fetch healthy recipes from online sources based on detected vegetables.
        
        In a real application, this would connect to an API like Spoonacular, Edamam, or Recipe Puppy.
        For this example, we'll simulate the API response.
        """
        vegetable_list = ", ".join([veg[0] for veg in vegetables])
        
        # Simulate API response
        online_recipes = [
            {
                "title": f"Healthy {vegetable_list.split(', ')[0].capitalize()} Salad",
                "ingredients": [veg[0] for veg in vegetables] + ["olive oil", "lemon juice", "salt", "pepper"],
                "instructions": "1. Wash and chop all vegetables. 2. Mix with olive oil, lemon juice, salt and pepper. 3. Serve immediately.",
                "cuisine": "Mediterranean",
                "calories": 150,
                "protein": 3,
                "carbs": 15,
                "fat": 9,
                "url": "https://example.com/recipes/healthy-salad"
            },
            {
                "title": f"Roasted {vegetable_list.split(', ')[0].capitalize()} and {vegetable_list.split(', ')[1].capitalize() if len(vegetable_list.split(', ')) > 1 else 'Herbs'}",
                "ingredients": [veg[0] for veg in vegetables] + ["olive oil", "rosemary", "thyme", "salt", "pepper"],
                "instructions": "1. Preheat oven to 425°F. 2. Chop vegetables into bite-sized pieces. 3. Toss with olive oil, herbs, salt and pepper. 4. Roast for 25 minutes.",
                "cuisine": "Italian",
                "calories": 180,
                "protein": 4,
                "carbs": 20,
                "fat": 10,
                "url": "https://example.com/recipes/roasted-vegetables"
            },
            {
                "title": f"{vegetable_list.split(', ')[0].capitalize()} Stir Fry",
                "ingredients": [veg[0] for veg in vegetables] + ["soy sauce", "garlic", "ginger", "brown rice"],
                "instructions": "1. Cook brown rice according to package instructions. 2. Stir fry vegetables with garlic and ginger. 3. Add soy sauce. 4. Serve over rice.",
                "cuisine": "Asian",
                "calories": 250,
                "protein": 6,
                "carbs": 45,
                "fat": 5,
                "url": "https://example.com/recipes/vegetable-stir-fry"
            }
        ]
        
        return online_recipes
    
    def get_health_benefits(self, vegetables):
        """Get health benefits for the detected vegetables."""
        benefits = {}
        for veg, _ in vegetables:
            if veg in self.nutrition_info:
                benefits[veg] = {
                    "nutrition": {
                        "calories": self.nutrition_info[veg]["calories"],
                        "protein": self.nutrition_info[veg]["protein"],
                        "fiber": self.nutrition_info[veg]["fiber"]
                    },
                    "benefits": self.nutrition_info[veg]["benefits"]
                }
        return benefits
    
    def recommend_dishes(self, vegetables):
        """Recommend healthy dishes based on detected vegetables."""
        # Get recommended dish types based on vegetables
        recommended_dishes = []
        
        for dish_type, dish_info in self.healthy_recipe_database.items():
            recommended_dishes.append({
                "dish_type": dish_type,
                "base_recipe": dish_info["base_recipe"],
                "health_benefits": dish_info["health_benefits"],
                "variations": list(dish_info["variations"].keys())
            })
        
        return recommended_dishes
    
    def analyze_and_recommend(self, image_path):
        """Main function to analyze a vegetable image and return healthy dish recommendations."""
        # Preprocess the image
        preprocessed_image = self.preprocess_image(image_path)
        
        # Detect vegetables
        vegetables = self.predict_vegetables(preprocessed_image)
        
        # Get health benefits
        health_benefits = self.get_health_benefits(vegetables)
        
        # Recommend dishes
        recommended_dishes = self.recommend_dishes(vegetables)
        
        # Fetch online recipes
        online_recipes = self.fetch_online_recipes(vegetables)
        
        # Prepare result
        result = {
            "detected_vegetables": [
                {"name": veg[0], "confidence": float(veg[1])} 
                for veg in vegetables
            ],
            "health_benefits": health_benefits,
            "recommended_dishes": recommended_dishes,
            "online_recipes": online_recipes
        }
        
        return result

# Function to display the results in a more user-friendly format
def display_results(result):
    print("\n===== HEALTHY VEGETABLE DISH RECOMMENDER =====")
    
    print("\n🍅 DETECTED VEGETABLES:")
    for veg in result["detected_vegetables"]:
        print(f"- {veg['name'].capitalize()} (Confidence: {veg['confidence']:.2f})")
    
    print("\n💪 HEALTH BENEFITS:")
    for veg, info in result["health_benefits"].items():
        print(f"\n{veg.capitalize()}:")
        print(f"  • Calories: {info['nutrition']['calories']} kcal")
        print(f"  • Protein: {info['nutrition']['protein']}g")
        print(f"  • Fiber: {info['nutrition']['fiber']}g")
        print(f"  • Benefits: {info['benefits']}")
    
    print("\n🍲 RECOMMENDED DISH TYPES:")
    for dish in result["recommended_dishes"]:
        print(f"\n{dish['dish_type'].capitalize()}:")
        print(f"  • Health Benefits: {dish['health_benefits']}")
        print(f"  • Variations: {', '.join(dish['variations'])}")
    
    print("\n🌐 ONLINE RECIPES:")
    for recipe in result["online_recipes"]:
        print(f"\n{recipe['title']} ({recipe['cuisine']} cuisine)")
        print(f"  • Calories: {recipe['calories']} kcal | Protein: {recipe['protein']}g | Carbs: {recipe['carbs']}g | Fat: {recipe['fat']}g")
        print(f"  • Ingredients: {', '.join(recipe['ingredients'])}")
        print(f"  • URL: {recipe['url']}")

# Example usage
def main():
    # Initialize the model
    recommender = HealthyVegetableDishRecommender()
    
    # Example image path
    image_path = "OIP.jpeg"
    
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