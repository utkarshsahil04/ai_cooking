from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from ultralytics import YOLO
import random
from datetime import datetime 
import json
# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize YOLO model
model = YOLO('yolosaved_bestdataset_2_epoch25.pt')

# Recipe database
# Extended Recipe Database with 200+ recipes
RECIPE_DATABASE = {
    # Vegetable-based soups
    "Vegetable Soup": ["carrot", "potato", "cabbage", "onion", "tomato"],
    "Pumpkin Soup": ["pumpkin", "onion", "carrot", "potato"],
    "Carrot and Ginger Soup": ["carrot", "ginger", "onion", "potato"],
    "Broccoli Cheese Soup": ["broccoli", "cheese", "onion", "potato"],
    "Tomato Basil Soup": ["tomato", "basil", "onion", "garlic"],
    "Butternut Squash Soup": ["butternut squash", "onion", "carrot", "apple"],
    "Cabbage Soup": ["cabbage", "carrot", "onion", "tomato"],
    "Cauliflower Soup": ["cauliflower", "potato", "onion", "garlic"],
    "Gazpacho": ["tomato", "cucumber", "bell pepper", "onion", "garlic"],
    "Minestrone": ["tomato", "carrot", "celery", "onion", "zucchini", "beans"],
    "Split Pea Soup": ["split peas", "carrot", "onion", "potato", "celery"],
    "Leek and Potato Soup": ["leek", "potato", "onion", "garlic"],
    "Mushroom Soup": ["mushroom", "onion", "garlic", "thyme"],
    "Asparagus Soup": ["asparagus", "potato", "onion", "garlic"],
    "Beetroot Soup": ["beetroot", "potato", "onion", "carrot"],
    
    # Salads
    "Garden Salad": ["tomato", "cucumber", "carrot", "bell pepper", "lettuce"],
    "Greek Salad": ["tomato", "cucumber", "bell pepper", "onion", "olives", "feta"],
    "Caesar Salad": ["lettuce", "croutons", "parmesan"],
    "Coleslaw": ["cabbage", "carrot", "onion", "mayonnaise"],
    "Potato Salad": ["potato", "onion", "celery", "mayonnaise"],
    "Waldorf Salad": ["apple", "celery", "walnut", "grape"],
    "Tabbouleh": ["parsley", "tomato", "onion", "bulgur", "mint"],
    "Caprese Salad": ["tomato", "mozzarella", "basil"],
    "Nicoise Salad": ["tuna", "tomato", "potato", "green beans", "olives", "egg"],
    "Spinach Salad": ["spinach", "strawberry", "walnut", "feta"],
    "Broccoli Salad": ["broccoli", "bacon", "red onion", "sunflower seeds"],
    "Cucumber Salad": ["cucumber", "red onion", "tomato", "feta"],
    "Carrot and Raisin Salad": ["carrot", "raisin", "apple", "mayonnaise"],
    "Beet Salad": ["beetroot", "goat cheese", "walnut", "arugula"],
    "Corn Salad": ["corn", "bell pepper", "onion", "cilantro"],
    
    # Vegetable Stir Fries and Sautés
    "Vegetable Stir Fry": ["carrot", "bell pepper", "broccoli", "onion", "cabbage"],
    "Broccoli Stir Fry": ["broccoli", "garlic", "ginger", "carrot"],
    "Bell Pepper and Onion Sauté": ["bell pepper", "onion", "garlic"],
    "Zucchini Stir Fry": ["zucchini", "garlic", "onion", "tomato"],
    "Mushroom Sauté": ["mushroom", "garlic", "onion", "herbs"],
    "Asparagus Stir Fry": ["asparagus", "garlic", "ginger", "bell pepper"],
    "Green Bean Stir Fry": ["green beans", "garlic", "onion", "bell pepper"],
    "Cabbage Stir Fry": ["cabbage", "carrot", "onion", "garlic"],
    "Snow Pea Stir Fry": ["snow peas", "carrot", "bell pepper", "garlic"],
    "Bok Choy Stir Fry": ["bok choy", "garlic", "ginger", "carrot"],
    "Cauliflower Stir Fry": ["cauliflower", "peas", "carrot", "onion", "garlic"],
    "Spinach Sauté": ["spinach", "garlic", "onion", "lemon"],
    "Kale Sauté": ["kale", "garlic", "onion", "lemon"],
    "Sweet Potato Stir Fry": ["sweet potato", "bell pepper", "onion", "spinach"],
    "Eggplant Stir Fry": ["eggplant", "bell pepper", "onion", "garlic"],
    
    # Roasted Vegetables
    "Roasted Vegetables": ["carrot", "bell pepper", "broccoli", "onion", "potato"],
    "Roasted Broccoli": ["broccoli", "garlic", "lemon"],
    "Roasted Cauliflower": ["cauliflower", "garlic", "parmesan"],
    "Roasted Brussels Sprouts": ["brussels sprouts", "balsamic", "garlic"],
    "Roasted Carrots": ["carrot", "honey", "thyme"],
    "Roasted Bell Peppers": ["bell pepper", "garlic", "olive oil"],
    "Roasted Sweet Potatoes": ["sweet potato", "cinnamon", "rosemary"],
    "Roasted Asparagus": ["asparagus", "lemon", "parmesan"],
    "Roasted Beets": ["beetroot", "thyme", "balsamic"],
    "Roasted Potatoes": ["potato", "rosemary", "garlic"],
    "Roasted Zucchini": ["zucchini", "garlic", "parmesan"],
    "Roasted Tomatoes": ["tomato", "basil", "garlic"],
    "Roasted Eggplant": ["eggplant", "garlic", "herbs"],
    "Roasted Pumpkin": ["pumpkin", "cinnamon", "sage"],
    "Roasted Radishes": ["radish", "thyme", "butter"],
    
    # Casseroles and Bakes
    "Broccoli Cheese Bake": ["broccoli", "cheese", "onion", "garlic"],
    "Potato Gratin": ["potato", "cream", "cheese", "garlic"],
    "Cauliflower Cheese": ["cauliflower", "cheese", "milk", "butter"],
    "Ratatouille": ["tomato", "bell pepper", "onion", "zucchini", "eggplant"],
    "Vegetable Lasagna": ["zucchini", "bell pepper", "onion", "tomato", "cheese"],
    "Spinach and Feta Pie": ["spinach", "feta", "onion", "eggs"],
    "Tuna Casserole": ["tuna", "pasta", "peas", "cheese"],
    "Sweet Potato Casserole": ["sweet potato", "cinnamon", "brown sugar", "pecan"],
    "Green Bean Casserole": ["green beans", "mushroom", "onion"],
    "Corn Pudding": ["corn", "milk", "eggs", "butter"],
    "Zucchini Gratin": ["zucchini", "cheese", "garlic", "breadcrumbs"],
    "Eggplant Parmesan": ["eggplant", "tomato sauce", "mozzarella", "parmesan"],
    "Cabbage Rolls": ["cabbage", "rice", "onion", "tomato"],
    "Shepherd's Pie": ["potato", "peas", "carrot", "onion"],
    "Brussels Sprout Gratin": ["brussels sprouts", "cheese", "breadcrumbs", "bacon"],
    
    # Stuffed Vegetables
    "Stuffed Bell Peppers": ["bell pepper", "rice", "onion", "tomato"],
    "Stuffed Zucchini": ["zucchini", "breadcrumbs", "cheese", "herbs"],
    "Stuffed Tomatoes": ["tomato", "breadcrumbs", "herbs", "cheese"],
    "Stuffed Mushrooms": ["mushroom", "breadcrumbs", "cheese", "garlic"],
    "Stuffed Eggplant": ["eggplant", "onion", "tomato", "herbs"],
    "Stuffed Cabbage": ["cabbage", "rice", "onion", "tomato sauce"],
    "Stuffed Acorn Squash": ["acorn squash", "quinoa", "cranberry", "pecan"],
    "Stuffed Sweet Potatoes": ["sweet potato", "black beans", "corn", "cheese"],
    "Stuffed Butternut Squash": ["butternut squash", "quinoa", "cranberry", "feta"],
    "Stuffed Potato Skins": ["potato", "cheese", "bacon", "sour cream"],
    "Stuffed Pumpkin": ["pumpkin", "rice", "apple", "sausage"],
    "Stuffed Artichokes": ["artichoke", "breadcrumbs", "garlic", "parmesan"],
    "Stuffed Onions": ["onion", "breadcrumbs", "herbs", "cheese"],
    "Stuffed Portobello Mushrooms": ["portobello mushroom", "spinach", "cheese", "garlic"],
    "Stuffed Avocados": ["avocado", "corn", "tomato", "lime"],
    
    # Vegetable-based Main Dishes
    "Vegetable Curry": ["potato", "carrot", "peas", "tomato", "onion"],
    "Vegetable Stew": ["potato", "carrot", "onion", "celery", "tomato"],
    "Vegetable Paella": ["bell pepper", "peas", "tomato", "onion", "garlic"],
    "Vegetable Risotto": ["mushroom", "peas", "onion", "garlic", "parmesan"],
    "Vegetable Biryani": ["carrot", "peas", "potato", "onion", "rice"],
    "Vegetable Pasta": ["tomato", "zucchini", "bell pepper", "onion", "garlic"],
    "Vegetable Chili": ["beans", "bell pepper", "onion", "tomato", "corn"],
    "Vegetable Pot Pie": ["carrot", "peas", "potato", "onion", "celery"],
    "Vegetable Quiche": ["spinach", "onion", "cheese", "eggs"],
    "Vegetable Frittata": ["potato", "bell pepper", "onion", "spinach", "eggs"],
    "Vegetable Jambalaya": ["bell pepper", "celery", "onion", "tomato", "rice"],
    "Vegetable Fajitas": ["bell pepper", "onion", "zucchini", "tomato"],
    "Vegetable Quesadillas": ["bell pepper", "onion", "corn", "cheese"],
    "Vegetable Moussaka": ["eggplant", "potato", "onion", "tomato"],
    "Vegetable Tagine": ["carrot", "potato", "bell pepper", "onion", "zucchini"],
    
    # Sauces and Condiments
    "Tomato Sauce": ["tomato", "onion", "garlic", "basil"],
    "Pesto": ["basil", "garlic", "pine nuts", "parmesan"],
    "Guacamole": ["avocado", "tomato", "onion", "lime", "cilantro"],
    "Salsa": ["tomato", "onion", "cilantro", "jalapeño", "lime"],
    "Hummus": ["chickpeas", "garlic", "lemon", "tahini"],
    "Tzatziki": ["cucumber", "yogurt", "garlic", "dill"],
    "Chimichurri": ["parsley", "garlic", "red pepper flakes", "vinegar"],
    "Romesco Sauce": ["bell pepper", "tomato", "garlic", "almonds"],
    "Marinara Sauce": ["tomato", "garlic", "onion", "basil"],
    "Baba Ganoush": ["eggplant", "tahini", "garlic", "lemon"],
    "Aioli": ["garlic", "egg yolk", "lemon", "oil"],
    "Bechamel Sauce": ["milk", "butter", "flour", "nutmeg"],
    "Cranberry Sauce": ["cranberry", "orange", "sugar", "cinnamon"],
    "Apple Sauce": ["apple", "cinnamon", "sugar", "lemon"],
    "Mango Chutney": ["mango", "vinegar", "sugar", "ginger"],
    
    # Vegetable-based Desserts
    "Carrot Cake": ["carrot", "cinnamon", "walnut", "cream cheese"],
    "Pumpkin Pie": ["pumpkin", "cinnamon", "nutmeg", "cloves"],
    "Zucchini Bread": ["zucchini", "cinnamon", "walnut", "vanilla"],
    "Sweet Potato Pie": ["sweet potato", "cinnamon", "nutmeg", "vanilla"],
    "Beet Chocolate Cake": ["beetroot", "chocolate", "flour", "sugar"],
    "Avocado Chocolate Mousse": ["avocado", "chocolate", "honey", "vanilla"],
    "Apple Crisp": ["apple", "cinnamon", "oats", "brown sugar"],
    "Pear Tart": ["pear", "cinnamon", "almond", "sugar"],
    "Rhubarb Crumble": ["rhubarb", "strawberry", "sugar", "oats"],
    "Banana Bread": ["banana", "cinnamon", "walnut", "vanilla"],
    "Coconut Macaroons": ["coconut", "egg white", "sugar", "vanilla"],
    "Peach Cobbler": ["peach", "cinnamon", "sugar", "flour"],
    "Berry Crisp": ["berry", "lemon", "sugar", "oats"],
    "Lemon Bars": ["lemon", "butter", "sugar", "flour"],
    "Orange Cake": ["orange", "almond", "sugar", "egg"],
    
    # Breakfast Dishes
    "Vegetable Omelette": ["tomato", "bell pepper", "onion", "cheese"],
    "Vegetable Frittata": ["potato", "spinach", "onion", "bell pepper"],
    "Vegetable Hash": ["potato", "bell pepper", "onion", "garlic"],
    "Vegetable Breakfast Burrito": ["potato", "bell pepper", "onion", "tomato"],
    "Vegetable Quiche": ["broccoli", "spinach", "cheese", "onion"],
    "Avocado Toast": ["avocado", "tomato", "lemon", "bread"],
    "Smoothie Bowl": ["banana", "strawberry", "blueberry", "spinach"],
    "Green Smoothie": ["spinach", "banana", "apple", "avocado"],
    "Breakfast Potato": ["potato", "bell pepper", "onion", "paprika"],
    "Sweet Potato Hash": ["sweet potato", "bell pepper", "onion", "egg"],
    "Breakfast Skillet": ["potato", "bell pepper", "onion", "tomato"],
    "Vegetable Breakfast Sandwich": ["tomato", "avocado", "spinach", "cheese"],
    "Fruit Parfait": ["yogurt", "strawberry", "blueberry", "granola"],
    "Apple Cinnamon Oatmeal": ["apple", "cinnamon", "oats", "honey"],
    "Banana Pancakes": ["banana", "cinnamon", "egg", "flour"],
    
    # Sandwiches and Wraps
    "Veggie Sandwich": ["tomato", "cucumber", "lettuce", "bell pepper"],
    "Hummus Wrap": ["hummus", "carrot", "cucumber", "bell pepper"],
    "Grilled Vegetable Panini": ["zucchini", "bell pepper", "eggplant", "cheese"],
    "Avocado BLT": ["avocado", "tomato", "lettuce", "bacon"],
    "Cucumber Sandwich": ["cucumber", "cream cheese", "dill", "bread"],
    "Falafel Wrap": ["falafel", "tomato", "cucumber", "lettuce"],
    "Caprese Sandwich": ["tomato", "mozzarella", "basil", "bread"],
    "Egg Salad Sandwich": ["egg", "mayonnaise", "mustard", "lettuce"],
    "Tuna Salad Sandwich": ["tuna", "mayonnaise", "celery", "onion"],
    "Roasted Vegetable Wrap": ["bell pepper", "zucchini", "eggplant", "hummus"],
    "Mediterranean Wrap": ["cucumber", "tomato", "feta", "olive"],
    "Buffalo Cauliflower Wrap": ["cauliflower", "buffalo sauce", "lettuce", "ranch"],
    "Pesto Vegetable Sandwich": ["pesto", "tomato", "mozzarella", "spinach"],
    "Veggie Burger": ["lettuce", "tomato", "onion", "pickle"],
    "Apple and Cheese Sandwich": ["apple", "cheddar", "mustard", "arugula"],
    
    # Snacks and Appetizers
    "Vegetable Platter": ["carrot", "celery", "bell pepper", "cucumber"],
    "Caprese Skewers": ["tomato", "mozzarella", "basil"],
    "Stuffed Mushrooms": ["mushroom", "garlic", "breadcrumbs", "cheese"],
    "Bruschetta": ["tomato", "basil", "garlic", "bread"],
    "Vegetable Spring Rolls": ["carrot", "cabbage", "bell pepper", "bean sprouts"],
    "Spinach Artichoke Dip": ["spinach", "artichoke", "cream cheese", "parmesan"],
    "Buffalo Cauliflower Bites": ["cauliflower", "buffalo sauce", "garlic", "flour"],
    "Potato Skins": ["potato", "cheese", "bacon", "sour cream"],
    "Zucchini Fritters": ["zucchini", "flour", "egg", "garlic"],
    "Cheese and Fruit Platter": ["apple", "grape", "cheese", "cracker"],
    "Cucumber Bites": ["cucumber", "cream cheese", "dill", "salmon"],
    "Jalapeño Poppers": ["jalapeño", "cream cheese", "cheddar", "bacon"],
    "Deviled Eggs": ["egg", "mayonnaise", "mustard", "paprika"],
    "Vegetable Quesadilla": ["bell pepper", "onion", "cheese", "tortilla"],
    "Onion Rings": ["onion", "flour", "egg", "breadcrumbs"],
    
    # International Dishes
    "Vegetable Curry": ["potato", "peas", "carrot", "tomato", "onion"],
    "Pad Thai": ["bean sprouts", "green onion", "carrot", "peanut"],
    "Falafel": ["chickpea", "parsley", "garlic", "cumin"],
    "Vegetable Sushi": ["carrot", "cucumber", "avocado", "rice"],
    "Tabbouleh": ["parsley", "tomato", "cucumber", "bulgur"],
    "Vegetable Tempura": ["sweet potato", "bell pepper", "broccoli", "eggplant"],
    "Vegetable Enchiladas": ["bell pepper", "onion", "corn", "beans"],
    "Greek Spanakopita": ["spinach", "feta", "onion", "dill"],
    "Indian Samosas": ["potato", "peas", "onion", "curry"],
    "Thai Green Curry": ["bell pepper", "eggplant", "bamboo shoots", "basil"],
    "Chinese Stir Fry": ["broccoli", "carrot", "snow peas", "water chestnuts"],
    "Italian Caponata": ["eggplant", "celery", "olive", "capers"],
    "French Ratatouille": ["eggplant", "zucchini", "bell pepper", "tomato"],
    "Korean Kimchi": ["cabbage", "radish", "garlic", "ginger"],
    "Mexican Guacamole": ["avocado", "tomato", "onion", "lime"]    
}

def detect_items(image_path):
    results = model(image_path)
    detected_items = []
    for result in results:
        for class_id in result.boxes.cls:
            class_name = model.names[int(class_id)]
            detected_items.append(class_name)
    return list(set(detected_items))

def get_recipe_details(detected_items):
    """Generate structured recipe suggestions and formatted text based on detected items."""
    detected_items_lower = [item.lower() for item in detected_items]
    matching_recipes = []
    
    for recipe_name, ingredients in RECIPE_DATABASE.items():
        matched = [ing for ing in ingredients if ing.lower() in detected_items_lower]
        matching_count = len(matched)
        
        if matching_count > 0:
            match_percentage = round((matching_count / len(ingredients)) * 100)
            missing = [ing for ing in ingredients if ing.lower() not in detected_items_lower]
            matching_recipes.append({
                "name": recipe_name,
                "score": match_percentage,
                "matching_ingredients": matching_count,
                "total_ingredients": len(ingredients),
                "matched": matched,
                "missing": missing
            })
    
    # Sort by match score descending
    matching_recipes.sort(key=lambda x: (x["score"], x["matching_ingredients"]), reverse=True)
    
    tips = [
        "Tip: Try combining vegetables for a healthy and colorful stir-fry!",
        "Tip: Most vegetables can be roasted with olive oil, salt, and pepper for a simple side dish.",
        "Tip: Don't forget herbs and spices to enhance your dishes.",
        "Tip: Leftover vegetables are perfect for making soup or stock."
    ]
    tip = random.choice(tips)
    
    # Text fallback for backwards compatibility
    if not matching_recipes:
        response_text = "I couldn't find any recipes that use these ingredients. Try adding some common produce or staples."
    else:
        response_text = "Here are some recipe suggestions based on your ingredients:\n\n"
        for i, recipe in enumerate(matching_recipes[:5], 1):
            response_text += f"{i}. {recipe['name']} - Match: {recipe['score']}%\n"
            response_text += f"   • Using {recipe['matching_ingredients']} of {recipe['total_ingredients']} ingredients\n"
            if recipe["missing"]:
                response_text += f"   • Also needed: {', '.join(recipe['missing'])}\n"
            response_text += "\n"
        response_text += tip
        
    return matching_recipes[:6], response_text, tip

def get_recipe_suggestions(detected_items):
    _, text, _ = get_recipe_details(detected_items)
    return text

@app.route('/')
def index():
    return send_from_directory('', 'index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('', 'styles.css')

@app.route('/samples/<path:filename>')
def serve_sample(filename):
    return send_from_directory('samples', filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided in request'}), 400

    image = request.files['image']
    
    if image.filename == '':
        return jsonify({'error': 'No file selected for upload'}), 400
    
    # Create tmp directory if it doesn't exist
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    
    # Generate a safe temporary filename
    safe_filename = f"{int(datetime.now().timestamp())}_{image.filename}"
    image_path = os.path.join('tmp', safe_filename)
    image.save(image_path)
    
    try:
        # Detect items using the YOLO model
        detected_items = detect_items(image_path)
        
        # Get recipe suggestions and structured metadata
        recipes_list, recipe_text, tip = get_recipe_details(detected_items)
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "detectedItems": detected_items,
            "recipeSuggestions": recipe_text
        }

        try:
            with open("logs.json", "a") as f:
                f.write(json.dumps(log_data) + "\n")
        except Exception as log_err:
            print(f"Logging warning: {log_err}")

        return jsonify({
            'detectedItems': detected_items,
            'recipeSuggestions': recipe_text,
            'recipes': recipes_list,
            'tip': tip
        })
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            'error': f"Error processing image: {str(e)}"
        }), 500
    finally:
        # Clean up temporary uploaded file
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception:
                pass

if __name__ == '__main__':
    app.run(debug=True)