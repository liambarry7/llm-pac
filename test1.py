# pip install -U "huggingface_hub"

# hf auth login
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto") # pip install accelerate

activities = ["sitting", "standing", "mixed-activity", "walking", "manual-work", "sports", "bicycling", "sleep"]
# input_text = "Write me a poem about Machine Learning."
# input_text = (f"You are classifying different physical activities. These activities are {activities}. "
#               "Given data from an accelerometer, what class of physical activity are you doing:"
#               "x=0.07865859, y=-1.0151173, z=-0.056527406.")
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

input_text = (
    f"Options: {' '.join(activities)}\n"
    f"Data: x={0.07865859}, y={-1.0151173}, z={-0.056527406}\n"
    "Question: Which activity from the options above best matches this raw accelerometer data?\n"
    "Answer: "
)
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=50, eos_token_id=tokenizer.encode("\n", add_special_tokens=False)[-1])
print(tokenizer.decode(outputs[0]))

# https://huggingface.co/google/gemma-2b

x,y,z = 0
x1,x2,x3,y1,y2,y3,z1,z2,z3,class1,class2,class3 = 0

# zero_shot_template = (
#     "Context: You are an expert in Physical Activity Classification. "
#     "The data was recorded from Axivity AX3 wrist-worn tri-axial accelerometer on their dominant hand. "
#     "The accelerometer was set to capture tri-axial acceleration data at 100 Hz with a dynamic range of  ±8g."
#     "Wearable cameras were used to collect ground truths of the participants’ activities while wearing the accelerometers. "
#     "Participants were given an OMG Life Autographer, a wearable camera worn around the neck which automatically takes photographs every 20 - 40 seconds\n", # string of activity classes
#     f"Labels: {', '.join(activities)}\n", # string of activity classes
#     f"Data: x = {x}, y = {y}, z = {z}\n",
#     "Question: Which activity from the options above best matches the data?",
#     "Answer: "
# )

zero_shot_template = f"""
    Context: 
    You are an expert in Physical Activity Classification.
    The data was recorded from Axivity AX3 wrist-worn tri-axial accelerometer on their dominant hand.
    The accelerometer was set to capture tri-axial acceleration data at 100 Hz with a dynamic range of ±8g.
    Wearable cameras were used to collect ground truths of the participants’ activities while wearing the accelerometers.
    Participants were given an OMG Life Autographer, a wearable camera worn around the neck which automatically takes photographs every 20 - 40 seconds
    
    Labels: {', '.join(activities)}
    Data: x = {x}, y = {y}, z = {z}
    Question: Which activity from the options above best matches the data?
    Answer:
"""

# few_shot_template = (
#     "Context: You are an expert in Physical Activity Classification. "
#     "The data was recorded from Axivity AX3 wrist-worn tri-axial accelerometer on their dominant hand. "
#     "The accelerometer was set to capture tri-axial acceleration data at 100 Hz with a dynamic range of  ±8g."
#     "Wearable cameras were used to collect ground truths of the participants’ activities while wearing the accelerometers. "
#     "Participants were given an OMG Life Autographer, a wearable camera worn around the neck which automatically takes photographs every 20 - 40 seconds\n", # string of activity classes
#     f"Examples of correctly labelled data: \n"
#     f"x = {x1}, y = {y1}, z = {z1}, Label = {class1}\n",
#     f"x = {x2}, y = {y2}, z = {z2}, Label = {class2}\n",
#     f"x = {x3}, y = {y3}, z = {z3}, Label = {class3}\n",
#     f"Labels: {', '.join(activities)}\n",  # string of activity classes
#     f"Data: x = {x}, y = {y}, z = {z}\n",
#     "Question: Which activity from the options above best matches the data?",
#     "Answer: "
# )

few_shot_template = f"""
    Context: 
    You are an expert in Physical Activity Classification.
    The data was recorded from Axivity AX3 wrist-worn tri-axial accelerometer on their dominant hand.
    The accelerometer was set to capture tri-axial acceleration data at 100 Hz with a dynamic range of ±8g.
    Wearable cameras were used to collect ground truths of the participants’ activities while wearing the accelerometers.
    Participants were given an OMG Life Autographer, a wearable camera worn around the neck which automatically takes photographs every 20 - 40 seconds
    
    Examples of correctly labelled data:
    x = {x1}, y = {y1}, z = {z1}, Label = {class1}
    x = {x1}, y = {y1}, z = {z1}, Label = {class1}
    x = {x1}, y = {y1}, z = {z1}, Label = {class1}
    x = {x1}, y = {y1}, z = {z1}, Label = {class1}
    x = {x1}, y = {y1}, z = {z1}, Label = {class1}
    
    Labels: {', '.join(activities)}
    Data: x = {x}, y = {y}, z = {z}
    Question: Which activity from the options above best matches the data?
    Answer:
"""

cot_template = f"""
    Context: 
        You are an expert in Physical Activity Classification.
        The data was recorded from Axivity AX3 wrist-worn tri-axial accelerometer on their dominant hand.
        The accelerometer was set to capture tri-axial acceleration data at 100 Hz with a dynamic range of ±8g.
        Wearable cameras were used to collect ground truths of the participants’ activities while wearing the accelerometers.
        Participants were given an OMG Life Autographer, a wearable camera worn around the neck which automatically takes photographs every 20 - 40 seconds
        
"""