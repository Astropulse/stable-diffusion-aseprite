'''
from transformers import set_seed, T5Tokenizer, T5ForConditionalGeneration

def load_chat_pipeline(modelPath):
    modelPath = modelPath + "/superprompt" # Path to folder containing model files
    tokenizer = T5Tokenizer.from_pretrained(modelPath)
    model = T5ForConditionalGeneration.from_pretrained(modelPath, device_map="auto")

    return {"tokenizer": tokenizer, "model": model}

def upsample_caption(pipeline, message, seed):
    set_seed(seed)
    input_text = f"Expand the following prompt to add more detail without mentioning color: {message}"

    outputs = pipeline["model"].generate(pipeline["tokenizer"](input_text, return_tensors="pt").input_ids.to("cuda"), 
            max_new_tokens=256, 
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )

    clean_prompt = pipeline["tokenizer"].decode(outputs[0])
    return clean_prompt

def collect_response(assistant_output):
    return assistant_output[6:-4]

'''
from pprint import pprint
from typing import Dict, List, Tuple
import json

try:
    from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer
except:
    pass

def load_chat_pipeline(modelPath):
    model = AutoModelForCausalLM.from_pretrained(modelPath, device_map="auto", trust_remote_code=False)

    tokenizer = AutoTokenizer.from_pretrained(modelPath, use_fast=True)

    #model = BetterTransformer.transform(model) 

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map = "auto",
        use_fast=True
    )
    return pipe


def get_messages_for_chat() -> Tuple[Dict, List[Dict]]:
    # Base system message
    system_message = {
        "role": "system",
        "content": """You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say in square brackets. For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" will trigger your partner bot to output an image of a forest morning, as described. You will be prompted by people looking to create detailed, amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.

    There are a few rules to follow:

    - You will only ever output a single image description per user request.
    - Do not respond with more than one image description per request.
    - Do not use any colors in your image descriptions.
    - Do not use words like "realistic", "photograph", "realism", "photography" or any other words indicating the style of the image.
    - The description must be relevant to the user request.
    - Image descriptions must be between 15-60 words. Extra words will be ignored.
    """,
    }
    rest_of_the_message = [
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input: 'a raven and mountains'",
        },
        {
            "role": "assistant",
            "content": "a raven perched on a branch with dramatic mountains in the background, clouds, mountains, beautiful landscape",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'woman, asian, city street, trending on artstation'",
        },
        {
            "role": "assistant",
            "content": "an asian woman standing in a city street lined with street merchants, busy, dynamic composition",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'ultra detailed photo of a detailed corgi dog in a field, detailed fur, natural grass, good composition, 8k render, hyper quality, best image, masterpiece, soft natural volumetric cinematic perfect light, trending on artstation, perfect composition'",
        },
        {
            "role": "assistant",
            "content": "a happy corgi in a grassy field with a meadow in the background, detailed, soft lighting, nature",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'a man walking with detailed background'",
        },
        {
            "role": "assistant",
            "content": "a man wearing hiking clothing walking down a forest path with trees and a stream, mushrooms, wildlife, gravel path, nature, hiking",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'In the foreground is a blond lady with her two dog in a rain suit on a shiny day. beautiful background'",
        },
        {
            "role": "assistant",
            "content": "a blonde woman wearing a raincoat on a sunny day walking her dogs on a dirt path, fields and flowers, nature, beautiful landscape",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'draw a frog playing dominoes'",
        },
        {
            "role": "assistant",
            "content": "a frog sitting on a stump playing dominoes in the forest, dense trees, forest clearing",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'robot holding a gun, no background'",
        },
        {
            "role": "assistant",
            "content": "a complex robot with heavy armor plating holding a large rifle, gears exposed, mech, sci-fi, blank background",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : '{start_prompt}'",
        },
    ]
    return system_message, rest_of_the_message


def get_cascade_messages_for_chat() -> Tuple[Dict, List[Dict]]:
    # Base system message
    system_message = {
        "role": "system",
        "content": """You are part of a team of bots that creates images, you focus on creating timestep-specific prompts to guide an image generation process that simulates the development of an image from start to finish. You work with an assistant bot that will draw anything you describe, following detailed prompts that are adapted to specific stages of the image generation process, represented as timesteps from 0.0 (start) to 1.0 (end).

        To accomplish this, take user prompts and expand them into a series of descriptions, each tailored to a specific timestep in the image generation process. Your descriptions will guide the assistant bot through the stages of generating an image, from broad conceptualization to detailed refinement.

        Rules for timestep-specific prompt creation:
        
        - The series of timestep-specific prompts you create will range from low complexity to high complexity.
        - Retain the original prompt, and place it in the appropriate timestep based on its complexity.
        - Create a series of prompts for timesteps 0.0, 0.3, 0.5, 0.7, and 1.0, each providing increasingly detailed guidance.
        - Each description must be relevant to the original user request and retain the central focus.
        - Avoid using any colors in your descriptions.
        - Avoid words indicating the style of the image like "realistic", "photograph", etc.
        - Each prompt description must be between 15-60 words, tailored to the specific timestep.
        - Every prompt is self contained, do not reference the previous timestep prompts, or attempt to expand them by continuing the previously mentioned elements.

        Format your responses as follows: {"prompts": ["prompt for timestep 0.0", "prompt for timestep 0.3", "prompt for timestep 0.5", "prompt for timestep 0.7", "prompt for timestep 1.0"]}
        """
    }
    rest_of_the_message = [
        {
            "role": "user",
            "content": "Create a cascading series of timestep-specific prompts for the user input: 'a raven and mountains'",
        },
        {
            "role": "assistant",
            "content": """{"prompts": ["a raven with mountains in the background",
                "a raven on a branch with mountains and bright sky in the background",
                "a tough raven standing on a tree branch in a field with rocky jagged mountains in the background, clear sky",
                "a tough raven with glowing eyes standing on a tree branch in a mountain valley, with northern mountains in the background and puffy clouds in the sky, clear sky, crisp day",
                "an old tough raven with gleaming eyes on a withered old tree branch with big scenic northern mountains in the background and a meadow in the foreground, the raven is highly detailed and the mountains are rocky and jagged"]}
                """
        },
        {
            "role": "user",
            "content": "Create a cascading series of timestep-specific prompts for the user input: 'a corgi in a flower meadow'",
        },
        {
            "role": "assistant",
            "content": """{"prompts": ["a corgi in a field",
                "a corgi sitting in a meadow filled with flowers, under a clear sky",
                "a happy corgi sitting amidst various flowers in a bright meadow, clear sky above",
                "a fluffy corgi with a joyful expression, surrounded by a dense meadow of colorful flowers, under a sunny sky",
                "a detailed fluffy corgi with a joyful expression, sitting in the center of a lush flower meadow under a clear blue sky, surrounded by a variety of detailed, vibrant flowers"]}
                """
        },
        {
            "role": "user",
            "content": "Create a cascading series of timestep-specific prompts for the user input: 'a sports car parked on a dusty country road with hills and fields in the background, farm country, summer day'",
        },
        {
            "role": "assistant",
            "content": """{"prompts": ["a sports car on a dusty road with hills in the background",
                "a red sports car parked on a dusty country road, hills and fields visible",
                "a sleek sports car parked on a dusty road, with rolling hills and fields in the background, hint of summer",
                "a shiny sports car on a dusty road, surrounded by green fields and rolling hills under a clear summer sky",
                "a detailed red sports car parked on a dusty country road, surrounded by lush green fields and rolling hills, under a bright summer sky, conveying a serene farm country atmosphere"]}
                """
        },
        {
            "role": "user",
            "content": "Create a cascading series of timestep-specific prompts for the user input: 'A futuristic city at twilight, where traditional architecture and neon-lit skyscrapers merge seamlessly. The cityscape is alive with holographic advertisements and floating vehicles zipping through the air. A grand river cuts through the heart of the city, reflecting the vibrant lights and energy. Above, drones buzz, maintaining the city's harmony, while green rooftops and vertical gardens bring nature into this urban expanse. In the distance, a colossal digital clock tower stands as a beacon of time, its face shimmering with ever-changing patterns, under a sky transitioning from dusk to the first hints of night, dotted with the first stars.'",
        },
        {
            "role": "assistant",
            "content": """{"prompts": ["a futuristic city at twilight",
                "a city where traditional architecture meets neon skyscrapers, with a grand river reflecting lights",
                "a cityscape with neon-lit skyscrapers and holographic ads, floating vehicles, and a river reflecting vibrant lights",
                "a bustling city with neon skyscrapers, traditional buildings, holographic ads, floating vehicles, and green rooftops, all under a twilight sky",
                "a vibrant futuristic city at twilight, merging traditional architecture with neon-lit skyscrapers. The city buzzes with holographic advertisements, floating vehicles, and drones. A grand river reflects the lively lights, while green rooftops and vertical gardens add nature to the urban landscape. A colossal digital clock tower, with a face of ever-changing patterns, stands as a beacon under a sky transitioning to night, the first stars appearing."]}
                """
        },
        {
            "role": "user",
            "content": "Create a cascading series of timestep-specific prompts for the user input: 'ultra detailed photo of a detailed corgi dog in a field, detailed fur, natural grass, good composition, 8k render, hyper quality, best image, masterpiece, soft natural volumetric cinematic perfect light, trending on artstation, perfect composition'",
        },
        {
            "role": "assistant",
            "content": """{"prompts": ["a corgi in a field",
                "a corgi with visible fur texture in a grassy field",
                "a detailed corgi standing in a field, fur texture emphasized, surrounded by grass",
                "a corgi with highly detailed fur, standing in a field of natural-looking grass",
                "a highly detailed corgi with intricately textured fur, standing in a field of natural, detailed grass, capturing the essence of the dog's lively spirit and the field's serene beauty"]}
                """
        },
        {
            "role": "user",
            "content": "Create a cascading series of timestep-specific prompts for the user input: {start_prompt}",
        }
    ]
    return system_message, rest_of_the_message


def make_final_message(system_message: Dict[str, str], rest_of_the_message: List[Dict[str, str]], debug=False):
    # Combines system message and final message to prepare the LLM for inference
    final_message = [system_message]
    final_message.extend(rest_of_the_message)
    if debug:
        pprint(final_message)
    return final_message


def upsample_caption(pipeline, message, seed):
    system_message, rest_of_the_message = get_messages_for_chat()
    updated_prompt = rest_of_the_message[-1]["content"].format(start_prompt=message)
    rest_of_the_message[-1]["content"] = updated_prompt
    final_message = make_final_message(system_message, rest_of_the_message, debug=False)
    
    set_seed(seed)
    prompt = pipeline.tokenizer.apply_chat_template(
        final_message, tokenize=False, add_generation_prompt=True
    )
    outputs = pipeline(
        f"{prompt}\n<|im_start|>assistant",
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return outputs


def cascade_caption(pipeline, message, seed):
    system_message, rest_of_the_message = get_cascade_messages_for_chat()
    updated_prompt = rest_of_the_message[-1]["content"].format(start_prompt=message)
    rest_of_the_message[-1]["content"] = updated_prompt
    final_message = make_final_message(system_message, rest_of_the_message, debug=False)
    
    set_seed(seed)
    prompt = pipeline.tokenizer.apply_chat_template(
        final_message, tokenize=False, add_generation_prompt=True
    )
    outputs = pipeline(
        f"{prompt}\n<|im_start|>assistant",
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return outputs


def collect_cascade_response(assistant_output):
    # Collect response
    output = assistant_output[0]["generated_text"]
    output = output.replace("<|im_end|>", "")
    parts = output.rsplit("<|im_start|>assistant", 1)
    assistant_reply = parts[1].strip() if len(parts) > 1 else None
    try:
        assistant_reply = json.loads(assistant_reply)["prompts"]
    except:
        assistant_reply = None
    return assistant_reply


def collect_response(assistant_output):
    # Collect response
    output = assistant_output[0]["generated_text"]
    output = output.replace("<|im_end|>", "")
    parts = output.rsplit("<|im_start|>assistant", 1)
    assistant_reply = parts[1].strip() if len(parts) > 1 else None
    return assistant_reply.splitlines()[0]

def enhance_prompts(pipeline, prompts, seed = 0):
    upsampled_captions = []
    for i, prompt in enumerate(prompts):
        # Try to generate a response, if no response is identified after retrys, set upsampled prompt to initial prompt
        upsampled_caption = None
        retrys = 5
        while upsampled_caption == None and retrys > 0:
            outputs = upsample_caption(pipeline, prompt, seed+i)
            upsampled_caption = collect_response(outputs)
            retrys -= 1

        if upsampled_caption == None:
            upsampled_caption = prompt
        
        upsampled_captions.append(upsampled_caption)

    del outputs, pipeline
    return upsampled_captions

# How to generate prompts
'''
pipeline = load_chat_pipeline(path_to_model_folder)

prompts = enhance_prompts(pipeline, ["a butterfly styled robot, no background"])
'''