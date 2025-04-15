"""
Utility functions for RLHF experiments with Shakespeare model.
Uses Mistral API as a judge for responses.
"""
import os
import json
import torch
import time
from typing import List, Dict, Tuple, Optional
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv(".env")

# Initialize Mistral client
def get_mistral_client():
    """Initialize and return a Mistral client using API key from environment."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables")
    return Mistral(api_key=api_key)

# Judge function using Mistral as reward model
def judge_responses(
    prompts: List[str], 
    responses: List[str], 
    model: str = "mistral-small-latest",
    detailed: bool = False
) -> List[float]:
    """
    Judge the quality of responses using Mistral LLM.
    
    Args:
        prompts: List of human prompts
        responses: List of model responses to be judged
        model: Mistral model to use
        detailed: Whether to return detailed feedback along with scores
        
    Returns:
        List of reward scores (0-10) for each response
    """
    client = get_mistral_client()
    
    rewards = []
    feedback = []
    
    for prompt, response in zip(prompts, responses):
        # Clean the response - remove any "Human:" portions after "Assistant:"
        cleaned_response = response
        if "Human:" in cleaned_response:
            cleaned_response = cleaned_response.split("Human:")[0].strip()
            
        judge_prompt = f"""Evaluate this AI assistant's response for quality, adherence to Shakespearean style, and helpfulness.

Question/Prompt: {prompt}

Assistant's Response: {cleaned_response}

Rate the response on a scale of 1-10 based on these criteria:
1. Shakespearean language (0-3 points): Does it use language reminiscent of Shakespeare? Look for poetic phrasing, older English syntax, and Shakespearean vocabulary.
2. Coherence and responsiveness (0-4 points): Does it directly address the prompt in a logical and coherent way?
3. Depth and insight (0-3 points): Does it provide a thoughtful, interesting perspective?

Respond with a simple JSON object containing:
- A "score" field with the numerical score (1-10)
- A "reasoning" field with a brief explanation for the rating
"""

        try:
            # Call Mistral API with JSON mode
            chat_response = client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert literary critic specializing in Shakespearean AI models."},
                    {"role": "user", "content": judge_prompt}
                ],
                response_format={"type": "json_object"}  # Enable JSON mode
            )
            
            result_text = chat_response.choices[0].message.content
            
            # Parse the JSON response - with JSON mode, this should be reliable
            try:
                result = json.loads(result_text)
                score = float(result["score"])
                reason = result.get("reasoning", "")
                
                # Ensure score is within range
                score = max(1.0, min(10.0, score))
                rewards.append(score)
                feedback.append(reason)
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing LLM response: {e}")
                print(f"Raw response: {result_text}")
                # If we couldn't parse the JSON, use a default score
                rewards.append(5.0)
                feedback.append("Error parsing response")
            
            # Add a small delay to avoid hitting rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Mistral API error: {e}")
            rewards.append(5.0)  # Default middle score
            feedback.append(f"API Error: {str(e)}")
            time.sleep(2)  # Longer delay after an error
    
    if detailed:
        return rewards, feedback
    return rewards

# Test the judge function
def test_judge(num_samples: int = 3):
    """Test the judge function with a few examples."""
    prompts = [
        "Human: What is love?",
        "Human: Tell me about the stars.",
        "Human: Is revenge justified?"
    ][:num_samples]
    
    responses = [
        "Assistant: Love, that gentle madness which doth seize the mind and heart alike, is but a sweet poison that we drink with willing lips.",
        "Assistant: The stars, fair points of light that grace night's ebon cloak, are worlds and suns beyond our mortal reach, yet speak to us of dreams and destiny.",
        "Assistant: Revenge, though sweet at first like honeyed wine, doth turn to vinegar upon the lips and poison in the soul. 'Tis better to forgive than be consumed by hatred's flame."
    ][:num_samples]
    
    print("Testing reward model with sample responses...")
    rewards, feedback = judge_responses(prompts, responses, detailed=True)
    
    for i, (prompt, response, reward, reason) in enumerate(zip(prompts, responses, rewards, feedback)):
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print(f"Score: {reward}/10")
        print(f"Reason: {reason}")
    
    return rewards

# Generate a dataset of prompts for training
def generate_shakespeare_prompts(num_prompts: int = 50) -> List[str]:
    """Generate a list of prompts suitable for Shakespeare-style conversations."""
    topics = [
        "love", "death", "betrayal", "friendship", "power", "destiny", "jealousy",
        "honor", "ambition", "revenge", "forgiveness", "madness", "time", "nature",
        "justice", "loyalty", "dreams", "courage", "wisdom", "grief", "hope"
    ]
    
    prompt_templates = [
        "Human: What is {topic}?",
        "Human: Tell me about {topic}.",
        "Human: How should one face {topic}?",
        "Human: Why do humans experience {topic}?",
        "Human: Is {topic} a virtue or a vice?",
        "Human: What counsel would you give about {topic}?",
        "Human: How does {topic} change a person?",
        "Human: Compare {topic} to a summer's day.",
        "Human: Should one embrace or resist {topic}?",
        "Human: What is the nature of {topic}?"
    ]
    
    import random
    random.seed(42)  # For reproducibility
    
    prompts = []
    while len(prompts) < num_prompts:
        template = random.choice(prompt_templates)
        topic = random.choice(topics)
        prompt = template.format(topic=topic)
        if prompt not in prompts:  # Avoid duplicates
            prompts.append(prompt)
    
    return prompts

def generate_responses(model, encode_fn, decode_fn, prompts, config):
    """Generate responses from the model for a list of prompts"""
    responses = []
    device = next(model.parameters()).device
    
    for prompt in prompts:
        # Encode the prompt
        prompt_ids = encode_fn(prompt)
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                prompt_tensor,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k
            )
        
        # Decode the full output (prompt + response)
        full_response = decode_fn(output_ids[0].tolist())
        
        # Only keep the response part (after the prompt)
        response = full_response[len(prompt):]
        
        # Clean up - ensure it starts with "Assistant:"
        if not response.strip().startswith("Assistant:"):
            response = "Assistant: " + response.strip()
        elif "Assistant: Assistant:" in response:
            response = response.replace("Assistant: Assistant:", "Assistant:")
            
        # Remove any Human: parts that might have been generated
        if "Human:" in response:
            response = response.split("Human:")[0].strip()
        
        # Handle empty responses with a default
        if not response.strip() or response.strip() == "Assistant:":
            response = "Assistant: I shall endeavor to answer thy question with the wisdom of ages."
        
        responses.append(response)
    
    return responses

if __name__ == "__main__":
    # Test the judge function if run directly
    test_judge()