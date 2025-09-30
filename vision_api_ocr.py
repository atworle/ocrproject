#!/usr/bin/env python3
"""
OpenAI Vision API OCR Demo
==========================

This script demonstrates how to use OpenAI's Vision API to directly transcribe
text from images, providing an alternative to traditional OCR + AI correction.

Usage:
    python vision_api_ocr.py <image_path> [--api-key <key>]

Requirements:
    - OpenAI API key (set as environment variable OPENAI_API_KEY or pass via --api-key)
    - Image file (JPG, PNG, etc.)
"""

import argparse
import os
import sys
from pathlib import Path
import base64
from openai import OpenAI


def encode_image(image_path):
    """
    Encode image to base64 for OpenAI Vision API.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def transcribe_with_vision_api(image_path, api_key):
    """
    Send image to OpenAI Vision API for text transcription.
    
    Args:
        image_path (str): Path to the image file
        api_key (str): OpenAI API key
        
    Returns:
        tuple: (transcribed_text, usage_info)
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Encode the image
        base64_image = encode_image(image_path)
        if not base64_image:
            return None, None
        
        # Determine image format
        image_format = Path(image_path).suffix.lower()
        if image_format not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            print(f"Warning: {image_format} may not be supported by Vision API")
        
        # Send request to OpenAI Vision API
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o for vision capabilities
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Please transcribe all the text visible in this image. The text may be in any language including English, Spanish, French, German, Chinese, Japanese, Korean, Arabic, Hindi, Tamil, or other languages.
                            
                            Instructions:
                            1. Extract ALL text from the image, preserving the original layout and structure
                            2. Maintain line breaks and paragraph structure
                            3. Do not add any commentary or interpretation
                            4. If text is unclear or partially obscured, transcribe what you can see
                            5. Preserve original spelling and formatting in the original language
                            6. Include headers, titles, dates, and all visible text elements
                            7. If the text is in a non-Latin script (like Arabic, Chinese, Tamil, etc.), transcribe it exactly as written
                            8. Do not translate the text - only transcribe it
                            
                            Return only the transcribed text in its original language."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format[1:]};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4000,
            temperature=0.1  # Low temperature for consistent transcription
        )
        
        # Extract usage information
        usage = response.usage
        usage_info = {
            'prompt_tokens': usage.prompt_tokens,
            'completion_tokens': usage.completion_tokens,
            'total_tokens': usage.total_tokens
        }
        
        transcribed_text = response.choices[0].message.content.strip()
        return transcribed_text, usage_info
        
    except Exception as e:
        print(f"Error calling OpenAI Vision API: {e}")
        return None, None


def calculate_cost(usage_info, model="gpt-4o"):
    """
    Calculate estimated cost based on token usage.
    
    Args:
        usage_info (dict): Token usage information
        model (str): Model used for the API call
        
    Returns:
        dict: Cost breakdown
    """
    # Pricing per 1K tokens (as of 2024)
    pricing = {
        "gpt-3.5-turbo": {
            "input": 0.0005,   # $0.50 per 1M tokens
            "output": 0.0015   # $1.50 per 1M tokens
        },
        "gpt-4o": {
            "input": 0.005,    # $5.00 per 1M tokens
            "output": 0.015    # $15.00 per 1M tokens
        }
    }
    
    if model not in pricing:
        model = "gpt-4o"  # Default fallback
    
    input_cost = (usage_info['prompt_tokens'] / 1000) * pricing[model]["input"]
    output_cost = (usage_info['completion_tokens'] / 1000) * pricing[model]["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "model": model
    }


def print_usage_summary(usage_info, cost_info):
    """
    Print token usage and cost summary to terminal.
    
    Args:
        usage_info (dict): Token usage information
        cost_info (dict): Cost breakdown
    """
    print("\n" + "=" * 50)
    print("OPENAI VISION API USAGE SUMMARY")
    print("=" * 50)
    print(f"Model: {cost_info['model']}")
    print(f"Prompt Tokens:  {usage_info['prompt_tokens']:,}")
    print(f"Output Tokens:  {usage_info['completion_tokens']:,}")
    print(f"Total Tokens:   {usage_info['total_tokens']:,}")
    print("-" * 50)
    print(f"Input Cost:     ${cost_info['input_cost']:.4f}")
    print(f"Output Cost:    ${cost_info['output_cost']:.4f}")
    print(f"Total Cost:     ${cost_info['total_cost']:.4f}")
    print("=" * 50)


def save_transcription(text, image_path, output_dir="vision_results"):
    """
    Save transcribed text to a file.
    
    Args:
        text (str): Transcribed text to save
        image_path (str): Path to original image
        output_dir (str): Directory to save results
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get base filename without extension
    base_name = Path(image_path).stem
    
    # Save transcribed text
    output_file = Path(output_dir) / f"{base_name}_vision_transcription.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Vision API transcription saved to: {output_file}")


def main():
    """Main function to run the Vision API OCR workflow."""
    parser = argparse.ArgumentParser(description="OCR using OpenAI Vision API")
    parser.add_argument("image_path", help="Path to the image file to process")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("--output-dir", default="vision_results", help="Directory to save results (default: vision_results)")
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.")
        sys.exit(1)
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key required.")
        print("Set OPENAI_API_KEY environment variable or use --api-key option.")
        sys.exit(1)
    
    print(f"Processing image: {args.image_path}")
    print("=" * 50)
    
    # Transcribe with Vision API
    print("Step 1: Sending image to OpenAI Vision API...")
    transcribed_text, usage_info = transcribe_with_vision_api(args.image_path, api_key)
    
    if not transcribed_text:
        print("Error: Vision API transcription failed.")
        sys.exit(1)
    
    print("Vision API transcription completed.")
    print(f"Transcribed {len(transcribed_text)} characters.")
    print("\nTranscribed text:")
    print("-" * 30)
    print(transcribed_text)
    print("-" * 30)
    
    # Calculate and display usage/cost
    if usage_info:
        cost_info = calculate_cost(usage_info, "gpt-4o")
        print_usage_summary(usage_info, cost_info)
    
    # Save results
    print("\nStep 2: Saving results...")
    save_transcription(transcribed_text, args.image_path, args.output_dir)
    
    print("\nProcess completed successfully!")


if __name__ == "__main__":
    main()
