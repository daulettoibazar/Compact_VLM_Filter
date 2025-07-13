import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import io
from qwen_vl_utils import process_vision_info
import re
import os
import json
import argparse

def load_multimodal_datasets(dataset_path):
    processed_data = []    

    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        for item in tqdm(data, desc="Loading dataset"):      
            processed_data.append({
                'image': item["image_path"],
                'text': item["caption"],
            })
    except Exception as e:
        print(f"Error loading your dataset: {e}")

    
    print(f"Total processed examples: {len(processed_data)}")
    return processed_data


class Qwen2VLFilter:
    """Class to filter image-text pairs using Compact VLM filtration model"""
    
    def __init__(self, model_path, batch_size=64):

        self.batch_size = batch_size
        
            
        print(f"Loading Compact VLM filtration model from {model_path}...")
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True, min_pixels=min_pixels, max_pixels=max_pixels)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto"
        )
        
        print(f"Model loaded successfully!")
    
    def preprocess_batch(self, batch_data):
        """
        Preprocess a batch of image-text pairs
        
        Args:
            batch_data: List of dictionaries containing 'image' and 'text'
            
        Returns:
            List of processed inputs ready for model
        """
        processed_inputs = []

        for item in batch_data:
            if isinstance(item['image'], str) and os.path.exists(item['image']):
                image = Image.open(item['image'])
            else:
                image = item['image']

            messages = [
            {
                "role": "system",
                "content": "You are helpful AI assistant"
            },
            {   
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "Analyze and assess the quality of following image-caption pair:\n" + item["text"]},
            ],}]

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            processed_inputs.append(inputs)

        return processed_inputs
    
    def process_batch_outputs(self, outputs_batch):
        """
        Process model outputs to determine which pairs to keep. Currently checks for a score >= 9. Please adjust as needed.
        
        Args:
            outputs_batch: List of model outputs
            
        Returns:
            List of booleans indicating which pairs to keep
        """
        keep_flags = []
        
        for output in outputs_batch:
            try:
                match = re.search(r'score:\s*\{(\d+)\}', output)
                if match:
                    score = int(match.group(1))
                    #TODO: Adjust the threshold as needed
                    if score>=9:
                        keep_flags.append(True)
                    else:
                        keep_flags.append(False)
                else:
                    print(f"Encountered unexpected outputs: {output}")
                    keep_flags.append(False)
                
            except Exception as e:
                print(f"Error extracting score from model output: {e}")
                print(f"Generated text: {output}")
                keep_flags.append(False)  # Default to not keeping if error
        
        return keep_flags
    
    def filter_dataset(self, dataset):
        """
        Filter dataset based on Qwen2-VL outputs
        
        Args:
            dataset: List of dictionaries with 'image', 'text', and 'source' keys
            progress_save_path: File path to save progress during filtering
            save_interval: How often to save progress (number of batches)
            
        Returns:
            Filtered dataset and scores
        """
        filtered_dataset = []
        
        # Process in batches
        num_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Filtering with Qwen2-VL"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(dataset))
            batch_data = dataset[start_idx:end_idx]
            
            try:

                processed_inputs = self.preprocess_batch(batch_data)
                
                batch_outputs = []

                with torch.inference_mode():
                    for inputs in processed_inputs:
                        inputs = inputs.to("cuda")

                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                        )

                        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        batch_outputs.append(output_text[0])
                
                keep_flags = self.process_batch_outputs(batch_outputs)
                

                for i, keep in enumerate(keep_flags):
                    if keep:
                        filtered_dataset.append(batch_data[i])
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
        
        print(f"Filtering complete! Kept {len(filtered_dataset)} out of {len(dataset)} examples ({len(filtered_dataset)/len(dataset)*100:.2f}%)")
        return filtered_dataset
    


def save_image(image, save_dir, image_name):
    """Save PIL image to a directory and return the file path"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_path = os.path.join(save_dir, image_name)
    image.save(image_path, format='PNG')
    return image_path

def save_filtered_dataset(filtered_data, output_image_folder, output_path):
    """
    Save filtered dataset to disk
    
    Args:
        filtered_data: List of filtered data items
        output_path: Path to save the dataset
    """

    final_data = []
    for i, item in tqdm(enumerate(filtered_data),desc="Getting saved training data for LLava model"):
        image = item['image']
        text = item['text']
        
        if isinstance(image, str):
            try:
                image = Image.open(image)
            except Exception as e:
                print(f"Error opening image: {e}")
                continue
        
        
        image_name = f"filtered_image_{i}.png"
        image_path = save_image(image, output_image_folder, image_name)
        
        res = {}
        res["image_path"] = image_path
        res["text"] = text

        final_data.append(res)

    with open(output_path, "w") as outfile:
        json.dump(final_data, outfile, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter image/caption dataset using Qwen2-VL model")
    parser.add_argument("--model_path", type=str, default="Dauka-transformers/Compact_VLM_filter", help="Path to the Qwen2-VL model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for filtering")
    parser.add_argument("--output_image_folder", type=str, default="./filtered_multimodal_dataset", help="Path to save the filtered images")
    parser.add_argument("--output_path", type=str, default="./filtered_dataset.json", help="Path to save the filtered dataset")
    parser.add_argument("--dataset_path", type=str, default="./data_to_filter.json", help="Path to the initial image/text dataset")

    args = parser.parse_args()
    
    dataset = load_multimodal_datasets(args.dataset_path)
    
    filter_model = Qwen2VLFilter(
        model_path=args.model_path,
        batch_size=args.batch_size
    )
    

    filtered_data = filter_model.filter_dataset(
        dataset=dataset
    )
    
    save_filtered_dataset(filtered_data, args.output_image_folder, args.output_path)
    print(f"Filtered dataset saved to {args.output_image_folder} and {args.output_path}")