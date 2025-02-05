import os
import time
import torch
import uvicorn
import asyncio
import random
import numpy as np
from hashlib import sha256
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from src.flux.xflux_pipeline import XFluxPipeline
import boto3
import logging
from concurrent.futures import ThreadPoolExecutor

# AWS Configuration
BUCKET_NAME = ""
BUCKET_REGION = "us--1"
AWS_ACCESS_KEY = ""
AWS_SECRET_KEY = ""

MODEL_ID = "Pranav"

# App Configuration
MAX_CONCURRENT_JOBS = 2  # Adjust based on GPU memory


# FastAPI Application
app = FastAPI(title="AI Image Generation API", 
             description="High-performance image generation service powered by FLUX-1 model")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS S3 Client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=BUCKET_REGION
)

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)
semaphore = asyncio.Semaphore(5)

class GenerationRequest(BaseModel):
    prompt: str
    neg_prompt: str = ""
    num_steps: int = 25
    guidance: float = 1.0
    seed: int = 123456789
    width: int = 1024
    height: int = 1024
    num_images_per_prompt: int = 1

class GenerationResponse(BaseModel):
    status: str
    image_urls: list[str] | None = None
    generation_time: float | None = None
    prompt: str | None = None
    details: dict | None = None
    error: str | None = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def startup_event():
    """Load model on startup and move to GPU"""
    try:
        logger.info("🌀 Clearing CUDA cache...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        logger.info("🌀 Loading model...")
        start = time.time()

        app.state.model = XFluxPipeline("flux-dev", device, offload=False)

        logger.info(f"✅ Model loaded in {time.time()-start:.2f}s")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

MAX_SEED = np.iinfo(np.int32).max


def generate_images_sync(model, params):
    """Synchronous image generation function to run in thread"""
    print("start")
    images = []
    with torch.inference_mode():  # Optimizes and prevents unnecessary gradient calculations
        for i in range(params.get('num_images_per_prompt', 1)):  # Generate multiple images if required
            result: Image.Image = model(
                prompt=params['prompt'],
                neg_prompt=params.get('neg_prompt', ""),
                num_steps=params.get('num_steps', 25),
                guidance=params.get('guidance', 1.0),
                seed=params.get('seed', 123456789),
                width=params.get('width', 1024),
                height=params.get('height', 1024)
            )
            images.append(result)
            params['seed'] += 1  # Increment seed for different image each time
        print("done")
    return images  # Return list of images



async def upload_to_s3(image_path, s3_key):
    """Async S3 upload with retries"""
    for attempt in range(3):
        try:
            await asyncio.to_thread(
                s3_client.upload_file,
                image_path,
                BUCKET_NAME,
                s3_key
            )
            return True
        except Exception as e:
            logger.warning(f"Upload attempt {attempt+1} failed: {str(e)}")
            await asyncio.sleep(1)
    return False

@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    """Generate images from text prompt"""
    start_time = time.time()
    response = {"status": "processing", "details": {}}
    
    try:
        async with semaphore:
            # Prepare generation parameters
            params = request.model_dump()  # Fixed dict() issue
            params['seed'] = params['seed'] or int(time.time() % 100000)
            
            # Run model in thread
            loop = asyncio.get_event_loop()
            images = await loop.run_in_executor(
                executor,
                generate_images_sync,
                app.state.model,
                params
            )
            
            # Process and upload images
            upload_tasks = []
            image_urls = []
            temp_files = []
            
            for idx, image in enumerate(images):
                print("done 1")
                image_hash = sha256(image.tobytes()).hexdigest()
                image_path = f"/tmp/{image_hash}.png"
                
                # Save image to temp file
                await loop.run_in_executor(
                    executor,
                    image.save,
                    image_path
                )
                temp_files.append(image_path)
                
                # Prepare upload task
                s3_key = f"generations/{datetime.now().strftime('%Y-%m')}/{image_hash}.png"
                upload_tasks.append(
                    upload_to_s3(image_path, s3_key)
                )
                image_urls.append(
                    f"https://{BUCKET_NAME}.s3.{BUCKET_REGION}.amazonaws.com/{s3_key}"
                )
            
            # Wait for all uploads to complete
            upload_results = await asyncio.gather(*upload_tasks)
            
            # Cleanup temp files
            for path in temp_files:
                try:
                    os.remove(path)
                except:
                    pass
            
            # Check for failed uploads
            if not all(upload_results):
                raise RuntimeError("Some images failed to upload to S3")
            
            # Build response
            response.update({
                "status": "success",
                "image_urls": image_urls,
                "generation_time": time.time() - start_time,
                "details": {
                    "model": MODEL_ID,
                    "resolution": f"{request.height}x{request.width}",
                    "seed": params['seed']
                }
            })
            
            logger.info(f"Generated {len(images)} images in {response['generation_time']:.2f}s")
            
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        response.update({
            "status": "error",
            "error": str(e),
            "generation_time": time.time() - start_time
        })
    
    return response

@app.get("/status")
async def service_status():
    """Service health check"""
    return {
        "status": "ready",
        "model": MODEL_ID,
        "concurrency": MAX_CONCURRENT_JOBS,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.get("/")
async def root():
    return {
        "message": "AI Image Generation Service",
        "docs": "/docs",
        "status": "/status"
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        timeout_keep_alive=300
    )
