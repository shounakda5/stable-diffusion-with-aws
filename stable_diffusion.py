import boto3
import json
import base64
import os

prompt_template = [{"text": "generate a 4k Ultra HD image depicting the lure of Vienna as in Billy Joel's famous song", "weight": 1}]
bedrock = boto3.client(service_name="bedrock-runtime")

payload = {
    "text_prompts": prompt_template,
    "cfg_scale": 10,
    "seed": 0,
    "steps": 50,
    "width": 1024,
    "height": 1024
}

model_id = "stability.stable-diffusion-xl-v1"
response_data = bedrock.invoke_model(modelId=model_id,
                                     contentType="application/json",
                                     accept="application/json",
                                     body=json.dumps(payload))

response_body = json.loads(response_data.get("body").read())
artifact = response_body.get("artifacts")[0]
image_encoded = artifact.get("base64").encode("utf-8")
image_bytes = base64.b64decode(image_encoded)

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
file_name = f"{output_dir}/generated-img.png"

with open(file_name, "wb") as f:
    f.write(image_bytes)