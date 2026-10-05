# SageMaker JumpStart inference guide: VoxCPM2, LocateAnything-3B, FLUX.2-small-decoder

A short, practical guide to invoking three Amazon SageMaker JumpStart community models. It
covers the request payload each model expects and the response shape each one returns, with
the example notebooks and a trimmed endpoint log from a real deployment.

These three models are not text generation models, so the generic "deploy text generation
model" sample code (which parses `response["generated_text"]`) does not work for them. The
payloads below are what each model actually expects.

## Models covered

| Model ID | Task | Input | Output |
|---|---|---|---|
| `huggingface-tts-openbmb-voxcpm2` | Text to speech | text | base64 WAV |
| `huggingface-od-nvidia-locateanything-3b` | Object detection / visual grounding | image + prompt | text with box coordinates |
| `huggingface-txt2img-black-forest-labs-flux-2-small-decoder` | VAE image decoder | image | base64 PNG |
| `huggingface-vlm-gemma-4-e2b-instruct` | Vision language model (chat) | text + image | text |

Note on the last one: its model ID contains `txt2img`, but the model is a distilled VAE
decoder. It takes an image and returns an image. It is not a text to image generator.

## Invoking an endpoint

Each model is deployed to a SageMaker real time endpoint and invoked with content type
`application/json`. The boilerplate is the same for all three:

```python
import json, boto3

runtime = boto3.client("runtime.sagemaker", region_name="your-region")

def query(endpoint_name, payload):
    resp = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    return json.loads(resp["Body"].read())
```

### VoxCPM2 (text to speech)

Request:

```json
{"inputs": "Your text here", "parameters": {"inference_timesteps": 10, "cfg_value": 2.0}}
```

Response keys: `audio_base64` (base64 WAV), `sample_rate` (48000), `format` ("wav"),
`num_samples`. Decode `audio_base64` to get the audio.

For a designed voice, put a description in parentheses at the start of the text, with the
content to speak immediately after the closing parenthesis (no space):

```json
{"inputs": "(A young woman, gentle and sweet voice)Hello, welcome to VoxCPM2!", "parameters": {"inference_timesteps": 10, "cfg_value": 2.0}}
```

### LocateAnything-3B (object detection and visual grounding)

Request:

```json
{"inputs": {"image": "<base64 PNG>", "prompt": "Locate all the instances that matches the following description: object."}, "parameters": {"generation_mode": "hybrid", "max_new_tokens": 256, "temperature": 0.0}}
```

Response keys: `text` (the grounding output, including box coordinates in the model's token
format) and `generation_mode`. `generation_mode` can be `hybrid` (default), `fast`, or
`slow`. Join multiple detection categories with `</c>` in the prompt.

### FLUX.2-small-decoder (VAE image decoder)

Request:

```json
{"image": "<base64 PNG>", "parameters": {"seed": 0}}
```

Response keys: `image_base64` (base64 PNG), `width`, `height`, `latent_shape`, `format`
("PNG"). You can also run a health check with a random latent and no image input:
`{"parameters": {"height": 256, "width": 256, "seed": 0}}`.

## Vision language models: finding the input contract

Some JumpStart community models are vision language models (VLMs) served through a
vLLM OpenAI compatible chat endpoint. For these, the example notebook can be very brief
and may not show how to pass an image or what fields are accepted. The reliable way to
know the contract is that these endpoints follow the OpenAI Chat Completions format.
vLLM documents the multimodal (image) format here:
https://docs.vllm.ai/en/latest/features/multimodal_inputs.html

Example model: `huggingface-vlm-gemma-4-e2b-instruct` (see the `vlm-gemma-4-e2b-instruct/`
folder for the notebook, payloads, response shapes, and a trimmed endpoint log).

Content type is `application/json`. A text only request:

```json
{"messages": [{"role": "user", "content": "What is deep learning? Answer in one sentence."}], "max_tokens": 128}
```

To pass an image, `content` becomes a list of parts. The `image_url` part must be an
object with a `url` field (not a plain string), and the `url` is a data URI with the
base64 encoded image:

```json
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_IMAGE>"}}]}], "max_tokens": 128}
```

The response is an OpenAI chat completion object. The generated text is at
`choices[0].message.content`, with a `usage` object for token counts. If you pass
`image_url` as a plain string instead of an object, the endpoint returns a 400 with a
validation message saying `image_url` must be a dictionary, which is a quick way to
check your payload shape.

## Repo layout

```
notebooks/   The model specific inference notebooks (one per model)
payloads/    The exact request bodies used to test each model
responses/   The response shape returned by each model (large base64 fields redacted to a length)
logs/        A trimmed endpoint log from a real deployment of each model
vlm-gemma-4-e2b-instruct/   VLM example: notebook, text and image payloads, response shapes, log
```

The response files show the keys and types only. Large base64 blobs are shown as
`<N chars>` so the files stay small and carry no binary payload.

## How these were verified

Each model was deployed to a real time endpoint on `ml.g6.2xlarge` and invoked with the
request in `payloads/`. The response shape in `responses/` is taken straight from the
endpoint response. The endpoint logs in `logs/` are trimmed from the container's own log
stream and show the `POST /invocations ... 200` line and the preprocess, predict, and
postprocess timings. Instance IDs and internal IP addresses have been redacted.

The request structures are the stable part to rely on. The exact set of fields returned in a
response can change if a model is updated to a newer version, so treat the response fields as
a guide to the shape rather than a fixed contract.

## Sources

The notebooks in `notebooks/` are the model specific inference notebooks published by Amazon
SageMaker JumpStart. They are available in the public JumpStart cache buckets, readable with
no credentials, at:

```
https://jumpstart-cache-prod-<region>.s3.<region>.amazonaws.com/huggingface-notebooks/<model-id>-inference-jl.ipynb
```

For example:

```
https://jumpstart-cache-prod-eu-south-2.s3.eu-south-2.amazonaws.com/huggingface-notebooks/huggingface-tts-openbmb-voxcpm2-inference-jl.ipynb
```

Each model also has a matching `-inference.ipynb` variant at the same prefix. The same keys
resolve from other regional buckets (for example `us-east-1`).

Model references:

- VoxCPM2: https://huggingface.co/openbmb/VoxCPM2
- LocateAnything-3B: https://huggingface.co/nvidia/LocateAnything-3B
- FLUX.2-small-decoder: https://huggingface.co/black-forest-labs/FLUX.2-small-decoder
- Amazon SageMaker JumpStart: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html

## Licence note

Each model carries its own licence (VoxCPM2 and FLUX.2-small-decoder are Apache-2.0;
LocateAnything-3B is under the NVIDIA licence for non-commercial use). Check the model card
before use. VoxCPM2's own terms prohibit using it for impersonation, fraud, or
disinformation, and ask that AI generated content be clearly labelled.
