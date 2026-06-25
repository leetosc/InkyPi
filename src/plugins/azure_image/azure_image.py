from plugins.base_plugin.base_plugin import BasePlugin
from openai import AzureOpenAI
from PIL import Image
from io import BytesIO
import base64
import logging

logger = logging.getLogger(__name__)

DEFAULT_API_VERSION = "2024-10-21"
DEFAULT_IMAGE_QUALITY = "medium"

class AzureImage(BasePlugin):
    def generate_settings_template(self):
        template_params = super().generate_settings_template()
        template_params['api_key'] = {
            "required": True,
            "service": "Azure",
            "expected_key": "AZURE_API_KEY"
        }
        return template_params

    def generate_image(self, settings, device_config):
        api_key = device_config.load_env_key("AZURE_API_KEY")
        if not api_key:
            raise RuntimeError("Azure API Key not configured.")

        endpoint = settings.get("endpoint", "").strip()
        if not endpoint:
            raise RuntimeError("Azure Endpoint not provided.")

        image_deployment = settings.get("imageDeployment", "").strip()
        if not image_deployment:
            raise RuntimeError("Image Deployment name not provided.")

        api_version = settings.get("apiVersion", "").strip() or DEFAULT_API_VERSION

        text_prompt = settings.get("textPrompt", "")
        image_quality = settings.get('quality', DEFAULT_IMAGE_QUALITY)
        randomize_prompt = settings.get('randomizePrompt') == 'true'

        chat_deployment = settings.get("chatDeployment", "").strip()
        if randomize_prompt and not chat_deployment:
            raise RuntimeError("Chat Deployment name is required when randomizing the prompt.")

        image = None
        try:
            ai_client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=api_key,
            )
            if randomize_prompt:
                text_prompt = AzureImage.fetch_image_prompt(ai_client, chat_deployment, text_prompt)

            image = AzureImage.fetch_image(
                ai_client,
                text_prompt,
                model=image_deployment,
                quality=image_quality,
                orientation=device_config.get_config("orientation")
            )
        except Exception as e:
            logger.error(f"Failed to make Azure request: {str(e)}")
            raise RuntimeError("Azure request failure, please check logs.")
        return image

    @staticmethod
    def fetch_image(ai_client, prompt, model, quality="medium", orientation="horizontal"):
        logger.info(f"Generating image for prompt: {prompt}, model: {model}, quality: {quality}")
        prompt += (
            ". The image should fully occupy the entire canvas without any frames, "
            "borders, or cropped areas. No blank spaces or artificial framing."
        )
        prompt += (
            "Focus on simplicity, bold shapes, and strong contrast to enhance clarity "
            "and visual appeal. Avoid excessive detail or complex gradients, ensuring "
            "the design works well with flat, vibrant colors."
        )
        size = "1536x1024" if orientation == "horizontal" else "1024x1536"

        response = ai_client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
        )
        image_base64 = response.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        img = Image.open(BytesIO(image_bytes))
        return img

    @staticmethod
    def fetch_image_prompt(ai_client, chat_deployment, from_prompt=None):
        logger.info(f"Getting random image prompt...")

        system_content = (
            "You are a creative assistant generating extremely random and unique image prompts. "
            "Avoid common themes. Focus on unexpected, unconventional, and bizarre combinations "
            "of art style, medium, subjects, time periods, and moods. No repetition. Prompts "
            "should be 20 words or less and specify random artist, movie, tv show or time period "
            "for the theme. Do not provide any headers or repeat the request, just provide the "
            "updated prompt in your response."
        )
        user_content = (
            "Give me a completely random image prompt, something unexpected and creative! "
            "Let's see what your AI mind can cook up!"
        )
        if from_prompt and from_prompt.strip():
            system_content = (
                "You are a creative assistant specializing in generating highly descriptive "
                "and unique prompts for creating images. When given a short or simple image "
                "description, your job is to rewrite it into a more detailed, imaginative, "
                "and descriptive version that captures the essence of the original while "
                "making it unique and vivid. Avoid adding irrelevant details but feel free to "
                "include creative and visual enhancements. Avoid common themes. Focus on "
                "unexpected, unconventional, and bizarre combinations of art style, medium, "
                "subjects, time periods, and moods. Do not provide any headers or repeat the "
                "request, just provide your updated prompt in the response. Prompts "
                "should be 20 words or less and specify random artist, movie, tv show or time "
                "period for the theme."
            )
            user_content = (
                f"Original prompt: \"{from_prompt}\"\n"
                "Rewrite it to make it more detailed, imaginative, and unique while staying "
                "true to the original idea. Include vivid imagery and descriptive details. "
                "Avoid changing the subject of the prompt."
            )

        # Make the API call
        response = ai_client.chat.completions.create(
            model=chat_deployment,
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            temperature=1
        )

        prompt = response.choices[0].message.content.strip()
        logger.info(f"Generated random image prompt: {prompt}")
        return prompt
