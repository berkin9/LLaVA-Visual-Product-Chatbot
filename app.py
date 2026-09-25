import spaces
import torch
import gradio as gr

from transformers import (
    AutoConfig,
    AutoProcessor,
    LlavaForConditionalGeneration,
)

MODEL_ID = "berkin9/llava-visual-product-chatbot-merged"
COMPUTE_DTYPE = torch.float16


# ---------------------------------------------------------
# Processor
# ---------------------------------------------------------

config = AutoConfig.from_pretrained(MODEL_ID)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    use_fast=False,
)

processor.patch_size = config.vision_config.patch_size
processor.vision_feature_select_strategy = (
    config.vision_feature_select_strategy
)
processor.num_additional_image_tokens = 1
processor.tokenizer.padding_side = "left"

if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

print("Loading merged model...")

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=COMPUTE_DTYPE,
    low_cpu_mem_usage=True,
)

model = model.to("cuda")

model.eval()
model.config.use_cache = True

print("Merged model loaded successfully.")


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

@spaces.GPU(duration=30)
def generate_product_answer(image, question):

    if image is None:
        return "Please upload a product image first."

    if question is None or not question.strip():
        return "Please enter a question."

    image = image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question.strip()},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to("cuda")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_length:]

    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    return answer.split("\n")[0].strip()


# ---------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------

def respond_to_product(image, message, history):

    if history is None:
        history = []

    if image is None:
        return history, "", "⚠️ Please upload a product image first."

    if message is None or not message.strip():
        return history, "", "⚠️ Please enter a question."

    try:
        question = message.strip()

        answer = generate_product_answer(
            image,
            question,
        )

        history = history + [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

        return history, "", "✅ Ready"

    except Exception as e:
        return history, message, f"❌ Error: {str(e)}"


def clear_chat():
    return [], "", None, "✅ Ready"


# ---------------------------------------------------------
# Interface
# ---------------------------------------------------------

with gr.Blocks(title="LLaVA Visual Product Chatbot") as demo:

    gr.Markdown(
        """
        # 🛍️ LLaVA Visual Product Chatbot

        Upload a product image and ask questions about it.

        **Example questions**
        - What type of product is this?
        - What color is it?
        - Describe this product briefly.
        - How much does this product cost?
        - What material is this product made from?
        """
    )

    with gr.Row():

        with gr.Column(scale=1):

            image_input = gr.Image(
                type="pil",
                label="Product Image",
            )

            status = gr.Markdown("✅ Ready")

        with gr.Column(scale=2):

            chatbot = gr.Chatbot(
                label="Product Assistant",
                height=420,
            )

            message_input = gr.Textbox(
                label="Question",
                placeholder="Ask a question about the product...",
            )

            with gr.Row():
                send_button = gr.Button(
                    "Send",
                    variant="primary",
                )

                clear_button = gr.Button("Clear")

    send_button.click(
        fn=respond_to_product,
        inputs=[
            image_input,
            message_input,
            chatbot,
        ],
        outputs=[
            chatbot,
            message_input,
            status,
        ],
    )

    message_input.submit(
        fn=respond_to_product,
        inputs=[
            image_input,
            message_input,
            chatbot,
        ],
        outputs=[
            chatbot,
            message_input,
            status,
        ],
    )

    clear_button.click(
        fn=clear_chat,
        outputs=[
            chatbot,
            message_input,
            image_input,
            status,
        ],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch()
