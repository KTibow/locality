import {
  Qwen2Tokenizer,
  Qwen2ForCausalLM,
  PretrainedConfig,
} from "@huggingface/transformers";

const loadONNX = async () => {
  return await import("./shaken-onnx.js");
};

let context;

self.onmessage = async (event) => {
  const { env, InferenceSession } = await loadONNX();

  const { type } = event.data;
  console.log(event, event.data);

  if (type == "init") {
    const buffer = event.data.buffer;

    if (import.meta.env.DEV) {
      env.wasm.wasmPaths = "./wasm/";
    }

    const session = await InferenceSession.create(buffer, {
      executionProviders: ["webnn"],
    });

    console.log("loading tokenizer...");
    const { default: tokenizerJSON } = await import("./tokenizer.json");
    const tokenizer = new Qwen2Tokenizer(tokenizerJSON, {
      add_prefix_space: false,
      added_tokens_decoder: {
        151643: {
          content: "<|endoftext|>",
          lstrip: false,
          normalized: false,
          rstrip: false,
          single_word: false,
          special: true,
        },
        151644: {
          content: "<|im_start|>",
          lstrip: false,
          normalized: false,
          rstrip: false,
          single_word: false,
          special: true,
        },
        151645: {
          content: "<|im_end|>",
          lstrip: false,
          normalized: false,
          rstrip: false,
          single_word: false,
          special: true,
        },
      },
      additional_special_tokens: ["<|im_start|>", "<|im_end|>"],
      bos_token: null,
      chat_template:
        "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
      clean_up_tokenization_spaces: false,
      eos_token: "<|im_end|>",
      errors: "replace",
      model_max_length: 32768,
      pad_token: "<|endoftext|>",
      split_special_tokens: false,
      tokenizer_class: "Qwen2Tokenizer",
      unk_token: null,
    });
    console.log("loading model...");
    const model = new Qwen2ForCausalLM(
      new PretrainedConfig({
        architectures: ["Qwen2ForCausalLM"],
        attention_dropout: 0.0,
        bos_token_id: 151643,
        eos_token_id: 151645,
        hidden_act: "silu",
        hidden_size: 1536,
        initializer_range: 0.02,
        intermediate_size: 8960,
        max_position_embeddings: 32768,
        max_window_layers: 28,
        model_type: "qwen2",
        num_attention_heads: 12,
        num_hidden_layers: 28,
        num_key_value_heads: 2,
        rms_norm_eps: 1e-6,
        rope_theta: 1000000.0,
        sliding_window: 32768,
        tie_word_embeddings: true,
        torch_dtype: "bfloat16",
        transformers_version: "4.40.1",
        use_cache: true,
        use_sliding_window: false,
        vocab_size: 151936,
      }),
      { model: session },
      {
        bos_token_id: 151643,
        pad_token_id: 151643,
        eos_token_id: [151645, 151643],
        repetition_penalty: 1.1,
        temperature: 0,
        transformers_version: "4.37.0",
      }
    );

    context = { tokenizer, model };
    postMessage({ type: "ready" });
  } else if (type == "generate") {
    console.log("generating...");
    const { id, input } = event.data;
    const { tokenizer, model } = context;

    const encoded = tokenizer(input);
    console.log(encoded);

    let isPrompt = true;
    let output = [];
    const streamer = {
      put([ids]) {
        if (isPrompt) {
          isPrompt = false;
          return;
        }
        output.push(...ids);
        postMessage({
          type: "output",
          id,
          output: tokenizer.batch_decode([output])[0],
        });
      },
      end() {
        postMessage({ type: "end", id });
      },
    };
    model.generate({
      ...encoded,
      streamer,
      max_new_tokens: 500,
    });
  } else {
    console.warn(`unknown type: ${type}`);
  }
};
