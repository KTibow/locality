import {
  Qwen2ForCausalLM,
  PretrainedConfig,
  AutoTokenizer,
} from "@xenova/transformers";
import { InferenceSession, env } from "onnxruntime-web/webgpu";

let context;

self.onmessage = async (event) => {
  const { type } = event.data;

  if (type == "init") {
    const buffer = event.data.buffer;

    env.wasm.wasmPaths = "/wasm/";

    const session = await InferenceSession.create(buffer, {
      executionProviders: ["webnn"],
    });

    console.log("loading tokenizer...");
    const tokenizer = await AutoTokenizer.from_pretrained(
      "Qwen/Qwen2-0.5B-Instruct"
    );
    console.log("loading model...");
    const model = new Qwen2ForCausalLM(
      new PretrainedConfig({
        architectures: ["Qwen2ForCausalLM"],
        attention_dropout: 0.0,
        bos_token_id: 151643,
        eos_token_id: 151645,
        hidden_act: "silu",
        hidden_size: 896,
        initializer_range: 0.02,
        intermediate_size: 4864,
        max_position_embeddings: 32768,
        max_window_layers: 24,
        model_type: "qwen2",
        num_attention_heads: 14,
        num_hidden_layers: 24,
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
