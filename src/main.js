import {
  AutoTokenizer,
  LlamaForCausalLM,
  PretrainedConfig,
} from "@xenova/transformers";
import { InferenceSession, env } from "onnxruntime-web/webgpu";
import { createGeneratorPipe } from "./_queue";

/** @type {{ tokenizer: import("@xenova/transformers").PreTrainedTokenizer, model: LlamaForCausalLM } | undefined} */
let context;

const getBuffer = async () => {
  const [fileH] = await window.showOpenFilePicker();
  const fileO = await fileH.getFile();
  return await fileO.arrayBuffer();
};
const button1 = document.querySelector("#button1");
button1.addEventListener("click", async () => {
  load();
});

const load = async () => {
  env.wasm.wasmPaths = "/wasm/";

  button1.remove();

  const statusEl = document.createElement("div");
  document.body.appendChild(statusEl);

  statusEl.innerText = "creating session...";
  const session = await InferenceSession.create(await getBuffer(), {
    executionProviders: ["webgpu", "wasm"],
  });

  statusEl.innerText = "loading model...";
  const tokenizer = await AutoTokenizer.from_pretrained(
    "Felladrin/onnx-Llama-160M-Chat-v1"
  );
  const model = new LlamaForCausalLM(
    new PretrainedConfig({
      _name_or_path: "Felladrin/Llama-160M-Chat-v1",
      architectures: ["LlamaForCausalLM"],
      attention_bias: false,
      bos_token_id: 1,
      eos_token_id: 2,
      hidden_act: "silu",
      hidden_size: 768,
      initializer_range: 0.02,
      intermediate_size: 3072,
      max_position_embeddings: 2048,
      model_type: "llama",
      num_attention_heads: 12,
      num_hidden_layers: 12,
      num_key_value_heads: 12,
      pad_token_id: 0,
      pretraining_tp: 1,
      rms_norm_eps: 1e-6,
      rope_scaling: null,
      rope_theta: 10000.0,
      tie_word_embeddings: false,
      transformers_version: "4.35.2",
      use_cache: true,
      vocab_size: 32000,
    }),
    { model: session },
    {
      pad_token_id: 0,
      bos_token_id: 1,
      eos_token_id: 2,
    }
  );

  statusEl.remove();

  buildChat();
  context = { tokenizer, model };
};

/**
 * @param {import("@xenova/transformers").PreTrainedTokenizer} tokenizer
 * @param {LlamaForCausalLM} model
 * @param {string} input
 */
const generate = async function* (tokenizer, model, input) {
  const pipe = createGeneratorPipe();
  console.log(input);

  // @ts-ignore
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
      pipe.callback(tokenizer.batch_decode([output]));
    },
    end() {
      pipe.end();
    },
  };
  model.generate({
    ...encoded,
    streamer,
    max_new_tokens: 500,
    repetition_penalty: 1.25,
    penalty_alpha: 0.5,
    top_k: 4,
  });
  for await (const item of pipe.iterate()) {
    yield item;
  }
};

const buildChat = () => {
  const root = document.createElement("div");
  root.classList.add("messages");
  const input = document.createElement("input");
  input.placeholder = "Type some big content.";

  input.addEventListener("keydown", async (e) => {
    if (e.key != "Enter") return;
    e.preventDefault();

    const value = input.value;
    input.value = "";

    const inputEl = document.createElement("div");
    inputEl.classList.add("user");
    root.appendChild(inputEl);
    const outputEl = document.createElement("div");
    outputEl.classList.add("assistant");
    root.appendChild(outputEl);

    inputEl.textContent = value;

    for await (const output of generate(
      context.tokenizer,
      context.model,
      `<|im_start|>system
You are a helpful assistant who answers user's questions with truth, details, and curiosity.<|im_end|>
<|im_start|>user
Explain ${value} using simple language.<|im_end|>
<|im_start|>assistant
`
    )) {
      outputEl.textContent = output;
    }
    // const userEl = document.createElement("div");
    // userEl.textContent = value;
    // userEl.classList.add("user");
    // root.appendChild(userEl);

    // const responseEl = document.createElement("div");
    // responseEl.classList.add("response");
    // root.appendChild(responseEl);

    // const userQ = { role: "user", content: value };
    // const responseQ = { role: "assistant", content: "" };
    // convo.push(userQ);
    // console.log(convo);

    // for await (const output of generate(
    //   context.tokenizer,
    //   context.model,
    //   convo,
    // )) {
    //   responseEl.textContent = output;
    //   responseQ.content = output;
    // }

    // convo.push(responseQ);
  });

  document.body.appendChild(root);
  document.body.appendChild(input);
};

window.onerror = (e) => alert(`error: ${e}`);
