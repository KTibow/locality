import { createGeneratorPipe } from "./_queue";

const getBuffer = async () => {
  const [fileH] = await window.showOpenFilePicker();
  const fileO = await fileH.getFile();
  return await fileO.arrayBuffer();
};

const button1 = document.querySelector("#button1");
button1.addEventListener("click", async () => {
  const buffer = await getBuffer();
  initWorker(buffer);
  button1.remove();
});

const initWorker = (buffer) => {
  const worker = new Worker("./worker.js", { type: "module" });

  worker.postMessage({ type: "init", buffer }, [buffer]);
  const go = () =>
    buildChat((input) => {
      const id = crypto.randomUUID();
      const pipe = createGeneratorPipe();

      worker.postMessage({
        type: "generate",
        id,
        input: `<|im_start|>system
You are a helpful assistant who answers user's questions with truth, details, and curiosity.<|im_end|>
<|im_start|>user
Explain ${input} in simple language.<|im_end|>
<|im_start|>assistant
`,
      });
      worker.addEventListener("message", ({ data }) => {
        switch (data.type) {
          case "output": {
            if (data.id != id) return;
            pipe.callback(data.output);
            break;
          }
          case "end": {
            if (data.id != id) return;
            pipe.end();
            break;
          }
        }
      });
      return pipe.iterate();
    });
  const handler = ({ data }) => {
    if (data.type == "ready") {
      go();
      worker.removeEventListener("message", handler);
    }
  };
  worker.addEventListener("message", handler);
};

const buildChat = (generate) => {
  const root = document.querySelector(".messages");

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
    for await (const output of generate(value)) {
      outputEl.textContent = output;
    }
  });

  document.body.appendChild(root);
  document.body.appendChild(input);
};

window.onerror = (e) => alert(`error: ${e}`);
