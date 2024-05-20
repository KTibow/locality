export function createGeneratorPipe() {
  let iterator;
  let done = false;
  let queue = [];

  const iterate = async function* () {
    while (!done) {
      await new Promise((resolve) => {
        iterator = resolve;
      });
      if (queue.length > 0) yield queue.shift();
      else break;
    }
  };

  const callback = (value) => {
    queue.push(value);
    if (iterator) {
      iterator();
      iterator = null;
    }
  };

  const end = () => {
    done = true;
    if (iterator) {
      iterator();
      iterator = null;
    }
  };

  return { iterate, callback, end };
}
