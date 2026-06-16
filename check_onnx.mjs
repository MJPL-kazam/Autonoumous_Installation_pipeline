import * as ort from 'onnxruntime-node';

async function run() {
  try {
    const session = await ort.InferenceSession.create('public/model/t400v100.onnx');
    
    // Create a dummy tensor of 1x3x640x640 with ones
    const dims = [1, 3, 640, 640];
    const size = dims.reduce((a, b) => a * b);
    const data = new Float32Array(size);
    data.fill(0.5);
    const tensor = new ort.Tensor('float32', data, dims);
    
    const results = await session.run({ [session.inputNames[0]]: tensor });
    const output = results[session.outputNames[0]];
    
    console.log('Output dims:', output.dims);
    console.log('First box:', Array.from(output.data.slice(0, 14)));
    
  } catch (e) {
    console.error(e);
  }
}
run();
