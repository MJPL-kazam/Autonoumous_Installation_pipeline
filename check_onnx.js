const fs = require('fs');

function readONNX(filename) {
  const buffer = fs.readFileSync(filename);
  // Extremely hacky way to find dimension values in the binary file
  // YOLOv8 often has strings like "images", "output0"
  // Let's just find "output0" and look around it
  const idx = buffer.indexOf(Buffer.from('output0'));
  if (idx > -1) {
    console.log("Found output0 at", idx);
    const slice = buffer.slice(idx, idx + 100);
    console.log("Hex:", slice.toString('hex'));
    console.log("ASCII:", slice.toString('ascii').replace(/[^a-zA-Z0-9]/g, '.'));
  } else {
    console.log("output0 not found");
  }
}
readONNX('public/t400v100.onnx');
