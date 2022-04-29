# Export the generator (the thing we care about to ONNX for ONNX Runtime Support

from config import *
import torch.onnx
import onnx
import onnxruntime
import numpy as np

def main() -> None:
    # Load model weights.
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    # Start the verification mode of the model.
    model.eval()

    batch_size = 1
    x = torch.randn(batch_size, 3, 96, 96, requires_grad=True)
    gpu_x = x.to(device)
    torch_out = model(gpu_x)


    # Export the model
    torch.onnx.export(model,               # model being run
                  gpu_x,                         # model input (or a tuple for multiple inputs)
                  "srgan_generator.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

    # Check the model
    onnx_model = onnx.load("srgan_generator.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("srgan_generator.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == "__main__":
    main()




