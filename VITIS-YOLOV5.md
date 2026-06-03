#Quantize and compiled a YOLOV5 with VITIS-AI

# Quantize

In your WSL run:

- `git clone -b v1.4.1 --recurse-submodules https://github.com/Xilinx/Vitis-AI`
- `docker pull xilinx/vitis-ai-cpu:latest`
- `cd Vitis-AI`
- `./docker_run.sh xilinx/vitis-ai-cpu:latest`
- ` conda activate vitis-ai-pytorch`
- `https://github.com/ultralytics/yolov5.git`

Before quantizing the YOLOV5 model some changes have to be made.
You need to modify the '`forward`' function of `Detect class` in the yolo. py script:

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        return x

The SiLU function is not supported (as seen here https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Currently-Supported-Operators). You need to replace it with LeakyReLU. The specific files that need to be modified are the **common.py** and **experimental.py** files, located in yolov5/models/, which are modified as follows.

    # old
    nn.SiLU()

    #new
    nn.LeakyReLU(0.1, inplace=True)

⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️

<font color='red'>**After these changes you have to train this modified network**</font>

⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️

Now it should be ready for quantization, create a **quant.py** file in YOLOV5 folder

`

    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device
    from pytorch_nndct.apis import torch_quantizer
    import torch

    from argparse import ArgumentParser

    def get_parser() -> ArgumentParser:

        parser = ArgumentParser()
        parser.add_argument("--model", default="runs/train/exp/weights/best.pt", help="the pt model")

        parser.add_argument('--quant_mode', default='calib', choices=['calib', 'test'], help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
        parser.add_argument('--output_dir', default='./compiled_xmodel', help='save xmodel')
        parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])

        return parser

    def quant(args):
        model = DetectMultiBackend(weights=args.model)
        device = torch.device(args.device)
        rand_in = torch.randn(1, 3, 320, 320)

        quantizer = torch_quantizer(args.quant_mode, model, rand_in)#, device=device)#, quant_config_file=None, target='DPUCZDX8G_ISA1_B4096')
        quantized_model = quantizer.quant_model
        quantized_model = quantized_model.to(device)

        quantized_model.eval()
        results = quantized_model(rand_in)

        if args.quant_mode == 'calib':
            quantizer.export_quant_config()
        elif args.quant_mode == 'test':
            quantizer.export_xmodel(output_dir =args.output_dir , deploy_check=True)

    if __name__ == "__main__":
        args = get_parser().parse_args()
        quant(args)

`

**Quantization 1**
with `quant_mode =  calib`
In the terminal after activating the vitis-ai-pytorch

> > `python quant.py --quant_mode calib`

![quan.png](./quan.png)

**Generating xmodel**
To export_xmodel, change `quant_mode = 'test'`

> > `python quant.py --quant_mode test`

![quant_test.png](./quant_test.png)

As output, you will have quantize_result folder indluging all relevant files:

**Compiling**
This step is needed if you want to run the xmodel on the device

As you can see the last line of the screenshot:
<font color='green'>**[VAIQ_NOTE]: =>Successfully convert 'DetectMultiBackend' to xmodel.(./compiled_xmodel/DetectMultiBackend_int.xmodel)**</font>

> > `vai_c_xir -x ./compiled_xmodel/DetectMultiBackend_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -o OUTPUTPATH -n yolov5_cpu`

⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️

**<font color='red'>**Pay attention**</font>** to whether the final DPU subgraph number is 1. If it is not 1, please check whether your model has an OP that is not supported by the DPU. When encountering an OP that is not supported by the DPU, the DPU will be divided into multiple subgraphs for execution, and will be executed by PS. After processing, it is sent to the DPU, which slows down efficiency. The generated xmodel can be used to view the network input and output structure using netron.
⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️

![subgraph.png](./subgraph.png)
