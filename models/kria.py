import numpy as np
import torch
from pynq_dpu import DpuOverlay

from utils.KriaDecoder import Post_Process

anchor_list = [
    [1.25, 1.6525, 2, 3.75, 4.125, 2.875],  # divided by 8
    [1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375],  # divided by 16
    [3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875],  # divided by 32
]


class KRIA(torch.nn.Module):
    def __init__(self, xmodel_path, class_names):
        """Initialize KRAI device."""
        super().__init__()
        self.class_names = class_names

        self.overlay = DpuOverlay("dpu.bit")
        self.overlay.load_model(xmodel_path)
        self.model = self.overlay.runner
        inputTensors = self.model.get_input_tensors()
        outputTensors = self.model.get_output_tensors()
        self.shapeIn = tuple(inputTensors[0].dims)
        shapeOut0 = tuple(outputTensors[0].dims)
        shapeOut1 = tuple(outputTensors[1].dims)
        shapeOut2 = tuple(outputTensors[2].dims)

        self.input_data = [np.empty(self.shapeIn, dtype=np.int8, order="C")]
        self.output_data = [
            np.empty(shapeOut0, dtype=np.int8, order="C"),
            np.empty(shapeOut1, dtype=np.int8, order="C"),
            np.empty(shapeOut2, dtype=np.int8, order="C"),
        ]
        self.image = self.input_data[0]
        self.post_process = Post_Process(anchors=anchor_list, nc=len(self.class_names))

    def forward(self, batch: torch.Tensor):
        """Make a prediction on test images.

        Args:
            batch: A batch of test images, with dimension (B, D, h, w).

        Returns:
            score_map: A tensor with the patch level scores, with dimension (B, H, W).
        """
        with torch.no_grad():
            batch = batch.cpu().detach().numpy()
        # batch = np.transpose(batch, (0, 2, 3, 1))
        batch = batch.astype(np.float32) * (2**4)

        if len(batch.shape) == 3:
            batch = batch[None]  # expand for batch dim

        self.image[0, ...] = batch.reshape(self.shapeIn[1:])

        job_id = self.model.execute_async(self.input_data, self.output_data)
        self.model.wait(job_id)

        kria_outputs = []
        for i, t in enumerate(self.output_data):
            if i == 2:
                kria_outputs.append(torch.as_tensor(t.astype(np.float32) / (2**2)))
            else:
                kria_outputs.append(torch.as_tensor(t.astype(np.float32) / (2**3)))
        kria_outputs = self.post_process.post_process(kria_outputs)

        return kria_outputs
