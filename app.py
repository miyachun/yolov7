import torch
import gradio as gr


model = torch.hub.load('./', 'custom', 'best.pt',force_reload=True, source='local',trust_repo=True)
examples=[["photo/a.jpg","Image1"],["photo/b.jpg","Image2"],
          ["photo/c.jpg","Image3"],["photo/d.jpg","Image4"],
          ["photo/e.jpg","Image5"],["photo/f.jpg","Image6"],
          ["photo/g.jpg","Image7"],["photo/h.jpg","Image8"]]

io=gr.Interface(fn=lambda img:model(img).render()[0],inputs=["image"],title="Yolov7 Custom Object Detection",examples=examples,outputs=["image"],).launch()
io.launch()