Sashimi is a multimodal model that can take text, images, video, audio, 3d meshes as inputs
at the byte level and output text, images and 3d meshes. With a self adaptive transformer^2 structure and surprise based memory as a layer using titans_pytorch. With chains of thought that can use image and 3d meshes as context.


This model is trained by DeepseekV3 using model distillation. Then trained by deepseek-r1 for CoT. And then finetuned for programming.






