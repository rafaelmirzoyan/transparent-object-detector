# 🫧 transparent-object-detector
Deep-learning–based transparent object segmentation system for robotic and drone perception, enabling reliable detection of glass, reflective, and low-texture objects in complex environments.

# 📖 Overview  
This project implements a computer vision pipeline for detecting and segmenting transparent and reflective objects using deep neural networks. Leveraging YOLOv8 instance segmentation, the system learns to identify objects that are traditionally difficult to perceive due to refraction, reflection, and lack of texture, making it suitable for autonomous drones, mobile robots, and perception research.

# 🧠 Features  
• Instance segmentation of transparent objects  
• YOLOv8-based deep learning architecture  
• Pixel-mask → polygon label conversion pipeline  
• Image & video inference support  
• Designed for robotic and aerial perception tasks  

# 📐 Method  
• Train a YOLOv8 segmentation model on transparent-object datasets  
• Convert pixel-level masks into YOLO polygon annotations  
• Learn object boundaries under reflection and refraction  
• Output segmentation masks for downstream navigation or planning  
