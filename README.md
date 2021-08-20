# CNN
Crack detection is important for the inspection and evaluation during the maintenance of concrete structures. However, conventional image-based methods need extract crack features using complex image preprocessing techniques, so it can lead to challenges when concrete surface contains various types of noise due to extensively varying real-world situations such as thin cracks, rough surface, shadows, etc. To overcome these challenges, this paper proposes an image-based crack detection method using a deep convolutional neural network (CNN). A CNN is designed through modifying AlexNet and then trained and validated using a built database with 60000 images. Through comparing validation accuracy under different base learning rates, 0.01 was chosen as the best base learning rate with the highest validation accuracy of 99.06%, and its training result is used in the following testing process. The robustness and adaptability of the trained CNN are tested on 205 images with 3120 × 4160 pixel resolutions which were not used for training and validation. The trained CNN is integrated into a smartphone application to mobile more public to detect cracks in practice. The results confirm that the proposed method can indeed detect cracks in images from real concrete surfaces.
![1_ZB6H4HuF58VcMOWbdpcRxQ](https://user-images.githubusercontent.com/70472055/130177398-993a26e0-2d7a-42bf-b08f-c122d1680c4e.png)
