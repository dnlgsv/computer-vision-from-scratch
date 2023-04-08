# computer-vision-from-scratch

From zero to one, it's normal

We want to use:

- tests
- pytorch 2.0
- python >= 3.10

SWIN or VOLO

ViT algorithm:

1. Devide an image into patches:
2. Flatten every patch
3. Pass every patch to ist own MLP
4. Take everything from step 3 and add positional ebmbedding
5. pass it to transformer encoder
6. MLP Head
7. Softmax?

Transformer encoder: 0. Input

1. Layer Normalization
2. Multy Head Attention
3. Step 2 output + step 0 output
4. Layer Normalization
5. MLP
6. Step 3 output + step 5 output
