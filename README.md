# computer-vision-from-scratch

From zero to one, it's normal

ViT algorithm:
The overall architecture can be described easily in five simple steps below:

1. Split an input image into patches.
2. Get linear embeddings (representation) from each patch referred to as Patch Embeddings.
3. Add position embeddings and a [cls] token to each of the Patch Embeddings.
4. Pass through a Transformer Encoder and get the output values for each of the [cls] tokens.
5. Pass the representations of [cls] tokens through a MLP Head to get final class predictions.

Again, but more straightforward:

1. Devide an image into patches
2. Flatten every patch
3. Pass every patch to ist own MLP
4. Take everything from step 3 and add positional ebmbedding
5. pass it to transformer encoder
6. MLP Head
7. Softmax?

Transformer encoder:

0. Input
1. Layer Normalization
2. Multy Head Attention
3. Step 2 output + step 0 output
4. Layer Normalization
5. MLP
6. Step 3 output + step 5 output
