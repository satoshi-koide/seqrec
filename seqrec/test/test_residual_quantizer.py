import torch
import seqrec.module.rqvae as rqvae

def test_codebook():
    code_size = 10
    embedding_dim = 18
    batch_size = 5

    quantizer = rqvae.Quantizer(code_size, embedding_dim, beta=0.25)
    input_vectors = torch.randn(batch_size, embedding_dim)

    # Initialize codebook with data samples
    with torch.no_grad():
        quantizer.init_codebooks(torch.cat([input_vectors, torch.randn(10, embedding_dim)], dim=0))

    gumbel_t = 0.0001

    output = quantizer(input_vectors, temperature=gumbel_t)
    assert output.quantized.shape == (batch_size, embedding_dim)
    assert output.indices is None
    assert isinstance(output.loss, torch.Tensor)
    print("Embedding (training mode):", output.quantized)    

    quantizer.eval()
    output_eval = quantizer(input_vectors, temperature=gumbel_t)
    assert output_eval.quantized.shape == (batch_size, embedding_dim)
    assert output_eval.indices.shape == (batch_size,)
    assert isinstance(output_eval.loss, torch.Tensor)
    print("Indices (eval mode):", output_eval.indices)
    print("Embedding (eval mode):", output_eval.quantized)
    print("CodeBook test passed.")

def test_residual_quantizer():
    code_sizes = [10, 10, 10]  # 3 codebooks with 10 entries each
    embedding_dim = 200
    batch_size = 5

    rq = rqvae.ResidualQuantizer(code_sizes, embedding_dim)
    input_vectors = torch.randn(batch_size, embedding_dim)

    gumbel_t = 0.0001

    output = rq(input_vectors, temperature=gumbel_t)
    assert output.quantized.shape == (batch_size, embedding_dim)
    assert output.indices is None
    assert isinstance(output.loss, torch.Tensor)
    print("Embedding (training mode):", output.quantized)

    rq.eval()
    output_eval = rq(input_vectors, temperature=gumbel_t)
    assert output_eval.quantized.shape == (batch_size, embedding_dim)
    assert output_eval.indices.shape == (batch_size, len(code_sizes))   
    assert isinstance(output_eval.loss, torch.Tensor)
    print("Embedding (eval mode):", output_eval.quantized)

    print("ResidualQuantizer test passed.")

if __name__ == "__main__":
    test_codebook()
    #test_residual_quantizer()